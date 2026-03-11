"""
Mixture of Experts (MoE): a router network learns to dispatch each token to a subset of
specialist MLPs, scaling model capacity without proportionally scaling compute.
"""
# Reference: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
# Mixture-of-Experts Layer" (2017). https://arxiv.org/abs/1701.06538
# Also: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with
# Simple and Efficient Sparsity" (2021). https://arxiv.org/abs/2101.03961
# Architecture reuses the microgpt embedding/LM-head pattern (Radford et al., 2019) with
# the transformer block replaced by a routed MoE layer.

# === TRADEOFFS ===
# + Scales model capacity without proportionally scaling compute (sparse activation)
# + Specialized experts can develop domain-specific knowledge
# + Compatible with standard transformer architectures (drop-in MLP replacement)
# - Load balancing is hard: experts tend to collapse without auxiliary losses
# - Routing decisions add latency and complicate batched inference
# - Total parameter count (and memory) scales with expert count even if compute doesn't
# WHEN TO USE: Scaling language models beyond dense compute budgets, or when
#   the task naturally decomposes into subtasks that benefit from specialization.
# WHEN NOT TO: Small-scale models where dense layers are sufficient, or
#   latency-sensitive serving where routing overhead is unacceptable.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture
N_EMBD = 8           # embedding dimension — smaller than microgpt (16) because MoE adds
                      # capacity through expert count rather than wider representations
N_EXPERTS = 4        # number of expert MLPs
TOP_K = 2            # experts selected per token — top-2 is the standard MoE choice;
                      # top-1 (Switch Transformer) is simpler but less robust to routing errors
EXPERT_HIDDEN = 16   # hidden dim within each expert MLP (2x expansion from N_EMBD)
BLOCK_SIZE = 12      # context window length

# Training parameters
LEARNING_RATE = 0.01
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
NUM_STEPS = 800
AUX_LOSS_COEFF = 0.1  # weight for load balancing auxiliary loss — controls the tradeoff
                       # between language modeling quality and even expert utilization.
                       # Too low: router collapses to 1-2 experts. Too high: forces uniform
                       # routing that prevents specialization. 0.1 is the standard starting point.

# Data
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: ~2,000 parameters total. Production MoE models (Mixtral-8x7B, Switch-C) have
# billions of parameters across hundreds of experts. The routing algorithm is identical;
# only the expert size and count differ. Our 4-expert, top-2 setup captures the full
# dynamic: router training, load balancing, expert specialization, and sparse activation.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """Download and parse the training corpus."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    with open(filename, "r") as f:
        docs = [line.strip() for line in f if line.strip()]

    return docs


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Every forward operation records its local derivative (dout/dinput). backward()
    replays the computation graph in reverse topological order, accumulating gradients
    via the chain rule: dLoss/dx = sum over paths (product of local gradients along path).
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()) -> None:
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent: float) -> Value:
        return Value(self.data ** exponent, (self,), (exponent * self.data ** (exponent - 1),))

    def __neg__(self) -> Value: return self * -1
    def __radd__(self, other: float) -> Value: return self + other
    def __sub__(self, other: Value | float) -> Value: return self + (-other)
    def __rsub__(self, other: float) -> Value: return other + (-self)
    def __rmul__(self, other: float) -> Value: return self * other
    def __truediv__(self, other: Value | float) -> Value: return self * (other ** -1)
    def __rtruediv__(self, other: float) -> Value: return other * (self ** -1)

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self) -> Value:
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self) -> Value:
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self) -> None:
        """Reverse-mode autodiff via topological sort of the computation graph."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# --- AUTOGRAD DIFFERENCES IN THIS SCRIPT ---
# This Value class follows the canonical interface (see docs/autograd-interface.md)
# with no additions. The router uses Value objects for automatic differentiation.
# Expert MLPs use plain floats with manual gradient computation.
# See docs/autograd-interface.md for the full canonical interface.

# IMPLEMENTATION NOTE: Experts use plain floats (not autograd Value objects) for
# runtime tractability. The router uses scalar autograd because routing decisions
# are the core MoE mechanism — gradients must flow through the gating function.
# Production MoE frameworks (Mixtral, Switch Transformer) vectorize everything;
# we split the approach to stay within pure-Python runtime constraints.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize weight matrix ~ N(0, std). Standard deviation 0.08 is empirically tuned
    for this tiny model; larger models use Xavier/Glorot scaling (std = 1/sqrt(d_in))."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def make_float_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[float]]:
    """Initialize a plain-float weight matrix for expert MLPs."""
    return [[random.gauss(0, std) for _ in range(ncols)] for _ in range(nrows)]


def init_expert_weights() -> list[dict[str, list[list[float]]]]:
    """Initialize 4 independent expert MLPs, each with its own weights.

    Each expert is a 2-layer MLP: input (N_EMBD) -> hidden (EXPERT_HIDDEN) -> output (N_EMBD).
    Experts start with different random weights so they can specialize on different input
    patterns during training. If all experts started identically, the router would have no
    reason to prefer one over another, and symmetry breaking would depend entirely on noise.
    """
    experts = []
    for _ in range(N_EXPERTS):
        # w1: [EXPERT_HIDDEN, N_EMBD] projects input to hidden dimension
        # w2: [N_EMBD, EXPERT_HIDDEN] projects hidden back to embedding dimension
        expert = {
            'w1': make_float_matrix(EXPERT_HIDDEN, N_EMBD),
            'w2': make_float_matrix(N_EMBD, EXPERT_HIDDEN),
        }
        experts.append(expert)
    return experts


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiply: y = W @ x. For W of shape [n_out, n_in] and x of shape
    [n_in], output y has shape [n_out] where y[i] = sum_j W[i,j] * x[j]."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """Stable softmax: subtract max before exp to prevent overflow.
    softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))"""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """RMS normalization: x / sqrt(mean(x^2) + eps).
    Simpler than LayerNorm (no mean centering, no learned affine). Used in LLaMA, Gemma."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """Clipped log for numerical stability. Prevents log(0) = -inf which would break
    gradient propagation. The node is built manually with prob as its child so
    gradients flow back through the computation graph (not severed by clamping)."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === EXPERT FORWARD PASS (PLAIN FLOATS) ===

def expert_forward_float(x: list[float], weights: dict[str, list[list[float]]]) -> list[float]:
    """Single expert MLP forward pass: x -> hidden (ReLU) -> output. All plain floats.

    Math: hidden = ReLU(W1 @ x), output = W2 @ hidden
    This is a standard 2-layer MLP. Each expert learns a different input-to-output mapping,
    so the MoE layer has 4x the capacity of a single MLP — but only activates 2 experts
    per token, keeping compute at 2x (not 4x) of a single expert.
    """
    w1 = weights['w1']
    w2 = weights['w2']

    # Hidden layer: W1 @ x with ReLU activation
    hidden = [sum(w1[i][j] * x[j] for j in range(len(x))) for i in range(len(w1))]
    hidden = [max(0.0, h) for h in hidden]  # ReLU

    # Output layer: W2 @ hidden (projects back to embedding dimension)
    output = [sum(w2[i][j] * hidden[j] for j in range(len(hidden))) for i in range(len(w2))]
    return output


def expert_backward_float(
    x: list[float],
    weights: dict[str, list[list[float]]],
    output_grads: list[float],
    lr: float,
) -> None:
    """Manual gradient computation and weight update for a single expert MLP.

    When the expert's output is wrapped as Value objects and multiplied by router scores,
    backward() sets .grad on those Value wrappers. We extract those gradients as output_grads
    and manually propagate them through the expert's plain-float layers.

    Chain rule through the expert MLP:
        d(loss)/d(w2[i][j]) = output_grads[i] * hidden[j]
        d(loss)/d(hidden[j]) = sum_i(output_grads[i] * w2[i][j]) * relu_grad(pre_relu[j])
        d(loss)/d(w1[i][j]) = hidden_grads[i] * x[j]

    This is standard backpropagation — the same algorithm the Value class automates, but
    done manually here for the plain-float expert weights.
    """
    w1 = weights['w1']
    w2 = weights['w2']

    # --- Recompute forward pass to get intermediate activations ---
    pre_relu = [sum(w1[i][j] * x[j] for j in range(len(x))) for i in range(len(w1))]
    hidden = [max(0.0, h) for h in pre_relu]

    # --- Backward through W2: output = W2 @ hidden ---
    # d(loss)/d(w2[i][j]) = output_grads[i] * hidden[j]
    for i in range(len(w2)):
        for j in range(len(w2[i])):
            w2[i][j] -= lr * output_grads[i] * hidden[j]

    # --- Backward through ReLU into hidden layer ---
    # d(loss)/d(hidden[j]) = sum_i(output_grads[i] * w2[i][j])
    # d(loss)/d(pre_relu[j]) = d(loss)/d(hidden[j]) * (1 if pre_relu[j] > 0 else 0)
    hidden_grads = [0.0] * len(w1)
    for j in range(len(hidden)):
        for i in range(len(w2)):
            hidden_grads[j] += output_grads[i] * w2[i][j]
        # ReLU gradient: pass through if pre-activation was positive, zero otherwise
        if pre_relu[j] <= 0:
            hidden_grads[j] = 0.0

    # --- Backward through W1: pre_relu = W1 @ x ---
    # d(loss)/d(w1[i][j]) = hidden_grads[i] * x[j]
    for i in range(len(w1)):
        for j in range(len(w1[i])):
            w1[i][j] -= lr * hidden_grads[i] * x[j]


# === MOE FORWARD PASS ===

def moe_forward(
    token_id: int,
    pos_id: int,
    params: dict[str, list[list[Value]]],
    expert_weights: list[dict[str, list[list[float]]]],
) -> tuple[list[Value], list[Value], list[int], list[float]]:
    """Forward pass through the MoE model for a single token.

    Architecture:
        1. Embed token (token + position embeddings)
        2. RMSNorm
        3. Router: linear projection to N_EXPERTS scores, softmax to probabilities
        4. Select top-K experts by router probability
        5. Run selected experts (plain floats), wrap outputs as Value objects
        6. Weighted sum of expert outputs using router scores
        7. LM head: project to vocabulary logits

    Returns:
        logits: vocabulary-sized logit vector (Value objects)
        router_probs: full router probability distribution (Value objects, for aux loss)
        selected_experts: indices of the top-K experts chosen
        x_float: input to experts as plain floats (cached for backward pass)
    """
    # --- Token embedding ---
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # --- Router: decide which experts process this token ---
    # The router is a simple linear layer followed by softmax. It maps the token's
    # representation to a probability distribution over experts.
    # Math: router_probs = softmax(W_router @ x)
    # Gradients flow through this softmax via the Value class, so the router learns
    # which experts are best for which tokens.
    router_logits = linear(x, params['w_router'])
    router_probs = softmax(router_logits)

    # --- Top-K expert selection ---
    # Select the K experts with the highest router probabilities.
    # Sparse activation is the defining feature of MoE: we have N_EXPERTS worth of
    # parameters but only compute TOP_K expert forward passes per token. This is how
    # MoE achieves the "scale capacity without scaling compute" property.
    scored = [(router_probs[i].data, i) for i in range(N_EXPERTS)]
    scored.sort(reverse=True)
    selected_experts = [idx for _, idx in scored[:TOP_K]]

    # --- Renormalize selected expert scores ---
    # After selecting top-K, renormalize their probabilities to sum to 1.
    # This ensures the weighted combination is properly scaled regardless of how much
    # probability mass the unselected experts had.
    selected_scores = [router_probs[i] for i in selected_experts]
    score_sum = sum(s.data for s in selected_scores)
    if score_sum > 1e-10:
        norm_scores = [s / score_sum for s in selected_scores]
    else:
        norm_scores = [s for s in selected_scores]

    # --- Expert computation (plain floats) ---
    # Extract the Value-based representation as plain floats for expert MLPs.
    # After experts compute their outputs, we wrap the results back as Value objects
    # and multiply by router scores — this creates the gradient bridge between the
    # autograd router and the plain-float experts.
    x_float = [v.data for v in x]

    # --- Weighted combination of expert outputs ---
    # Math: output = sum_i(score_i * expert_i(x)) for i in selected experts
    # Each selected expert processes the same input independently, then their outputs
    # are blended using the (renormalized) router probabilities as weights.
    combined = [Value(0.0)] * N_EMBD
    for k_idx, expert_idx in enumerate(selected_experts):
        expert_out = expert_forward_float(x_float, expert_weights[expert_idx])

        # Wrap expert output as Value objects so multiplication by the router score
        # (a Value) creates a computation graph node. After backward(), the Value
        # wrappers accumulate d(loss)/d(expert_output), which we use to manually
        # update the expert weights.
        for j in range(N_EMBD):
            expert_val = Value(expert_out[j])
            combined[j] = combined[j] + norm_scores[k_idx] * expert_val

    # --- LM head: project to vocabulary ---
    logits = linear(combined, params['lm_head'])
    return logits, router_probs, selected_experts, x_float


# === LOAD BALANCING AUXILIARY LOSS ===

def compute_aux_loss(
    expert_assignment_counts: list[int],
    router_prob_sums: list[float],
    total_tokens: int,
) -> Value:
    """Compute the load balancing auxiliary loss for the current training step.

    Without this loss, the router collapses: it learns to send all tokens to 1-2 experts
    (whichever happen to produce slightly lower loss early in training). The unused experts
    never receive gradients and remain at their random initialization — a positive feedback
    loop called "expert collapse" or "rich get richer."

    The auxiliary loss penalizes uneven distribution by multiplying two quantities:
        f_i = fraction of tokens assigned to expert i (binary assignment indicator)
        P_i = average router probability for expert i (soft, continuous signal)

    Math: L_aux = N_EXPERTS * sum_i(f_i * P_i)

    Why the product f_i * P_i? If expert i receives many tokens (high f_i) AND has high
    average probability (high P_i), the product is large and the loss penalizes this.
    The minimum occurs when f_i = P_i = 1/N for all experts (uniform distribution).

    The N_EXPERTS scaling factor makes the loss magnitude roughly comparable across
    different expert counts, so AUX_LOSS_COEFF doesn't need expert-count-specific tuning.
    """
    if total_tokens == 0:
        return Value(0.0)

    aux = Value(0.0)
    for i in range(N_EXPERTS):
        # f_i: fraction of tokens routed to expert i
        f_i = expert_assignment_counts[i] / total_tokens
        # P_i: mean router probability for expert i across all tokens
        p_i = router_prob_sums[i] / total_tokens
        # Product f_i * P_i penalizes experts that are both frequently selected
        # and receive high router probability
        aux = aux + Value(f_i * p_i)

    return aux * N_EXPERTS


# === TRAINING ===

if __name__ == "__main__":
    start_time = time.time()

    # -- Load and prepare data --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} documents")
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # === Initialize Model Parameters ===

    params: dict[str, list[list[Value]]] = {}

    # Token and position embeddings (Value objects)
    params['wte'] = make_matrix(VOCAB_SIZE, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    # Router: linear projection from embedding space to expert scores
    # Shape: [N_EXPERTS, N_EMBD] — one score per expert
    params['w_router'] = make_matrix(N_EXPERTS, N_EMBD)

    # LM head: project MoE output to vocabulary logits
    params['lm_head'] = make_matrix(VOCAB_SIZE, N_EMBD)

    # Expert MLPs (plain floats — not tracked by autograd)
    expert_weights = init_expert_weights()

    # -- Count parameters for architecture summary --
    router_params = N_EXPERTS * N_EMBD
    expert_params_each = EXPERT_HIDDEN * N_EMBD + N_EMBD * EXPERT_HIDDEN  # w1 + w2
    expert_params_total = expert_params_each * N_EXPERTS
    embd_params = VOCAB_SIZE * N_EMBD + BLOCK_SIZE * N_EMBD + VOCAB_SIZE * N_EMBD  # wte + wpe + lm_head

    print(f"\n=== MoE Model Architecture ===")
    print(f"Router parameters: {router_params} (Value class autograd)")
    print(f"Expert parameters: {expert_params_each} x {N_EXPERTS} experts = {expert_params_total} (plain floats)")
    print(f"Embedding parameters: {embd_params} (Value class autograd)")
    print(f"Total parameters: {router_params + expert_params_total + embd_params:,}")

    # -- Collect autograd parameters for Adam optimizer --
    autograd_param_list: list[Value] = []
    for matrix in params.values():
        for row in matrix:
            autograd_param_list.extend(row)

    m_state = [0.0] * len(autograd_param_list)
    v_state = [0.0] * len(autograd_param_list)

    # -- Expert utilization tracking --
    # Track which experts are selected across all tokens to detect collapse.
    # A healthy MoE distributes tokens roughly evenly; collapse means 1-2 experts
    # receive the vast majority of assignments.
    cumulative_expert_counts = [0] * N_EXPERTS
    utilization_report_interval = 200

    # Smoothed loss for reporting — individual step losses are noisy because each step
    # trains on a single document. The exponential moving average (alpha=0.05) smooths
    # over ~20 steps, giving a more accurate picture of learning progress.
    smooth_lm_loss = 3.3  # initialize near expected starting loss
    smooth_alpha = 0.05

    # === Training Loop ===
    print(f"\n=== Training ===")

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        # Per-step tracking for auxiliary loss computation
        step_expert_counts = [0] * N_EXPERTS
        step_router_prob_sums = [0.0] * N_EXPERTS
        step_token_count = 0

        losses: list[Value] = []
        for pos in range(seq_len):
            input_token = tokens[pos]
            target_token = tokens[pos + 1]

            logits, router_probs, selected_experts, x_float = moe_forward(
                input_token, pos, params, expert_weights,
            )

            # Track expert utilization
            for eidx in selected_experts:
                step_expert_counts[eidx] += 1
                cumulative_expert_counts[eidx] += 1

            # Track router probabilities for auxiliary loss
            for i in range(N_EXPERTS):
                step_router_prob_sums[i] += router_probs[i].data
            step_token_count += 1

            # Cross-entropy loss: -log P(target)
            probs = softmax(logits)
            loss_t = -safe_log(probs[target_token])
            losses.append(loss_t)

        # -- Compute total loss: LM loss + auxiliary load balancing loss --
        lm_loss = (1.0 / seq_len) * sum(losses)
        aux_loss = compute_aux_loss(step_expert_counts, step_router_prob_sums, step_token_count)
        total_loss = lm_loss + AUX_LOSS_COEFF * aux_loss

        # -- Backward pass through autograd graph --
        total_loss.backward()

        # -- Update autograd parameters (embeddings, router, LM head) with Adam --
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
        for i, param in enumerate(autograd_param_list):
            m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * param.grad
            v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * param.grad ** 2
            m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            param.grad = 0.0

        # -- Update expert weights via manual gradient computation --
        # Autograd cannot reach expert weights (plain floats). We analytically compute
        # d(loss)/d(expert_output) for each token using the standard cross-entropy
        # gradient, then backpropagate through each expert MLP manually.
        #
        # Gradient path: loss -> softmax -> logits -> lm_head -> combined -> score * expert_out
        # The cross-entropy gradient d(-log softmax(z)[t])/d(z[i]) = softmax(z)[i] - 1{i==t}
        # is well-known and avoids expensive finite differences.
        expert_lr = lr_t * 0.5  # lower LR for experts — SGD is noisier than Adam

        # Cache LM head as floats once per step (constant across positions)
        lm_head_float = [[v.data for v in row] for row in params['lm_head']]

        for pos in range(seq_len):
            input_token = tokens[pos]
            target_token = tokens[pos + 1]

            # Re-run partial forward to recover router decisions and expert inputs
            tok_emb = params['wte'][input_token]
            pos_emb = params['wpe'][pos]
            x = [t + p for t, p in zip(tok_emb, pos_emb)]
            x = rmsnorm(x)

            router_logits = linear(x, params['w_router'])
            router_probs_re = softmax(router_logits)
            scored = [(router_probs_re[i].data, i) for i in range(N_EXPERTS)]
            scored.sort(reverse=True)
            selected = [idx for _, idx in scored[:TOP_K]]

            selected_scores_data = [router_probs_re[i].data for i in selected]
            score_sum = sum(selected_scores_data)
            if score_sum > 1e-10:
                norm_score_data = [s / score_sum for s in selected_scores_data]
            else:
                norm_score_data = selected_scores_data

            x_float_re = [v.data for v in x]

            # Run each selected expert and compute the combined output
            expert_outputs: dict[int, list[float]] = {}
            for eidx in selected:
                expert_outputs[eidx] = expert_forward_float(x_float_re, expert_weights[eidx])

            combined_float = [0.0] * N_EMBD
            for k_idx, eidx in enumerate(selected):
                for j in range(N_EMBD):
                    combined_float[j] += norm_score_data[k_idx] * expert_outputs[eidx][j]

            # Compute softmax(logits) for the cross-entropy gradient
            logits_float = [
                sum(lm_head_float[i][j] * combined_float[j] for j in range(N_EMBD))
                for i in range(VOCAB_SIZE)
            ]
            max_logit = max(logits_float)
            exp_logits = [math.exp(l - max_logit) for l in logits_float]
            sum_exp = sum(exp_logits)
            probs_float = [e / sum_exp for e in exp_logits]

            # d(loss)/d(logits[i]) = softmax(logits)[i] - 1{i == target}
            d_logits = [probs_float[i] - (1.0 if i == target_token else 0.0)
                        for i in range(VOCAB_SIZE)]

            # d(loss)/d(combined[j]) = sum_i d(loss)/d(logits[i]) * lm_head[i][j]
            d_combined = [0.0] * N_EMBD
            for j in range(N_EMBD):
                for i in range(VOCAB_SIZE):
                    d_combined[j] += d_logits[i] * lm_head_float[i][j]
                d_combined[j] /= seq_len  # scale to match averaged LM loss

            # Chain through the weighted combination: d(loss)/d(expert_out) = d_combined * score
            for k_idx, eidx in enumerate(selected):
                d_expert_out = [d_combined[j] * norm_score_data[k_idx] for j in range(N_EMBD)]
                expert_backward_float(
                    x_float_re, expert_weights[eidx], d_expert_out, expert_lr,
                )

        # -- Update smoothed loss --
        smooth_lm_loss = smooth_alpha * lm_loss.data + (1 - smooth_alpha) * smooth_lm_loss

        # -- Logging --
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | lm_loss: {lm_loss.data:.4f} "
                  f"(smooth: {smooth_lm_loss:.4f}) | aux_loss: {aux_loss.data:.4f} | "
                  f"total: {total_loss.data:.4f}")

        if (step + 1) % utilization_report_interval == 0:
            total_assignments = sum(cumulative_expert_counts)
            if total_assignments > 0:
                pcts = [100 * c / total_assignments for c in cumulative_expert_counts]
                pct_str = " ".join(f"E{i}={pcts[i]:.0f}%" for i in range(N_EXPERTS))
                print(f"  step {step + 1:>4}: {pct_str}")

    elapsed_train = time.time() - start_time
    print(f"\nTraining complete. Smoothed LM loss: {smooth_lm_loss:.4f}")
    print(f"Training time: {elapsed_train:.1f}s")

    # === Expert Analysis ===
    print(f"\n=== Expert Analysis ===")
    total_assignments = sum(cumulative_expert_counts)
    print("Final expert utilization:")
    all_above_threshold = True
    for i in range(N_EXPERTS):
        pct = 100 * cumulative_expert_counts[i] / total_assignments if total_assignments > 0 else 0
        print(f"  Expert {i}: {pct:.1f}% of tokens")
        if pct < 10.0:
            all_above_threshold = False

    if all_above_threshold:
        print("\nAll experts receive >10% of tokens (no collapse)")
    else:
        print("\nWARNING: Expert collapse detected — some experts below 10%")

    print(f"Load balancing loss: {aux_loss.data:.4f}")

    # === Generation ===
    TEMPERATURE = 0.7
    NUM_SAMPLES = 15

    print(f"\n=== Generation ===")
    print(f"Generating {NUM_SAMPLES} samples (temperature={TEMPERATURE}):\n")

    for sample_idx in range(NUM_SAMPLES):
        token_id = BOS
        generated: list[str] = []
        experts_used: set[int] = set()

        for pos in range(BLOCK_SIZE):
            # Forward pass for generation (no gradient tracking needed,
            # but we reuse the same function for consistency)
            logits, router_probs, selected, _ = moe_forward(
                token_id, pos, params, expert_weights,
            )
            experts_used.update(selected)

            # Temperature-scaled sampling
            scaled_logits = [logit / TEMPERATURE for logit in logits]
            probs = softmax(scaled_logits)

            token_id = random.choices(
                range(VOCAB_SIZE), weights=[p.data for p in probs]
            )[0]

            if token_id == BOS:
                break
            generated.append(unique_chars[token_id])

        name = ''.join(generated)
        experts_str = ','.join(str(e) for e in sorted(experts_used))
        print(f"  {sample_idx + 1:>2}. {name} (experts used: {experts_str})")

    elapsed_total = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_total:.1f}s")
