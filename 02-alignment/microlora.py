"""
Low-Rank Adaptation (LoRA): fine-tuning a frozen language model by injecting tiny trainable
matrices — proving that weight updates live in a low-dimensional subspace.
"""
# Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).
# https://arxiv.org/abs/2106.09685
# Architecture reuses the microgpt pattern (Radford et al., 2019) with pedagogical
# simplifications: RMSNorm, ReLU, no biases. LoRA adapters applied to Q and V projections.

# === TRADEOFFS ===
# + Trains <1% of parameters while matching full fine-tuning quality
# + Multiple LoRA adapters can share a single frozen base model
# + No inference latency overhead: adapter weights merge into base at deployment
# - Low rank limits expressiveness for tasks requiring large weight changes
# - Choosing which layers to adapt requires experimentation
# - Cannot change the model's fundamental capabilities, only steer behavior
# WHEN TO USE: Fine-tuning large pretrained models for specific tasks when
#   compute or memory is limited. Standard for LLM customization.
# WHEN NOT TO: When the task requires fundamentally new capabilities not present
#   in the base model, or when full fine-tuning is affordable and data is abundant.

from __future__ import annotations

import math
import os
import random
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture — identical to microgpt for direct comparison
N_EMBD = 16         # embedding dimension (d_model)
N_HEAD = 4          # number of attention heads
N_LAYER = 1         # transformer blocks
BLOCK_SIZE = 16     # context window length
HEAD_DIM = N_EMBD // N_HEAD  # 4 dimensions per head

# LoRA hyperparameters
LORA_RANK = 2       # rank of the adaptation matrices (r << d_model)
# Rank 2 means each adapter pair contributes a rank-2 perturbation to the weight matrix.
# Production LoRA typically uses r=4..64. With d_model=16, even r=2 captures meaningful
# structure while keeping the parameter count visibly small for demonstration.

# Training — base model
BASE_LR = 0.01
BASE_STEPS = 800
# Training — LoRA adaptation
LORA_LR = 0.01
LORA_STEPS = 500

# Shared optimizer constants
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8

# Data
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: ~4,200 base parameters. Production models have billions. LoRA's value becomes
# dramatic at scale: adapting a 7B model with r=16 means training ~0.1% of parameters.
# At our toy scale the ratio is less extreme but the mechanism is identical.


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


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# See docs/autograd-interface.md for the full specification.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize weight matrix ~ N(0, std). Standard deviation 0.08 is empirically tuned
    for this tiny model; larger models use Xavier/Glorot scaling (std = 1/sqrt(d_in))."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def make_lora_A(nrows: int, ncols: int) -> list[list[Value]]:
    """Initialize LoRA A matrix ~ N(0, 0.02).
    Small random init ensures the two adapters (A and B) break symmetry. Since B starts
    at zero, the initial LoRA contribution is A @ 0 = 0 regardless of A's values — but
    once B starts learning, A's random directions provide diverse gradient signals."""
    return [[Value(random.gauss(0, 0.02)) for _ in range(ncols)] for _ in range(nrows)]


def make_lora_B(nrows: int, ncols: int) -> list[list[Value]]:
    """Initialize LoRA B matrix to zeros.
    # Math: W_adapted = W_frozen + A @ B
    # At init: A @ B = A @ 0 = 0, so the adapted model is identical to the base model.
    # This is critical: it means LoRA starts from the pretrained solution and makes
    # small perturbations, rather than starting from a random offset that would
    # immediately destroy what the base model learned."""
    return [[Value(0.0) for _ in range(ncols)] for _ in range(nrows)]


def init_parameters(vocab_size: int) -> dict[str, list[list[Value]]]:
    """Initialize all base model parameters: embeddings, attention, MLP, and LM head."""
    params: dict[str, list[list[Value]]] = {}

    params['wte'] = make_matrix(vocab_size, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    for layer_idx in range(N_LAYER):
        params[f'layer{layer_idx}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)

        # MLP: expand 4x then contract (GPT convention for feedforward capacity)
        params[f'layer{layer_idx}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)

    params['lm_head'] = make_matrix(vocab_size, N_EMBD)

    return params


def init_lora_adapters() -> dict[str, list[list[Value]]]:
    """Create LoRA adapter matrices for Q and V attention projections.

    Why Q and V, not K or O? The original LoRA paper (Hu et al., 2021) found that
    adapting Q and V projections captures the most task-relevant information per parameter.
    Intuitively: Q controls "what to look for" and V controls "what to extract" — both
    are highly task-specific. K ("what to advertise") and O ("how to combine") change less
    across tasks. Production LoRA often adapts all four for maximum quality.
    """
    adapters: dict[str, list[list[Value]]] = {}

    for layer_idx in range(N_LAYER):
        # Q adapter: A is (N_EMBD, LORA_RANK), B is (LORA_RANK, N_EMBD)
        # Math: Q_adapted = W_q @ x + A_q @ (B_q @ x)
        #   where A_q @ B_q is a rank-r perturbation to W_q
        adapters[f'layer{layer_idx}.lora_q_A'] = make_lora_A(N_EMBD, LORA_RANK)
        adapters[f'layer{layer_idx}.lora_q_B'] = make_lora_B(LORA_RANK, N_EMBD)

        # V adapter: same structure
        adapters[f'layer{layer_idx}.lora_v_A'] = make_lora_A(N_EMBD, LORA_RANK)
        adapters[f'layer{layer_idx}.lora_v_B'] = make_lora_B(LORA_RANK, N_EMBD)

    return adapters


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiply: y = W @ x. For W of shape [n_out, n_in] and x of shape
    [n_in], output y has shape [n_out] where y[i] = sum_j W[i,j] * x[j]."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def lora_linear(
    x: list[Value],
    w_frozen: list[list[Value]],
    lora_A: list[list[Value]],
    lora_B: list[list[Value]],
) -> list[Value]:
    """LoRA-augmented linear operation: y = W_frozen @ x + A @ (B @ x).

    Math: W_adapted = W_frozen + A @ B  (but we never form this explicitly)
    Instead we compute: base_out = W_frozen @ x     (shape: d_out)
                        lora_mid = B @ x             (shape: r)     -- project to low rank
                        lora_out = A @ lora_mid      (shape: d_out) -- project back up
                        result   = base_out + lora_out

    The low-rank bottleneck (r=2) means the adaptation can only modify the output
    in a 2-dimensional subspace. This is not a limitation — it's the insight:
    fine-tuning weight updates are empirically low-rank, so a rank-2 perturbation
    captures most of the useful adaptation signal.

    Signpost: Production LoRA also applies a scaling factor alpha/r to the adapter
    output. We omit this because at r=2 the effect is absorbed into the learning rate.
    """
    base_out = linear(x, w_frozen)
    # B projects from d_in to r (compression step)
    lora_hidden = linear(x, lora_B)
    # A projects from r back to d_out (expansion step)
    lora_out = linear(lora_hidden, lora_A)
    return [b + l for b, l in zip(base_out, lora_out)]


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


# === GPT FORWARD PASS ===

def gpt_forward(
    token_id: int,
    pos_id: int,
    keys: list[list[list[Value]]],
    values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
    lora: dict[str, list[list[Value]]] | None = None,
) -> list[Value]:
    """Single-token forward pass. When lora is provided, Q and V projections use
    LoRA-augmented linear operations; all other weights remain frozen.

    The key insight: the forward pass is structurally identical whether or not LoRA
    is active. The only difference is that Q and V computations go through lora_linear()
    instead of linear(). This composability is why LoRA is so practical — it requires
    zero changes to the model architecture, only to selected weight applications.
    """
    # Embedding: token identity + positional encoding
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)

        # Q and V use LoRA adapters when available; K and O are always base-only.
        if lora is not None:
            q = lora_linear(
                x,
                params[f'layer{layer_idx}.attn_wq'],
                lora[f'layer{layer_idx}.lora_q_A'],
                lora[f'layer{layer_idx}.lora_q_B'],
            )
            v_proj = lora_linear(
                x,
                params[f'layer{layer_idx}.attn_wv'],
                lora[f'layer{layer_idx}.lora_v_A'],
                lora[f'layer{layer_idx}.lora_v_B'],
            )
        else:
            q = linear(x, params[f'layer{layer_idx}.attn_wq'])
            v_proj = linear(x, params[f'layer{layer_idx}.attn_wv'])

        k = linear(x, params[f'layer{layer_idx}.attn_wk'])

        keys[layer_idx].append(k)
        values[layer_idx].append(v_proj)

        # Multi-head attention: each head operates on a HEAD_DIM slice
        x_attn: list[Value] = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM

            q_head = q[hs : hs + HEAD_DIM]
            k_head = [k_t[hs : hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_head = [v_t[hs : hs + HEAD_DIM] for v_t in values[layer_idx]]

            # Scaled dot-product attention: score = (q . k) / sqrt(d_head)
            attn_logits = [
                sum(q_head[j] * k_head[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_head))
            ]
            attn_weights = softmax(attn_logits)

            head_output = [
                sum(attn_weights[t] * v_head[t][j] for t in range(len(v_head)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_output)

        x = linear(x_attn, params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x

        # MLP block: expand 4x, ReLU, contract
        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, params['lm_head'])


# === OPTIMIZER ===

def adam_step(
    param_list: list[Value],
    m_state: list[float],
    v_state: list[float],
    step: int,
    lr: float,
) -> None:
    """One Adam update step with bias correction and linear LR decay.

    Adam maintains per-parameter momentum (m) and variance (v) estimates.
    Bias correction compensates for the zero initialization of m and v,
    which would otherwise make early updates too small.
    """
    lr_t = lr * (1 - step / max(step + 1, 1))
    for i, param in enumerate(param_list):
        m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * param.grad
        v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * param.grad ** 2
        m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
        v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
        param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
        param.grad = 0.0


def flatten_params(params: dict[str, list[list[Value]]]) -> list[Value]:
    """Collect all Value objects from a parameter dict into a flat list."""
    return [p for matrix in params.values() for row in matrix for p in row]


# === EVALUATION ===

def evaluate_loss(
    docs: list[str],
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    params: dict[str, list[list[Value]]],
    lora: dict[str, list[list[Value]]] | None = None,
    num_samples: int = 50,
) -> float:
    """Compute average cross-entropy loss over a sample of documents.
    Uses .data only (no gradient tracking) for efficiency."""
    total_loss = 0.0
    total_tokens = 0
    for idx in range(min(num_samples, len(docs))):
        doc = docs[idx]
        tokens = [bos] + [unique_chars.index(ch) for ch in doc] + [bos]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]
        for pos in range(seq_len):
            logits = gpt_forward(tokens[pos], pos, keys, vals, params, lora)
            probs = softmax(logits)
            prob_target = max(probs[tokens[pos + 1]].data, 1e-10)
            total_loss += -math.log(prob_target)
            total_tokens += 1
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def generate_names(
    params: dict[str, list[list[Value]]],
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    lora: dict[str, list[list[Value]]] | None = None,
    num_samples: int = 5,
    temperature: float = 0.5,
) -> list[str]:
    """Generate names by autoregressively sampling from the model."""
    results: list[str] = []
    for _ in range(num_samples):
        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]
        token_id = bos
        generated: list[str] = []
        for pos in range(BLOCK_SIZE):
            logits = gpt_forward(token_id, pos, keys, vals, params, lora)
            scaled = [logit / temperature for logit in logits]
            probs = softmax(scaled)
            token_id = random.choices(
                range(vocab_size), weights=[p.data for p in probs]
            )[0]
            if token_id == bos:
                break
            generated.append(unique_chars[token_id])
        results.append(''.join(generated))
    return results


# === TRAINING ===

if __name__ == "__main__":
    # -- Load and split data --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    print(f"Loaded {len(docs)} documents")

    # Split by first letter: A-M for base training, N-Z for LoRA adaptation.
    # This creates a clean distribution shift — the two halves have different character
    # frequency distributions (e.g., N-Z names are heavier on letters n, s, t, r).
    # LoRA must adapt the model's learned character statistics without retraining from scratch.
    base_docs = [d for d in docs if d[0].upper() <= 'M']
    lora_docs = [d for d in docs if d[0].upper() > 'M']
    random.shuffle(base_docs)
    random.shuffle(lora_docs)

    print(f"Base training set: {len(base_docs)} names (A-M)")
    print(f"LoRA adaptation set: {len(lora_docs)} names (N-Z)")

    # Build vocabulary from the full corpus (both splits share the same character set)
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    # === Phase A: Base Model Training ===
    print("\n=== Phase A: Base Model Training ===")
    params = init_parameters(VOCAB_SIZE)
    base_param_list = flatten_params(params)
    print(f"Parameters: {len(base_param_list):,}")

    m_base = [0.0] * len(base_param_list)
    v_base = [0.0] * len(base_param_list)

    for step in range(BASE_STEPS):
        doc = base_docs[step % len(base_docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        losses: list[Value] = []
        for pos in range(seq_len):
            logits = gpt_forward(tokens[pos], pos, keys, vals, params)
            probs = softmax(logits)
            losses.append(-safe_log(probs[tokens[pos + 1]]))

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        # Linear LR decay prevents overshooting as the loss landscape sharpens near the optimum
        lr_t = BASE_LR * (1 - step / BASE_STEPS)
        for i, p in enumerate(base_param_list):
            m_base[i] = BETA1 * m_base[i] + (1 - BETA1) * p.grad
            v_base[i] = BETA2 * v_base[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_base[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_base[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{BASE_STEPS} | loss: {loss.data:.4f}")

    print(f"Base training complete. Final loss: {loss.data:.4f}")

    # === Phase B: LoRA Adaptation ===
    print("\n=== Phase B: LoRA Adaptation ===")

    lora_adapters = init_lora_adapters()
    lora_param_list = flatten_params(lora_adapters)

    print(f"Base parameters (frozen): {len(base_param_list):,}")
    print(f"LoRA parameters (trainable): {len(lora_param_list):,}")
    print(f"LoRA rank: {LORA_RANK}")
    print(f"Adapted matrices: Q, V projections")

    m_lora = [0.0] * len(lora_param_list)
    v_lora = [0.0] * len(lora_param_list)

    for step in range(LORA_STEPS):
        doc = lora_docs[step % len(lora_docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        keys = [[] for _ in range(N_LAYER)]
        vals = [[] for _ in range(N_LAYER)]

        losses = []
        for pos in range(seq_len):
            # Forward pass uses LoRA-augmented Q and V projections
            logits = gpt_forward(tokens[pos], pos, keys, vals, params, lora_adapters)
            probs = softmax(logits)
            losses.append(-safe_log(probs[tokens[pos + 1]]))

        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        # Freeze base model: zero all base parameter gradients after backward.
        # backward() propagates gradients through the entire graph, including frozen
        # weights. We discard those gradients here, ensuring only LoRA parameters update.
        # This is the core LoRA mechanism: the pretrained knowledge is preserved in W_frozen
        # while the adaptation signal flows exclusively through A and B.
        for p in base_param_list:
            p.grad = 0.0

        # Update only LoRA parameters
        lr_t = LORA_LR * (1 - step / LORA_STEPS)
        for i, p in enumerate(lora_param_list):
            m_lora[i] = BETA1 * m_lora[i] + (1 - BETA1) * p.grad
            v_lora[i] = BETA2 * v_lora[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_lora[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_lora[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{LORA_STEPS} | loss: {loss.data:.4f}")

    print(f"LoRA training complete. Final loss: {loss.data:.4f}")

    # === Results ===
    print("\n=== Results ===")

    pct = 100 * len(lora_param_list) / len(base_param_list)
    print(f"Trainable params \u2014 Full fine-tune: {len(base_param_list):,} | "
          f"LoRA: {len(lora_param_list):,} ({pct:.1f}%)")

    # Generate from the base model (no LoRA adapters)
    print("\nGenerating from BASE model (trained on A-M names):")
    base_names = generate_names(params, unique_chars, BOS, VOCAB_SIZE, num_samples=5)
    for i, name in enumerate(base_names):
        print(f"  {i + 1}. {name}")

    # Generate from the LoRA-adapted model
    print("\nGenerating from LoRA-ADAPTED model (adapted to N-Z names):")
    lora_names = generate_names(
        params, unique_chars, BOS, VOCAB_SIZE, lora=lora_adapters, num_samples=5
    )
    for i, name in enumerate(lora_names):
        print(f"  {i + 1}. {name}")

    # Cross-evaluate: measure loss on both splits with both models.
    # If LoRA works correctly:
    #   - Base model should do well on A-M (its training data), poorly on N-Z
    #   - LoRA-adapted model should improve on N-Z while not degrading much on A-M
    #     (because W_frozen preserves A-M knowledge and A@B only adds a small perturbation)
    loss_base_am = evaluate_loss(base_docs, unique_chars, BOS, VOCAB_SIZE, params)
    loss_base_nz = evaluate_loss(lora_docs, unique_chars, BOS, VOCAB_SIZE, params)
    loss_lora_am = evaluate_loss(
        base_docs, unique_chars, BOS, VOCAB_SIZE, params, lora_adapters
    )
    loss_lora_nz = evaluate_loss(
        lora_docs, unique_chars, BOS, VOCAB_SIZE, params, lora_adapters
    )

    print(f"\nLoss on A-M split \u2014 Base: {loss_base_am:.2f} | LoRA-adapted: {loss_lora_am:.2f}")
    print(f"Loss on N-Z split \u2014 Base: {loss_base_nz:.2f} | LoRA-adapted: {loss_lora_nz:.2f}")
