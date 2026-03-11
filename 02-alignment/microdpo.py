"""
Direct Preference Optimization (DPO): aligning a language model with human preferences using
a single contrastive loss — no reward model, no reinforcement learning, just supervised learning
on preference pairs.
"""
# Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model is
# Secretly a Reward Model" (2023). https://arxiv.org/abs/2305.18290
# Architecture reuses the microgpt pattern (Radford et al., 2019) with pedagogical
# simplifications: RMSNorm, ReLU, no biases.

# === TRADEOFFS ===
# + No reward model needed: directly optimizes policy from preference pairs
# + Stable training: standard cross-entropy loss, no RL instability
# + Simpler pipeline than PPO (one model instead of three)
# - Requires high-quality preference data (garbage preferences = garbage alignment)
# - Less flexible than RL: cannot optimize arbitrary reward functions
# - Beta hyperparameter sensitivity: too low = no effect, too high = mode collapse
# WHEN TO USE: Aligning language models when you have preference pairs and want
#   a simpler, more stable alternative to full RLHF.
# WHEN NOT TO: When you need to optimize a complex reward function, when preference
#   data is noisy or contradictory, or when online exploration is required.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture — identical to microgpt for direct comparison
N_EMBD = 16         # embedding dimension (d_model)
N_HEAD = 4          # number of attention heads
N_LAYER = 1         # transformer blocks
BLOCK_SIZE = 16     # context window length
HEAD_DIM = N_EMBD // N_HEAD  # 4 dimensions per head

# Training — base model pretraining
BASE_LR = 0.01
BASE_STEPS = 700

# Training — DPO alignment
DPO_LR = 0.003
DPO_STEPS = 60
DPO_BETA = 0.1
# Beta controls alignment strength. Low beta (0.01) barely moves the policy from the
# reference; high beta (1.0) aggressively reshapes the distribution toward preferred
# completions but risks mode collapse. 0.1 is the standard starting point in the paper.
# Intuitively: beta is the inverse temperature of the implicit reward model — higher beta
# means sharper preferences.

# Shared optimizer constants
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8

# Data
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Inference
TEMPERATURE = 0.5
NUM_SAMPLES = 10

# Signpost: ~4,200 parameters total. Production DPO (Llama-2-Chat, Zephyr) aligns models
# with billions of parameters on thousands of human-labeled preference pairs. The algorithm
# is identical; only the scale differs.


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
# with no additions beyond the base set. safe_log() is used for numerical stability
# in log-probability computation.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize weight matrix ~ N(0, std). Standard deviation 0.08 is empirically tuned
    for this tiny model; larger models use Xavier/Glorot scaling (std = 1/sqrt(d_in))."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_parameters(vocab_size: int) -> dict[str, list[list[Value]]]:
    """Initialize all model parameters: embeddings, attention, MLP, and LM head."""
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


def flatten_params(params: dict[str, list[list[Value]]]) -> list[Value]:
    """Collect all Value objects from a parameter dict into a flat list."""
    return [p for matrix in params.values() for row in matrix for p in row]


def snapshot_weights(params: dict[str, list[list[Value]]]) -> dict[str, list[list[float]]]:
    """Deep-copy all parameter values as plain floats for the reference model.

    The reference model is a frozen snapshot of the pretrained policy. DPO needs it to
    compute log-probability ratios: how much has the policy diverged from its starting point?
    Storing plain floats (not Value objects) is critical for two reasons:
    1. No autograd overhead — reference forward passes run ~10x faster without graph construction
    2. No gradient contamination — the reference model must never receive gradient updates
    """
    return {
        key: [[v.data for v in row] for row in matrix]
        for key, matrix in params.items()
    }


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiply: y = W @ x. For W of shape [n_out, n_in] and x of shape
    [n_in], output y has shape [n_out] where y[i] = sum_j W[i,j] * x[j]."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def linear_float(x: list[float], w: list[list[float]]) -> list[float]:
    """Matrix-vector multiply using plain floats. Identical to linear() but operates
    on raw floats for the reference model forward pass — no autograd graph construction."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """Stable softmax: subtract max before exp to prevent overflow.
    softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))"""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def softmax_float(logits: list[float]) -> list[float]:
    """Stable softmax on plain floats for the reference model."""
    max_val = max(logits)
    exp_vals = [math.exp(v - max_val) for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """RMS normalization: x / sqrt(mean(x^2) + eps).
    Simpler than LayerNorm (no mean centering, no learned affine). Used in LLaMA, Gemma."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def rmsnorm_float(x: list[float]) -> list[float]:
    """RMS normalization on plain floats for the reference model."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """Clipped log for numerical stability. Prevents log(0) = -inf which would break
    gradient propagation. The node is built manually with prob as its child so
    gradients flow back through the computation graph (not severed by clamping)."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === GPT FORWARD PASS (POLICY MODEL — AUTOGRAD) ===

def gpt_forward(
    token_id: int,
    pos_id: int,
    keys: list[list[list[Value]]],
    values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
) -> list[Value]:
    """Single-token forward pass through the policy model (with autograd).

    Returns logits over the vocabulary. Keys/values accumulate the KV cache for
    causal attention without an explicit mask.
    """
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, params[f'layer{layer_idx}.attn_wq'])
        k = linear(x, params[f'layer{layer_idx}.attn_wk'])
        v_proj = linear(x, params[f'layer{layer_idx}.attn_wv'])

        keys[layer_idx].append(k)
        values[layer_idx].append(v_proj)

        x_attn: list[Value] = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM
            q_head = q[hs : hs + HEAD_DIM]
            k_head = [k_t[hs : hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_head = [v_t[hs : hs + HEAD_DIM] for v_t in values[layer_idx]]

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

        x = rmsnorm(x)
        x = linear(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, params['lm_head'])


# === GPT FORWARD PASS (REFERENCE MODEL — PLAIN FLOATS) ===

def gpt_forward_float(
    token_id: int,
    pos_id: int,
    keys: list[list[list[float]]],
    values: list[list[list[float]]],
    params: dict[str, list[list[float]]],
) -> list[float]:
    """Single-token forward pass using plain floats for the frozen reference model.

    Structurally identical to gpt_forward() but operates entirely on floats.
    No computation graph is built, making this ~10x faster. The reference model
    is never updated, so gradients are unnecessary.
    """
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm_float(x)

    for layer_idx in range(N_LAYER):
        x_residual = x
        x = rmsnorm_float(x)

        q = linear_float(x, params[f'layer{layer_idx}.attn_wq'])
        k = linear_float(x, params[f'layer{layer_idx}.attn_wk'])
        v_proj = linear_float(x, params[f'layer{layer_idx}.attn_wv'])

        keys[layer_idx].append(k)
        values[layer_idx].append(v_proj)

        x_attn: list[float] = []
        for head in range(N_HEAD):
            hs = head * HEAD_DIM
            q_head = q[hs : hs + HEAD_DIM]
            k_head = [k_t[hs : hs + HEAD_DIM] for k_t in keys[layer_idx]]
            v_head = [v_t[hs : hs + HEAD_DIM] for v_t in values[layer_idx]]

            attn_logits = [
                sum(q_head[j] * k_head[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                for t in range(len(k_head))
            ]
            attn_weights = softmax_float(attn_logits)

            head_output = [
                sum(attn_weights[t] * v_head[t][j] for t in range(len(v_head)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_output)

        x = linear_float(x_attn, params[f'layer{layer_idx}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x

        x = rmsnorm_float(x)
        x = linear_float(x, params[f'layer{layer_idx}.mlp_fc1'])
        x = [max(0.0, xi) for xi in x]  # ReLU on plain floats
        x = linear_float(x, params[f'layer{layer_idx}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear_float(x, params['lm_head'])


# === SEQUENCE LOG-PROBABILITY ===

def sequence_log_prob_policy(
    tokens: list[int],
    params: dict[str, list[list[Value]]],
) -> Value:
    """Compute log P(sequence) under the policy model using autograd.

    Math: log P(x_0, x_1, ..., x_T) = sum_{t=0}^{T-1} log P(x_{t+1} | x_0..x_t)
    Each term is the log-probability of the next token given all preceding tokens.
    The sum flows through the autograd graph so DPO gradients reach every parameter.
    """
    keys = [[] for _ in range(N_LAYER)]
    vals = [[] for _ in range(N_LAYER)]
    total_log_prob = Value(0.0)

    for pos in range(len(tokens) - 1):
        logits = gpt_forward(tokens[pos], pos, keys, vals, params)
        probs = softmax(logits)
        total_log_prob = total_log_prob + safe_log(probs[tokens[pos + 1]])

    return total_log_prob


def sequence_log_prob_reference(
    tokens: list[int],
    ref_params: dict[str, list[list[float]]],
) -> float:
    """Compute log P(sequence) under the frozen reference model using plain floats.

    Same math as sequence_log_prob_policy, but no autograd overhead. Returns a plain
    float because the reference model is never updated — its log-probs are constants
    in the DPO loss.
    """
    keys: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
    vals: list[list[list[float]]] = [[] for _ in range(N_LAYER)]
    total_log_prob = 0.0

    for pos in range(len(tokens) - 1):
        logits = gpt_forward_float(tokens[pos], pos, keys, vals, ref_params)
        probs = softmax_float(logits)
        prob_next = max(probs[tokens[pos + 1]], 1e-10)
        total_log_prob += math.log(prob_next)

    return total_log_prob


# === DPO LOSS ===

def dpo_loss(
    chosen_tokens: list[int],
    rejected_tokens: list[int],
    params: dict[str, list[list[Value]]],
    ref_params: dict[str, list[list[float]]],
    beta: float,
) -> tuple[Value, float, float]:
    """Compute the DPO loss for a single preference pair.

    Math: L_DPO = -log(sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x))))

    Where:
        y_w = chosen (preferred) completion
        y_l = rejected completion
        pi = policy model (trainable)
        pi_ref = reference model (frozen pretrained snapshot)
        sigma = sigmoid function
        beta = alignment strength (inverse temperature of implicit reward)

    The log-ratio log(pi/pi_ref) measures how much the policy has diverged from the
    reference for a given sequence. DPO pushes this ratio UP for chosen sequences
    and DOWN for rejected sequences. The key insight from Rafailov et al.: this
    contrastive objective is equivalent to RL with an implicit reward model
    r(x, y) = beta * log(pi(y|x) / pi_ref(y|x)), but without ever training that
    reward model explicitly.

    Returns: (loss, chosen_reward, rejected_reward) where rewards are the implicit
    reward values used for monitoring alignment progress.
    """
    # Policy log-probs (through autograd graph)
    log_pi_chosen = sequence_log_prob_policy(chosen_tokens, params)
    log_pi_rejected = sequence_log_prob_policy(rejected_tokens, params)

    # Reference log-probs (plain floats, no gradients needed)
    log_ref_chosen = sequence_log_prob_reference(chosen_tokens, ref_params)
    log_ref_rejected = sequence_log_prob_reference(rejected_tokens, ref_params)

    # Log-ratios: how much has the policy diverged from the reference?
    # Math: log(pi(y|x) / pi_ref(y|x)) = log pi(y|x) - log pi_ref(y|x)
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected

    # Implicit reward difference (the argument to sigmoid)
    # Math: delta = beta * (log_ratio_chosen - log_ratio_rejected)
    # Positive delta means the policy prefers chosen over rejected MORE than the reference does.
    delta = beta * (log_ratio_chosen - log_ratio_rejected)

    # Numerically stable sigmoid: -log(sigma(x)) = log(1 + exp(-x))
    # For large positive x: log(1 + exp(-x)) ~ exp(-x) ~ 0 (loss near 0, correct preference)
    # For large negative x: log(1 + exp(-x)) ~ -x (loss grows, wrong preference)
    # We implement this via: loss = -log(sigma(delta)) = log(1 + exp(-delta))
    # Using the logsigmoid identity avoids computing sigma directly, which can overflow.
    neg_delta = -delta
    # Stable log(1 + exp(z)): use z itself when z >> 0, exp(z) when z << 0
    if neg_delta.data > 20.0:
        # For very large neg_delta, exp(-delta) dominates: log(1+exp(z)) ~ z
        loss = neg_delta
    else:
        loss = (Value(1.0) + neg_delta.exp()).log()

    # Implicit rewards for monitoring (plain float, not part of loss graph)
    # Math: r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))
    # The reward the DPO policy implicitly assigns to each completion.
    chosen_reward = beta * log_ratio_chosen.data
    rejected_reward = beta * log_ratio_rejected.data

    return loss, chosen_reward, rejected_reward


# === PREFERENCE PAIR CONSTRUCTION ===

def create_preference_pairs(
    docs: list[str],
    unique_chars: list[str],
    bos: int,
    min_prompt_len: int = 2,
    max_prompt_len: int = 3,
) -> list[tuple[list[int], list[int]]]:
    """Create synthetic preference pairs from the training data.

    Strategy: prefer longer names over shorter ones. For each name, take the first
    2-3 characters as a shared prompt prefix, then pair a long completion (chosen)
    with a short completion (rejected) that shares the same prefix.

    Why length as a preference signal? It is simple, verifiable, and doesn't require
    human annotation. After DPO training, the policy should generate longer names than
    the reference model — a measurable behavioral shift that proves the algorithm works.

    Production DPO uses human-labeled preference pairs (e.g., "response A is more helpful
    than response B"). The loss function is identical; only the preference signal differs.
    """
    # Group names by their prefix (first 2-3 characters)
    from collections import defaultdict
    prefix_groups: dict[str, list[str]] = defaultdict(list)

    for doc in docs:
        if len(doc) < 2:
            continue
        for plen in range(min_prompt_len, min(max_prompt_len + 1, len(doc))):
            prefix = doc[:plen]
            prefix_groups[prefix].append(doc)

    pairs: list[tuple[list[int], list[int]]] = []

    for prefix, names in prefix_groups.items():
        # Separate into long (chosen) and short (rejected) completions
        long_names = [n for n in names if len(n) >= 5]
        short_names = [n for n in names if len(n) <= 3]

        if not long_names or not short_names:
            continue

        # Create pairs: each long name paired with a short name sharing the prefix
        for long_name in long_names[:2]:  # cap to avoid excessive pairs per prefix
            short_name = random.choice(short_names)

            # Tokenize: full sequence including BOS markers
            chosen_tokens = [bos] + [unique_chars.index(ch) for ch in long_name] + [bos]
            rejected_tokens = [bos] + [unique_chars.index(ch) for ch in short_name] + [bos]

            # Truncate to block size
            chosen_tokens = chosen_tokens[:BLOCK_SIZE + 1]
            rejected_tokens = rejected_tokens[:BLOCK_SIZE + 1]

            pairs.append((chosen_tokens, rejected_tokens))

    random.shuffle(pairs)
    return pairs


# === GENERATION ===

def generate_names(
    params: dict[str, list[list[Value]]],
    unique_chars: list[str],
    bos: int,
    vocab_size: int,
    num_samples: int = 10,
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
            logits = gpt_forward(token_id, pos, keys, vals, params)
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
    start_time = time.time()

    # -- Load and prepare data --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(docs)} documents")

    # =========================================================================
    # === Phase 1: Pretraining Base Model ===
    # =========================================================================
    # Standard language model training: predict the next character given all preceding
    # characters. This produces the base policy that DPO will later align.
    print("\n=== Phase 1: Pretraining Base Model ===")
    params = init_parameters(VOCAB_SIZE)
    param_list = flatten_params(params)
    print(f"Parameters: {len(param_list):,}")

    m_state = [0.0] * len(param_list)
    v_state = [0.0] * len(param_list)

    for step in range(BASE_STEPS):
        doc = docs[step % len(docs)]
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

        lr_t = BASE_LR * (1 - step / BASE_STEPS)
        for i, p in enumerate(param_list):
            m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * p.grad
            v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{BASE_STEPS} | loss: {loss.data:.4f}")

    print(f"Pretraining complete. Final loss: {loss.data:.4f}")

    # =========================================================================
    # === Phase 2: Creating Preference Pairs ===
    # =========================================================================
    # Snapshot the pretrained weights as the frozen reference model BEFORE creating
    # preference pairs. The reference model anchors the DPO loss — it prevents the
    # policy from straying too far from the pretrained distribution (KL regularization).
    print("\n=== Phase 2: Creating Preference Pairs ===")

    ref_params = snapshot_weights(params)

    preference_pairs = create_preference_pairs(docs, unique_chars, BOS)

    # Cap pairs to keep runtime bounded. Production DPO uses thousands to millions of pairs;
    # we use 150 because each pair requires two full sequence forward passes (policy + reference).
    max_pairs = 150
    if len(preference_pairs) > max_pairs:
        preference_pairs = preference_pairs[:max_pairs]

    print(f"Created {len(preference_pairs)} preference pairs (prefer longer completions)")

    # Show an example pair for interpretability
    if preference_pairs:
        chosen_ex, rejected_ex = preference_pairs[0]
        chosen_str = ''.join(unique_chars[t] for t in chosen_ex[1:-1])  # strip BOS
        rejected_str = ''.join(unique_chars[t] for t in rejected_ex[1:-1])
        print(f'Example: chosen="{chosen_str}" | rejected="{rejected_str}"')

    # =========================================================================
    # === Phase 3: DPO Training ===
    # =========================================================================
    # The core DPO loop. For each preference pair (chosen, rejected):
    # 1. Compute log P(chosen) and log P(rejected) under the policy (with autograd)
    # 2. Compute log P(chosen) and log P(rejected) under the reference (plain floats)
    # 3. DPO loss pushes the policy to increase the log-ratio gap: prefer chosen more,
    #    reject rejected more, relative to what the reference model does.
    #
    # Why DPO instead of RLHF (PPO)?
    # Standard RLHF requires: (1) train a reward model, (2) run PPO with the reward model
    # as the signal. DPO collapses both steps into a single supervised loss by proving that
    # the optimal policy under the reward model has a closed-form relationship to the
    # preference data. The result is simpler code, fewer hyperparameters, and no reward model.
    print("\n=== Phase 3: DPO Training ===")
    print(f"Beta: {DPO_BETA}")

    # Reset optimizer state for DPO phase (different learning dynamics than pretraining)
    m_state = [0.0] * len(param_list)
    v_state = [0.0] * len(param_list)

    for step in range(DPO_STEPS):
        # Sample a mini-batch of preference pairs
        # Using a batch reduces variance in the gradient estimate
        batch_size = 4
        batch_indices = [random.randint(0, len(preference_pairs) - 1) for _ in range(batch_size)]

        total_loss = Value(0.0)
        total_chosen_reward = 0.0
        total_rejected_reward = 0.0

        for idx in batch_indices:
            chosen_tokens, rejected_tokens = preference_pairs[idx]
            loss_i, cr, rr = dpo_loss(
                chosen_tokens, rejected_tokens, params, ref_params, DPO_BETA
            )
            total_loss = total_loss + loss_i
            total_chosen_reward += cr
            total_rejected_reward += rr

        # Average over the batch
        avg_loss = total_loss * (1.0 / batch_size)
        avg_loss.backward()

        lr_t = DPO_LR * (1 - step / DPO_STEPS)
        for i, p in enumerate(param_list):
            m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * p.grad
            v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 10 == 0 or step == 0:
            mean_cr = total_chosen_reward / batch_size
            mean_rr = total_rejected_reward / batch_size
            print(
                f"  step {step + 1:>3}/{DPO_STEPS} | "
                f"dpo_loss: {avg_loss.data:.4f} | "
                f"mean_chosen_reward: {mean_cr:.2f} | "
                f"mean_rejected_reward: {mean_rr:.2f}"
            )

    print("DPO training complete.")

    # =========================================================================
    # === Results ===
    # =========================================================================
    # Compare generation from the reference model (pretrained) and the DPO-aligned policy.
    # The aligned model should produce longer names on average — that is the preference signal.
    print("\n=== Results ===")

    # Generate from the reference model by temporarily loading reference weights
    # into fresh Value objects (we need Value objects for the generation function).
    ref_value_params: dict[str, list[list[Value]]] = {
        key: [[Value(v) for v in row] for row in matrix]
        for key, matrix in ref_params.items()
    }

    print("Generating from REFERENCE model:")
    ref_names = generate_names(ref_value_params, unique_chars, BOS, VOCAB_SIZE, NUM_SAMPLES, TEMPERATURE)
    for i, name in enumerate(ref_names):
        print(f"  {i + 1:>2}. {name} (length {len(name)})")

    print("\nGenerating from DPO-ALIGNED model:")
    dpo_names = generate_names(params, unique_chars, BOS, VOCAB_SIZE, NUM_SAMPLES, TEMPERATURE)
    for i, name in enumerate(dpo_names):
        print(f"  {i + 1:>2}. {name} (length {len(name)})")

    ref_avg = sum(len(n) for n in ref_names) / len(ref_names) if ref_names else 0
    dpo_avg = sum(len(n) for n in dpo_names) / len(dpo_names) if dpo_names else 0
    print(f"\nAverage generated length -- Reference: {ref_avg:.1f} | DPO-aligned: {dpo_avg:.1f}")

    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.1f}s")
