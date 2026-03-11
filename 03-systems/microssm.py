"""
The alternative to attention -- how state space models process sequences in linear time
through selective state transitions.
"""
# Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
# (2023). This implementation builds a simplified Mamba-style SSM from scratch: continuous
# state space formulation, Euler discretization, and the selective mechanism that makes
# SSMs content-aware. Trained on character-level name generation for direct comparison
# with microgpt.py (attention-based) and micrornn.py (recurrence-based).

# === TRADEOFFS ===
# + Linear-time sequence processing: O(L) vs. attention's O(L^2)
# + Constant memory per step during inference (like RNNs, unlike KV-cache growth)
# + Selective mechanism makes state transitions input-dependent (content-aware)
# - Less expressive than full attention for tasks requiring arbitrary token-pair interaction
# - Parallel training requires careful scan implementations (hardware-specific optimization)
# - Smaller ecosystem: fewer pretrained models and less tooling than transformers
# WHEN TO USE: Very long sequences (>8K tokens) where attention's quadratic cost is
#   prohibitive, or streaming applications with strict memory budgets.
# WHEN NOT TO: Tasks requiring dense token-pair interactions (retrieval, copying), or
#   when leveraging the transformer pretrained model ecosystem is more practical.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture -- dimensions chosen to match microgpt for direct comparison
N_EMBD = 16          # embedding/model dimension (D in the Mamba paper)
N_STATE = 8          # SSM hidden state dimension per channel (N in the paper)
BLOCK_SIZE = 16      # maximum sequence length (context window)
N_LAYER = 1          # number of stacked SSM layers

# Training parameters
LEARNING_RATE = 0.01  # Adam base learning rate
BETA1 = 0.85          # Adam first moment decay
BETA2 = 0.99          # Adam second moment decay
EPS_ADAM = 1e-8       # Adam epsilon
NUM_STEPS = 800       # total training steps
TRAIN_SIZE = 250      # small training subset for tractability with scalar autograd

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: ~2,800 parameters total. Real Mamba models have hundreds of millions.
# The architecture implements the same selective scan mechanism; this toy scale
# lets us train on CPU in minutes. N_STATE=8 is tiny -- production SSMs use
# N=16 with D_MODEL=768+, giving them far more capacity per layer.


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

    Tracks computational history via ._children and ._local_grads, enabling
    gradient computation through the chain rule. Every forward operation stores
    its local derivative (dout/dinput) as a closure, then backward() replays
    the computation graph in reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data          # scalar float value
        self.grad = 0.0           # accumulated gradient (dLoss/dself)
        self._children = children # parent Values in the computation graph
        self._local_grads = local_grads  # dself/dchild for each child

    # Arithmetic operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent):
        # d(x^n)/dx = n * x^(n-1)
        return Value(self.data ** exponent, (self,), (exponent * self.data ** (exponent - 1),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    # Activation functions
    def tanh(self):
        # d(tanh(x))/dx = 1 - tanh(x)^2
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self):
        # d(e^x)/dx = e^x
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        # d(log(x))/dx = 1/x
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        # d(relu(x))/dx = 1 if x > 0 else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """Compute gradients via reverse-mode automatic differentiation.

        Builds a topological ordering of the computation graph, then propagates
        gradients backward using the chain rule.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed: gradient of loss with respect to itself is 1
        self.grad = 1.0

        # Reverse topological order: gradients flow backward from output to inputs
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # Chain rule: dLoss/dchild += dLoss/dv * dv/dchild
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# See docs/autograd-interface.md for the full specification.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise.

    Standard deviation of 0.08 is chosen empirically for this tiny model.
    Larger models typically use std = 1/sqrt(d_in) (Xavier/Glorot initialization).
    """
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def make_vector(size: int, val: float = 0.0) -> list[Value]:
    """Initialize a bias vector to a constant value."""
    return [Value(val) for _ in range(size)]


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]], b: list[Value] | None = None) -> list[Value]:
    """Matrix-vector multiplication: y = W @ x + b (bias optional).

    For weight matrix W with shape [n_out, n_in] and input vector x with
    shape [n_in], computes output y with shape [n_out].
    """
    y = [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]
    if b is not None:
        y = [y_i + b_i for y_i, b_i in zip(y, b)]
    return y


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: converts logits to probabilities.

    Subtracts max before exp() to prevent overflow. Softmax is translation-invariant
    so this doesn't change the result.
    Math: softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def safe_log(prob: Value) -> Value:
    """Clipped logarithm for numerical stability in loss computation.

    Prevents log(0) which returns -inf and breaks gradient backpropagation.
    Critical: we keep `prob` as a child node so gradients flow through the graph.
    """
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


def rmsnorm(x: list[Value]) -> list[Value]:
    """Root Mean Square normalization: scale vector to unit RMS magnitude.

    RMSNorm is LayerNorm without mean centering. Fewer ops, empirically works
    just as well (used in LLaMA, Gemma, Mamba-2).
    Math: RMSNorm(x) = x / sqrt(mean(x^2) + eps)
    """
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# === SSM LAYER (DISCRETIZED STATE SPACE MODEL) ===

# The continuous-time state space model is defined by:
#   dx/dt = A*x + B*u        (state equation)
#   y     = C*x              (output equation)
#
# Where:
#   x(t) is the hidden state vector (dim N = N_STATE)
#   u(t) is the scalar input signal
#   y(t) is the scalar output
#   A is the state transition matrix (N x N, but we use diagonal for efficiency)
#   B is the input projection vector (N x 1)
#   C is the output projection vector (1 x N)
#
# To process discrete sequences (tokens), we discretize using Euler's method:
#   A_bar = I + delta * A    (Euler approximation of matrix exponential exp(delta*A))
#   B_bar = delta * B
#   x_k   = A_bar * x_{k-1} + B_bar * u_k
#   y_k   = C * x_k
#
# Signpost: The exact discretization uses matrix exponential A_bar = exp(delta*A)
# and B_bar = A^{-1}(exp(delta*A) - I)*B. Euler is a first-order approximation
# that works well for small delta. S4 uses the bilinear (Tustin) method instead.
# We use Euler because it's simplest and this is a pedagogical implementation.

def init_ssm_params() -> dict:
    """Initialize parameters for a selective SSM model.

    Architecture per SSM layer:
    - Input projection: maps N_EMBD -> N_EMBD (pre-processing)
    - A diagonal: N_STATE values (log-parameterized for stability)
    - delta projection: N_EMBD -> N_EMBD (input-dependent step size)
    - B projection: N_EMBD -> N_STATE (input-dependent, the "selective" part)
    - C projection: N_EMBD -> N_STATE (input-dependent)
    - Output projection: N_EMBD -> N_EMBD

    The selective mechanism: In classical SSMs (S4, LSSL), B and C are fixed learned
    parameters. Mamba's key innovation is making B and C functions of the input u_k.
    This means the state transition changes at every timestep based on what the model
    is reading, giving it content-aware filtering -- the ability to selectively
    remember or forget information based on the input.
    """
    params = {}

    # Token and position embeddings (same pattern as microgpt)
    params['wte'] = make_matrix(VOCAB_SIZE, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    for layer_idx in range(N_LAYER):
        prefix = f'layer{layer_idx}'

        # Input projection: processes the embedding before SSM scan
        params[f'{prefix}.in_proj'] = make_matrix(N_EMBD, N_EMBD)

        # A diagonal: initialized to negative values (log-space) so exp(A) < 1,
        # ensuring the state decays rather than explodes. This is the HiPPO
        # initialization insight from S4: the state matrix should be stable.
        # We use A_i = -exp(log_A_i) to guarantee A < 0.
        #
        # Signpost: Real SSMs use HiPPO initialization for A matrix -- structured
        # matrices that let the state approximate polynomial bases of the input
        # history, enabling long-range memory. We use simplified random init.
        params[f'{prefix}.log_A'] = [Value(random.gauss(-1.0, 0.3))
                                     for _ in range(N_STATE)]

        # Delta projection: computes the discretization step size from the input.
        # delta controls how much new information each timestep absorbs.
        # Large delta = absorb more input, small delta = retain more state.
        # Parameterized as softplus(linear(x)) to ensure delta > 0.
        params[f'{prefix}.delta_proj'] = make_matrix(N_EMBD, N_EMBD)
        params[f'{prefix}.delta_bias'] = make_vector(N_EMBD, -2.0)
        # Bias initialized to -2.0 so softplus(-2) ~ 0.13, giving moderate step sizes

        # Selective B and C projections: these are the key Mamba innovation.
        # Instead of fixed B and C matrices, we compute them from the input:
        #   B_k = W_B @ u_k    (input-dependent input projection)
        #   C_k = W_C @ u_k    (input-dependent output projection)
        # This makes the SSM content-aware: it can choose what to store in state
        # based on what it's currently reading.
        params[f'{prefix}.W_B'] = make_matrix(N_STATE, N_EMBD)
        params[f'{prefix}.W_C'] = make_matrix(N_STATE, N_EMBD)

        # Output projection: maps SSM output back to model dimension
        params[f'{prefix}.out_proj'] = make_matrix(N_EMBD, N_EMBD)

    # Language model head: projects final hidden states to vocabulary logits
    params['lm_head'] = make_matrix(VOCAB_SIZE, N_EMBD)

    return params


# === SELECTIVE SCAN (MAMBA'S CORE OPERATION) ===

def selective_scan(
    u: list[Value],
    ssm_state: list[Value],
    params: dict,
    prefix: str,
) -> tuple[list[Value], list[Value]]:
    """Process one timestep through the selective SSM.

    This is the heart of the Mamba architecture. Unlike attention (which compares
    the current token to ALL previous tokens in O(n^2)), the SSM maintains a fixed-size
    state vector that evolves at each step in O(1). The sequence of operations:

    1. Compute input-dependent B_k and C_k (selective mechanism)
    2. Compute input-dependent delta_k (discretization step)
    3. Discretize: A_bar = 1 + delta*A, B_bar = delta*B (Euler method)
    4. Update state: x_k = A_bar * x_{k-1} + B_bar * u_k
    5. Compute output: y_k = C_k . x_k

    The state x carries a compressed summary of the entire history, updated at each
    step. This is fundamentally different from attention, where the "memory" is the
    full KV cache of all past tokens.

    Math-to-code mapping:
        u_k           -> u (input vector, shape [N_EMBD])
        x_{k-1}       -> ssm_state (previous hidden state, [N_STATE * N_EMBD])
        B_k = W_B@u_k -> B_k (input-dependent, [N_STATE])
        C_k = W_C@u_k -> C_k (input-dependent, [N_STATE])
        delta_k       -> delta (softplus of projected input, [N_EMBD])
        A_bar = 1+d*A -> a_bar (Euler discretization of diagonal A)
        B_bar = d*B   -> b_bar (Euler discretization)
        x_k = A_bar*x_{k-1} + B_bar*u_k -> new state per channel
        y_k = C_k . x_k -> output per channel

    Signpost: Production Mamba uses parallel scan for O(n) parallel training;
    our scalar loop processes one timestep at a time (sequential).
    """
    # Step 1: Compute input-dependent B and C (the selective mechanism)
    # In classical SSMs, B and C are fixed parameters learned during training.
    # Mamba makes them functions of the current input, so the model can decide
    # what information to store (B) and retrieve (C) based on content.
    B_k = linear(u, params[f'{prefix}.W_B'])  # [N_STATE]
    C_k = linear(u, params[f'{prefix}.W_C'])  # [N_STATE]

    # Step 2: Compute delta (discretization step size) from input.
    # delta controls the "speed" of the continuous dynamics at this timestep:
    # large delta absorbs more input and decays more state; small delta preserves
    # existing state and ignores input. This is another form of selectivity --
    # the model controls its own temporal resolution based on content.
    delta_pre = linear(u, params[f'{prefix}.delta_proj'], params[f'{prefix}.delta_bias'])

    # Softplus ensures delta > 0: softplus(x) = log(1 + exp(x))
    # Gradient: d(softplus)/dx = sigmoid(x) = exp(x) / (1 + exp(x))
    delta = []
    for d in delta_pre:
        d_clamped = min(d.data, 20.0)  # clamp to prevent overflow in exp()
        sp_val = math.log(1.0 + math.exp(d_clamped))
        sp_grad = math.exp(d_clamped) / (1.0 + math.exp(d_clamped))
        delta.append(Value(sp_val, (d,), (sp_grad,)))

    # Step 3: Retrieve A diagonal. A = -exp(log_A) guarantees A < 0, so the
    # state naturally decays. Without this, the state grows exponentially.
    log_A = params[f'{prefix}.log_A']
    A_diag = []
    for la in log_A:
        neg_exp = -math.exp(la.data)
        # d(-exp(log_A))/d(log_A) = -exp(log_A)
        A_diag.append(Value(neg_exp, (la,), (neg_exp,)))

    # Step 4: Discretize and scan -- process each of N_EMBD channels independently
    # Each channel d has its own N_STATE-dimensional state vector.
    # This is the key efficiency insight: N_EMBD independent scalar SSMs running
    # in parallel, each with its own delta but sharing B and C.
    new_state = []
    y_channels = []

    for d in range(N_EMBD):
        delta_d = delta[d]  # per-channel discretization step
        y_d = Value(0.0)    # per-channel output accumulator

        for n in range(N_STATE):
            state_idx = d * N_STATE + n

            # Euler discretization for this (d, n) element:
            # a_bar = 1 + delta_d * A_n  (scalar approximation of exp(delta*A))
            a_bar = Value(1.0) + delta_d * A_diag[n]

            # b_bar = delta_d * B_k[n]
            b_bar = delta_d * B_k[n]

            # State update: x_k[d,n] = a_bar * x_{k-1}[d,n] + b_bar * u[d]
            # This is the discrete recurrence. a_bar controls how much old state
            # is retained (decay), b_bar controls how much new input is absorbed.
            x_new = a_bar * ssm_state[state_idx] + b_bar * u[d]

            new_state.append(x_new)

            # Output accumulation: y_d += C_k[n] * x_new
            y_d = y_d + C_k[n] * x_new

        y_channels.append(y_d)

    return y_channels, new_state


# === MODEL FORWARD PASS ===

def ssm_forward(
    token_id: int,
    pos_id: int,
    ssm_states: list[list[Value]],
    params: dict,
) -> list[Value]:
    """Single-token forward pass through the SSM model.

    The ssm_states list maintains the running hidden state for each layer --
    analogous to the KV cache in transformers but fixed-size: the KV cache grows
    with sequence length O(n), while SSM state is constant regardless of tokens processed.

    Architecture: input -> rmsnorm -> in_proj -> selective_scan -> out_proj -> residual
    """
    # Embedding: token + position (same pattern as microgpt)
    tok_emb = params['wte'][token_id]
    pos_emb = params['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    for layer_idx in range(N_LAYER):
        prefix = f'layer{layer_idx}'
        x_residual = x

        # Pre-norm before SSM layer (matches transformer pre-norm convention)
        x = rmsnorm(x)

        # Input projection
        u = linear(x, params[f'{prefix}.in_proj'])

        # Selective scan: the core SSM operation -- O(1) per token, not O(n)
        y, new_state = selective_scan(u, ssm_states[layer_idx], params, prefix)
        ssm_states[layer_idx] = new_state

        # Output projection
        x = linear(y, params[f'{prefix}.out_proj'])

        # Residual connection: gradient highway, same purpose as in transformers
        x = [a + b for a, b in zip(x, x_residual)]

    # Project to vocabulary logits
    logits = linear(x, params['lm_head'])
    return logits


# === TRAINING LOOP ===

if __name__ == "__main__":
    start_time = time.time()

    # -- Prepare vocabulary and data --
    print("Loading data...")
    all_docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(all_docs)

    # Small subset: each step builds a full autograd graph through the SSM recurrence
    docs = all_docs[:TRAIN_SIZE]

    # Build vocabulary from all names (same charset as microgpt)
    unique_chars = sorted(set(''.join(all_docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1

    print(f"Loaded {len(all_docs)} documents, training on {len(docs)}")
    print(f"Vocabulary size: {VOCAB_SIZE} (characters + BOS token)")

    # Initialize parameters
    params = init_ssm_params()

    # Flatten all parameters into a single list for optimizer bookkeeping.
    # The params dict contains both list[Value] (bias vectors, log_A) and
    # list[list[Value]] (weight matrices), so we handle both cases.
    param_list = []
    for key, val in params.items():
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], Value):
                param_list.extend(val)
            elif isinstance(val[0], list):
                for row in val:
                    param_list.extend(row)

    print(f"Parameters: {len(param_list):,}")
    print(f"SSM state size per layer: {N_STATE * N_EMBD} "
          f"({N_STATE} state dims x {N_EMBD} channels)\n")

    # Initialize Adam optimizer state
    # m: first moment (momentum), v: second moment (variance)
    m = [0.0] * len(param_list)
    v = [0.0] * len(param_list)

    # -- Training --
    print("Training SSM...")
    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]

        # Tokenize: [BOS, char_0, char_1, ..., char_n, BOS]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)

        # Initialize SSM states to zero for each new sequence.
        # Shape per layer: [N_EMBD * N_STATE] (N_EMBD channels, each N_STATE wide).
        # This is the entire "memory" of the model -- contrast with transformers
        # where the KV cache grows linearly with sequence length.
        ssm_states = [[Value(0.0) for _ in range(N_STATE * N_EMBD)]
                       for _ in range(N_LAYER)]

        # Forward pass: process each token sequentially through the SSM.
        # This is O(n * d * N) total -- linear in sequence length.
        losses = []
        for pos in range(seq_len):
            input_token = tokens[pos]
            target_token = tokens[pos + 1]

            logits = ssm_forward(input_token, pos, ssm_states, params)
            probs = softmax(logits)

            # Cross-entropy loss: -log(p(target))
            loss_t = -safe_log(probs[target_token])
            losses.append(loss_t)

        # Average loss over sequence (makes loss scale-invariant to doc length)
        loss = (1.0 / seq_len) * sum(losses)

        # Backward pass
        loss.backward()

        # Adam optimizer step with linear LR decay
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)

        for i, param in enumerate(param_list):
            m[i] = BETA1 * m[i] + (1 - BETA1) * param.grad
            v[i] = BETA2 * v[i] + (1 - BETA2) * param.grad ** 2

            # Bias correction: compensates for zero initialization of m and v
            m_hat = m[i] / (1 - BETA1 ** (step + 1))
            v_hat = v[i] / (1 - BETA2 ** (step + 1))

            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            param.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - start_time
            print(f"  step {step + 1:>4}/{NUM_STEPS:>4} | loss: {loss.data:.4f} | "
                  f"time: {elapsed:.1f}s")

    training_time = time.time() - start_time
    print(f"\nTraining complete. Final loss: {loss.data:.4f}")
    print(f"Training time: {training_time:.1f}s\n")


    # === INFERENCE ===

    TEMPERATURE = 0.5
    NUM_SAMPLES = 20

    print(f"Generating {NUM_SAMPLES} samples (temperature={TEMPERATURE}):\n")

    for sample_idx in range(NUM_SAMPLES):
        # Fresh SSM state for each sample -- unlike the KV cache in transformers
        # which grows linearly with sequence length, this state is always
        # N_STATE * N_EMBD scalars regardless of how long the sequence gets.
        ssm_states = [[Value(0.0) for _ in range(N_STATE * N_EMBD)]
                       for _ in range(N_LAYER)]

        token_id = BOS
        generated = []

        for pos in range(BLOCK_SIZE):
            logits = ssm_forward(token_id, pos, ssm_states, params)

            # Temperature scaling: lower = more deterministic, higher = more random
            scaled_logits = [logit / TEMPERATURE for logit in logits]
            probs = softmax(scaled_logits)

            token_id = random.choices(
                range(VOCAB_SIZE),
                weights=[p.data for p in probs]
            )[0]

            if token_id == BOS:
                break

            generated.append(unique_chars[token_id])

        print(f"  {sample_idx + 1:>2}. {''.join(generated)}")

    # === ARCHITECTURE COMPARISON: RNN vs TRANSFORMER vs SSM ===
    #
    # This comparison is conceptual (no runtime benchmark). The key trade-offs
    # between the three dominant sequence modeling paradigms:

    print("\n" + "=" * 70)
    print("RNN vs TRANSFORMER vs SSM: ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(f"\n{'Property':<25} | {'RNN/GRU':<18} | {'Transformer':<18} | {'SSM (Mamba)':<18}")
    print("-" * 85)
    print(f"{'Per-step compute':<25} | {'O(1)':<18} | {'O(n)':<18} | {'O(1)':<18}")
    print(f"{'Full sequence':<25} | {'O(n * d^2)':<18} | {'O(n^2 * d)':<18} | {'O(n * d * N)':<18}")
    print(f"{'Memory (inference)':<25} | {'O(d)':<18} | {'O(n * d)':<18} | {'O(d * N)':<18}")
    print(f"{'Training parallel':<25} | {'No':<18} | {'Yes':<18} | {'Yes (scan)':<18}")
    print(f"{'Content-aware':<25} | {'Yes (gates)':<18} | {'Yes (QKV)':<18} | {'Yes (selective)':<18}")
    print(f"{'Long-range memory':<25} | {'Weak':<18} | {'Exact':<18} | {'Compressed':<18}")

    print(f"\nWhere n=sequence length, d=model dimension, N=state dimension")

    # Why SSMs matter: they combine the best properties of both predecessors
    print("\nKey insight: SSMs achieve the best of both worlds.")
    print("  - Like RNNs: O(1) per-step compute and fixed-size state (no KV cache)")
    print("  - Like Transformers: parallelizable training (via convolution/scan view)")
    print("  - Mamba's addition: content-aware state transitions (selective mechanism)")
    print("    makes B, C input-dependent so the model chooses what to remember")

    # Signpost: what production systems do differently
    print("\nProduction differences:")
    print("  - Parallel scan: prefix-sum over recurrence, O(log n) parallel on GPU")
    print("  - Hardware-aware scan: fuses discretization + scan into one CUDA kernel")
    print("  - HiPPO initialization: structured A matrix for long-range memory")
    print("  - Mamba-2: SSD (Structured State Space Duality) connects SSMs and")
    print("    attention through a shared matrix multiplication framework")

    print(f"\nTotal runtime: {time.time() - start_time:.1f}s")
