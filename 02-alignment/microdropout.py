"""
Why randomly destroying neurons during training prevents overfitting — dropout, weight decay,
and early stopping as complementary strategies.
"""
# Reference: Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from
# Overfitting" (2014). The core insight: by randomly zeroing activations during training,
# no single neuron can memorize patterns alone. This forces the network to distribute
# knowledge across many neurons, producing a model that generalizes rather than memorizes.

# === TRADEOFFS ===
# + Prevents co-adaptation: forces distributed representations across neurons
# + Approximates ensemble of exponentially many sub-networks
# + Zero additional parameters: regularization via training-time noise
# - Increases training time (effectively training on partial network each step)
# - Requires scaling at inference time (or inverted dropout during training)
# - Interacts unpredictably with batch normalization (both modify activations)
# WHEN TO USE: When your model has excess capacity relative to data size and is
#   overfitting on training data. Standard for fully connected and attention layers.
# WHEN NOT TO: When the model is underfitting (dropout will make it worse), or
#   in convolutional layers where spatial dropout is more appropriate.

from __future__ import annotations

import math
import os
import random
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture — deliberately large relative to the data so it CAN overfit.
# A model that can't overfit can't demonstrate regularization effects.
N_EMBD = 16          # embedding dimension
N_HIDDEN = 64        # hidden layer size (4x embedding = substantial excess capacity)
CONTEXT_SIZE = 3     # number of preceding characters used as input

# Training parameters
LEARNING_RATE = 0.1
NUM_STEPS = 2000      # enough steps for the unregularized model to memorize
MAX_NAMES = 50        # tiny subset — forces overfitting with excess model capacity

# Regularization hyperparameters
DROPOUT_P = 0.3       # probability of zeroing each hidden activation
WEIGHT_DECAY = 0.001  # L2 penalty coefficient (lambda)

# Early stopping
EARLY_STOP_PATIENCE = 5   # consecutive checks with rising val loss before stopping
EVAL_INTERVAL = 200       # steps between validation loss checks

# Data parameters
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: ~5,300 parameters with 27 vocab chars on only 50 names (~300 examples).
# This is massively oversized — nearly 18 parameters per training example. Production
# character models use larger architectures with much more data where overfitting is
# less extreme. Here, the excess capacity guarantees overfitting so we can clearly
# measure the regularization effect. Scalar autograd limits our model size; production
# systems use tensor-level ops (PyTorch, JAX) that are 1000x faster.


# === DATA LOADING AND SPLITTING ===

def load_data(url: str, filename: str) -> list[str]:
    """Download and parse the training corpus."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)

    with open(filename, "r") as f:
        names = [line.strip() for line in f if line.strip()]

    return names


def build_dataset(
    names: list[str],
    stoi: dict[str, int],
    context_size: int,
) -> tuple[list[list[int]], list[int]]:
    """Build (context, target) pairs from names using a sliding window.

    Each training example is a fixed-size context window of character indices
    followed by the target character index. Names are padded with '.' (the
    boundary token) so the model learns to start and end names.

    Example with context_size=3: "emma" -> [(..., ..., .), e], [(..., ., e), m], ...
    """
    xs: list[list[int]] = []
    ys: list[int] = []
    for name in names:
        padded = ['.'] * context_size + list(name) + ['.']
        for i in range(len(padded) - context_size):
            context = [stoi[ch] for ch in padded[i : i + context_size]]
            target = stoi[padded[i + context_size]]
            xs.append(context)
            ys.append(target)
    return xs, ys


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Tracks computational history via ._children and ._local_grads, enabling
    gradient computation through the chain rule. Every forward operation stores
    its local derivative (dout/dinput) as a closure, then backward() replays
    the computation graph in reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()):
        self.data = float(data)
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

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return other + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * (other ** -1)

    def __rtruediv__(self, other: float) -> Value:
        return other * (self ** -1)

    def tanh(self) -> Value:
        # d(tanh(x))/dx = 1 - tanh(x)^2
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self) -> Value:
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self) -> Value:
        # d(relu(x))/dx = 1 if x > 0 else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self) -> None:
        """Reverse-mode autodiff via topological sort of the computation graph."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


def safe_log(prob: Value) -> Value:
    """Clipped log to prevent log(0) = -inf from breaking gradients."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === MODEL DEFINITION ===

# Architecture: character-level MLP with one hidden layer.
# Input: CONTEXT_SIZE character indices -> concatenated embeddings -> hidden layer -> output logits.
# This is essentially Bengio et al. (2003) neural language model, simple enough to overfit
# quickly but complex enough that regularization makes a measurable difference.

def make_matrix(nrows: int, ncols: int, std: float = 0.1) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_model(vocab_size: int) -> dict[str, list[list[Value]]]:
    """Initialize model parameters: embeddings, hidden layer, output layer."""
    params: dict[str, list[list[Value]]] = {}

    # Character embeddings: [vocab_size, n_embd]
    params['emb'] = make_matrix(vocab_size, N_EMBD)

    # Hidden layer: [n_hidden, context_size * n_embd]
    # Takes concatenated context embeddings as input.
    params['w1'] = make_matrix(N_HIDDEN, CONTEXT_SIZE * N_EMBD)
    params['b1'] = make_matrix(1, N_HIDDEN)

    # Output layer: [vocab_size, n_hidden]
    # Projects hidden activations to logits over vocabulary.
    params['w2'] = make_matrix(vocab_size, N_HIDDEN)
    params['b2'] = make_matrix(1, vocab_size)

    return params


def get_all_params(params: dict[str, list[list[Value]]]) -> list[Value]:
    """Flatten all parameter matrices into a single list for optimizer access."""
    return [p for matrix in params.values() for row in matrix for p in row]


# === REGULARIZATION IMPLEMENTATIONS ===

def apply_dropout(
    activations: list[Value],
    drop_prob: float,
    training: bool,
) -> list[Value]:
    """Apply inverted dropout to a layer's activations.

    During training: randomly zero each activation with probability drop_prob,
    then scale surviving activations by 1/(1-p) to maintain expected value.

    During inference: use all activations unchanged. The 1/(1-p) scaling during
    training ("inverted dropout") means no correction is needed at test time.

    Math: h_drop[i] = h[i] * mask[i] / (1 - p) where mask[i] ~ Bernoulli(1 - p)

    Intuition: dropout forces redundancy. If any neuron might be absent, the network
    can't rely on a single neuron to encode a pattern. Instead, information is spread
    across many neurons — a kind of learned ensemble. Srivastava et al. showed this is
    approximately equivalent to averaging over 2^n subnetworks (where n = number of
    hidden units), which is a massive implicit ensemble at zero extra inference cost.

    Signpost: production transformers typically use dropout=0.1. The higher rate here
    (0.3) is needed because our model is small and overfits aggressively. Larger models
    have better implicit regularization from sheer parameter count.
    """
    if not training or drop_prob == 0.0:
        return activations

    scale = 1.0 / (1.0 - drop_prob)
    result: list[Value] = []
    for act in activations:
        if random.random() < drop_prob:
            # Zero this activation — it's "dropped out" for this training step.
            # We create a new Value(0) disconnected from the graph, so no gradient
            # flows back through this neuron. This is the key mechanism: the neuron
            # receives no gradient update when dropped, preventing co-adaptation.
            result.append(Value(0.0))
        else:
            # Scale surviving activation to compensate for missing neurons.
            # Without scaling, the expected sum of activations would shrink by (1-p),
            # causing a train/test mismatch.
            result.append(act * scale)
    return result


class EarlyStopper:
    """Monitor validation loss and signal when to stop training.

    Tracks the best validation loss seen so far. If validation loss fails to
    improve for `patience` consecutive checks, signals that training should stop.

    This prevents the model from continuing to memorize training data after it
    has already learned the generalizable patterns. The validation loss curve
    typically has a U-shape: it decreases as the model learns real patterns,
    reaches a minimum, then increases as the model starts memorizing noise.
    Early stopping halts training near the minimum.

    Signpost: production systems often save model checkpoints at the best
    validation loss and restore that checkpoint after stopping. We skip
    checkpointing here — the goal is demonstrating the effect, not building
    a production training loop.
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0

    def check(self, val_loss: float) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# === FORWARD PASS AND LOSS ===

def forward(
    context: list[int],
    params: dict[str, list[list[Value]]],
    training: bool = True,
    use_dropout: bool = False,
    drop_prob: float = 0.0,
) -> list[Value]:
    """Forward pass through the character-level MLP.

    1. Look up embeddings for each context character
    2. Concatenate into a single input vector
    3. Hidden layer: linear transform -> tanh activation -> optional dropout
    4. Output layer: linear transform -> logits

    The architecture is simple enough that each regularization technique's
    effect is clearly visible in the train/val loss gap.
    """
    # Step 1: Embedding lookup and concatenation
    # Each context character becomes an N_EMBD-dimensional vector; we concatenate
    # them into a single (CONTEXT_SIZE * N_EMBD)-dimensional input.
    emb_concat: list[Value] = []
    for idx in context:
        emb_concat.extend(params['emb'][idx])

    # Step 2: Hidden layer — h = tanh(W1 @ x + b1)
    hidden: list[Value] = []
    w1 = params['w1']
    b1 = params['b1'][0]
    for i in range(N_HIDDEN):
        act = b1[i]
        for j in range(len(emb_concat)):
            act = act + w1[i][j] * emb_concat[j]
        hidden.append(act.tanh())

    # Step 3: Apply dropout to hidden activations (only during training if enabled)
    if use_dropout:
        hidden = apply_dropout(hidden, drop_prob, training)

    # Step 4: Output layer — logits = W2 @ h + b2
    w2 = params['w2']
    b2 = params['b2'][0]
    logits: list[Value] = []
    for i in range(len(w2)):
        act = b2[i]
        for j in range(N_HIDDEN):
            act = act + w2[i][j] * hidden[j]
        logits.append(act)

    return logits


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp to prevent overflow."""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def eval_loss(
    xs: list[list[int]],
    ys: list[int],
    params: dict[str, list[list[Value]]],
) -> float:
    """Evaluate loss without dropout (inference mode) over all examples.

    No dropout applied — at inference time, all neurons are active.
    With 200 names (~1500 total examples), evaluating the full set is feasible
    and gives accurate loss estimates without sampling noise.
    """
    total = 0.0
    for i in range(len(xs)):
        logits = forward(xs[i], params, training=False, use_dropout=False)
        # Compute softmax on raw floats (no autograd needed for evaluation)
        max_val = max(v.data for v in logits)
        exp_vals = [math.exp(v.data - max_val) for v in logits]
        exp_sum = sum(exp_vals)
        prob_target = max(exp_vals[ys[i]] / exp_sum, 1e-10)
        total += -math.log(prob_target)

    return total / len(xs)


# === TRAINING LOOP ===

def train_model(
    train_xs: list[list[int]],
    train_ys: list[int],
    val_xs: list[list[int]],
    val_ys: list[int],
    vocab_size: int,
    config_name: str,
    use_dropout: bool = False,
    drop_prob: float = 0.0,
    use_weight_decay: bool = False,
    weight_decay_lam: float = 0.0,
    use_early_stopping: bool = False,
) -> tuple[float, float, int]:
    """Train one model configuration and return (train_loss, val_loss, steps_run).

    Each configuration starts from a fresh random initialization (re-seeded for
    fair comparison) and trains for up to NUM_STEPS steps. The only difference
    between runs is the regularization strategy applied.
    """
    # Re-seed for fair comparison: each config starts from identical random state.
    # Without this, different weight initializations would confound the comparison.
    random.seed(42)

    params = init_model(vocab_size)
    param_list = get_all_params(params)

    # SGD keeps the regularization effects cleanest — no optimizer-level implicit
    # regularization (Adam's adaptive rates act as a form of implicit regularization).
    stopper = EarlyStopper(EARLY_STOP_PATIENCE) if use_early_stopping else None
    steps_run = 0

    print(f"\n--- {config_name} ---")

    for step in range(NUM_STEPS):
        # Single-example SGD (stochastic gradient descent in the truest sense).
        # Scalar autograd makes batching expensive — each example builds a full
        # computation graph. Single-example updates are noisy but fast.
        idx = random.randint(0, len(train_xs) - 1)

        # Forward pass
        logits = forward(train_xs[idx], params, training=True,
                         use_dropout=use_dropout, drop_prob=drop_prob)
        probs = softmax(logits)
        loss = -safe_log(probs[train_ys[idx]])

        # Backward pass
        loss.backward()

        # SGD update with linear learning rate decay
        lr_t = LEARNING_RATE * (1.0 - step / NUM_STEPS)
        for p in param_list:
            # Weight decay (L2 regularization): add lambda * theta to the gradient.
            # Math: L_total = L_data + (lambda/2) * sum(theta^2)
            # Gradient: dL/d(theta) = dL_data/d(theta) + lambda * theta
            # So the update becomes: theta -= lr * (grad + lambda * theta)
            #
            # Intuition: large weights mean the model is very sensitive to small
            # input changes — it has memorized specific training examples. Weight
            # decay shrinks weights toward zero each step, pushing the model toward
            # simpler solutions with smoother decision boundaries.
            #
            # We apply weight decay directly in the update rather than through the
            # autograd graph because building sum-of-squares over thousands of Value
            # nodes per step is prohibitively expensive in scalar autograd.
            if use_weight_decay:
                p.data -= lr_t * (p.grad + weight_decay_lam * p.data)
            else:
                p.data -= lr_t * p.grad
            p.grad = 0.0

        steps_run = step + 1

        # Periodic evaluation
        if (step + 1) % EVAL_INTERVAL == 0:
            t_loss = eval_loss(train_xs, train_ys, params)
            v_loss = eval_loss(val_xs, val_ys, params)
            print(f"  step {step + 1:>4}/{NUM_STEPS} | train: {t_loss:.4f} | val: {v_loss:.4f}")

            # Early stopping check
            if stopper is not None and stopper.check(v_loss):
                print(f"  ** early stopping at step {step + 1} (patience={EARLY_STOP_PATIENCE})")
                break

    # Final evaluation
    final_train = eval_loss(train_xs, train_ys, params)
    final_val = eval_loss(val_xs, val_ys, params)
    print(f"  final    | train: {final_train:.4f} | val: {final_val:.4f}")

    return final_train, final_val, steps_run


# === COMPARISON AND RESULTS ===

if __name__ == "__main__":
    # -- Load and prepare data --
    print("Loading data...")
    names = load_data(DATA_URL, DATA_FILE)

    # Build vocabulary: 26 lowercase letters + '.' boundary token
    chars = sorted(set(''.join(names)))
    chars = ['.'] + chars  # '.' at index 0 as the boundary/padding token
    stoi = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    # Use a subset of names for tractable training with scalar autograd.
    # Smaller training set makes overfitting more pronounced, which is exactly
    # what we want — it amplifies the regularization signal.
    random.shuffle(names)
    names = names[:MAX_NAMES]

    print(f"Using {len(names)} names (subset), vocabulary size: {vocab_size}")

    # Split data 80/20 before building examples.
    # Splitting names (not examples) prevents data leakage: no name appears
    # in both train and validation, so the model can't memorize specific names
    # during training and get credit for them during validation.
    split_idx = int(0.8 * len(names))
    train_names = names[:split_idx]
    val_names = names[split_idx:]

    train_xs, train_ys = build_dataset(train_names, stoi, CONTEXT_SIZE)
    val_xs, val_ys = build_dataset(val_names, stoi, CONTEXT_SIZE)

    print(f"Training examples: {len(train_xs)}, Validation examples: {len(val_xs)}")
    print(f"Parameters per model: {vocab_size * N_EMBD + N_HIDDEN * CONTEXT_SIZE * N_EMBD + N_HIDDEN + vocab_size * N_HIDDEN + vocab_size}")

    # -- Run five training configurations --
    # Each configuration uses identical architecture and initialization (via re-seeding).
    # The only variable is the regularization strategy. This controlled experiment
    # isolates the effect of each technique.

    results: list[tuple[str, float, float, int]] = []

    # Config 1: No regularization (baseline — expected to overfit)
    t, v, s = train_model(
        train_xs, train_ys, val_xs, val_ys, vocab_size,
        config_name="No regularization (baseline)",
    )
    results.append(("No regularization", t, v, s))

    # Config 2: Dropout only
    t, v, s = train_model(
        train_xs, train_ys, val_xs, val_ys, vocab_size,
        config_name=f"Dropout only (p={DROPOUT_P})",
        use_dropout=True, drop_prob=DROPOUT_P,
    )
    results.append((f"Dropout (p={DROPOUT_P})", t, v, s))

    # Config 3: Weight decay only
    t, v, s = train_model(
        train_xs, train_ys, val_xs, val_ys, vocab_size,
        config_name=f"Weight decay only (lambda={WEIGHT_DECAY})",
        use_weight_decay=True, weight_decay_lam=WEIGHT_DECAY,
    )
    results.append((f"Weight decay (l={WEIGHT_DECAY})", t, v, s))

    # Config 4: Dropout + weight decay (combined)
    t, v, s = train_model(
        train_xs, train_ys, val_xs, val_ys, vocab_size,
        config_name=f"Dropout + weight decay (p={DROPOUT_P}, lambda={WEIGHT_DECAY})",
        use_dropout=True, drop_prob=DROPOUT_P,
        use_weight_decay=True, weight_decay_lam=WEIGHT_DECAY,
    )
    results.append(("Dropout + weight decay", t, v, s))

    # Config 5: Early stopping (monitors validation loss, stops when it rises)
    t, v, s = train_model(
        train_xs, train_ys, val_xs, val_ys, vocab_size,
        config_name=f"Early stopping (patience={EARLY_STOP_PATIENCE})",
        use_early_stopping=True,
    )
    results.append(("Early stopping", t, v, s))

    # -- Print comparison table --
    # The "gap" column is the key metric: it measures how much worse the model
    # performs on unseen data compared to training data. A large gap = overfitting.
    # Effective regularization reduces this gap while keeping validation loss low.
    print("\n\n" + "=" * 78)
    print("REGULARIZATION COMPARISON")
    print("=" * 78)
    print(f"{'Strategy':<28} {'Train':>8} {'Val':>8} {'Gap':>8} {'Steps':>7}")
    print("-" * 78)
    for name, train_loss, val_loss, steps in results:
        gap = val_loss - train_loss
        print(f"{name:<28} {train_loss:>8.4f} {val_loss:>8.4f} {gap:>+8.4f} {steps:>7}")
    print("-" * 78)

    # Identify best configuration by validation loss
    best = min(results, key=lambda r: r[2])
    print(f"\nBest generalization: {best[0]} (val loss: {best[2]:.4f})")

    # -- Interpretation guide --
    print("\nInterpretation:")
    print("  - Gap = val_loss - train_loss. Positive gap = overfitting.")
    print("  - Baseline (no regularization) should show the largest gap.")
    print("  - Dropout forces redundancy: no single neuron can memorize a pattern.")
    print("  - Weight decay shrinks weights toward zero, preferring simpler models.")
    print("  - Early stopping halts before the model memorizes training noise.")
    print("  - Combined strategies typically achieve the best generalization.")
