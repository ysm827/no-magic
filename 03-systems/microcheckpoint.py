"""
How to train models twice as deep on the same hardware — trading compute for memory
by recomputing activations during the backward pass.
"""
# Reference: Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)
# https://arxiv.org/abs/1604.06174
#
# Standard backpropagation stores every intermediate activation during the forward pass
# so they're available during backward. For an n-layer network, that's O(n) memory.
# Gradient checkpointing stores activations only at sqrt(n) evenly-spaced "checkpoint"
# layers, then recomputes the activations between checkpoints during backward. This
# reduces memory to O(sqrt(n)) at the cost of one extra forward pass per segment
# (~2x compute total).
#
# Signpost: PyTorch implements this as torch.utils.checkpoint.checkpoint(). It's used
# routinely when training large transformers (GPT-3, LLaMA, Gemini) where GPU memory
# is the binding constraint, not compute time.

# === TRADEOFFS ===
# + Reduces activation memory from O(n) to O(sqrt(n)) layers
# + Enables training deeper models on fixed hardware budgets
# + Drop-in compatible: no changes to model architecture or optimizer
# - Approximately doubles training compute (one extra forward pass per segment)
# - Checkpoint placement requires tuning (sqrt(n) spacing is optimal for uniform cost)
# - No benefit for inference (activations are not stored during forward-only passes)
# WHEN TO USE: Training deep models where GPU memory is the binding constraint,
#   not training time. Standard for large transformer training (GPT-3, LLaMA).
# WHEN NOT TO: Shallow models where activation memory is already small, or when
#   training throughput is the bottleneck (the 2x compute cost is unacceptable).

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

N_LAYERS = 10          # deep enough that memory savings are meaningful
HIDDEN_DIM = 8         # width of each hidden layer
N_CLASSES = 4          # number of concentric rings in the dataset
N_SAMPLES = 200        # training examples
LEARNING_RATE = 0.05   # SGD learning rate
NUM_STEPS = 80         # training iterations (both methods run identical steps)

# Checkpoint placement: every sqrt(n) layers gives optimal memory-compute tradeoff.
# For 10 layers, sqrt(10) ~ 3.16, so we checkpoint every 3 layers.
# Checkpointed layers: 0 (input), 3, 6, 9 -- 4 stored vs 11 total.
CHECKPOINT_EVERY = int(math.isqrt(N_LAYERS)) or 1


# === SYNTHETIC DATASET ===
# Concentric rings: each class is a ring at a different radius.
# Simple enough to learn, complex enough to need a deep network.

def make_rings(n_samples: int, n_classes: int) -> tuple[list[list[float]], list[int]]:
    """Generate 2D points arranged in concentric rings, one ring per class.

    Each ring has radius (class_id + 1) with Gaussian noise added.
    This creates a classification problem that requires nonlinear decision
    boundaries -- linear models cannot separate concentric rings.
    """
    xs: list[list[float]] = []
    ys: list[int] = []
    per_class = n_samples // n_classes

    for class_id in range(n_classes):
        radius = (class_id + 1) * 1.0
        for _ in range(per_class):
            angle = random.uniform(0, 2 * math.pi)
            noise = random.gauss(0, 0.15)
            x = (radius + noise) * math.cos(angle)
            y = (radius + noise) * math.sin(angle)
            xs.append([x, y])
            ys.append(class_id)

    return xs, ys


# === SCALAR AUTOGRAD ENGINE ===

_value_count = 0


class Value:
    """Scalar with reverse-mode automatic differentiation and memory tracking.

    Identical to the canonical Value class (see 01-foundations/microgpt.py) with
    one addition: a global counter tracks object creation to measure memory.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple[Value, ...] = (), local_grads: tuple[float, ...] = ()):
        global _value_count
        _value_count += 1
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

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

    def relu(self) -> Value:
        return Value(max(0.0, self.data), (self,), (float(self.data > 0),))

    def exp(self) -> Value:
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def backward(self) -> None:
        """Reverse-mode autodiff: topological sort then chain rule in reverse."""
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
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


def reset_value_count() -> None:
    global _value_count
    _value_count = 0


def get_value_count() -> int:
    return _value_count


# === MLP DEFINITION ===

def make_layer_weights(n_in: int, n_out: int) -> list[list[Value]]:
    """Xavier-initialized weight matrix: std = sqrt(2 / (n_in + n_out)).

    Xavier initialization keeps activation variance roughly constant across layers,
    critical for deep networks. Without it, activations explode or vanish.
    """
    std = math.sqrt(2.0 / (n_in + n_out))
    return [[Value(random.gauss(0, std)) for _ in range(n_in)] for _ in range(n_out)]


def make_bias(n: int) -> list[Value]:
    return [Value(0.0) for _ in range(n)]


def init_mlp(input_dim: int, hidden_dim: int, n_layers: int, output_dim: int) -> dict[str, list]:
    """Initialize a deep MLP: input -> [linear -> ReLU] x n_layers -> linear -> output."""
    params: dict[str, list] = {}
    params['w0'] = make_layer_weights(input_dim, hidden_dim)
    params['b0'] = make_bias(hidden_dim)
    for i in range(1, n_layers):
        params[f'w{i}'] = make_layer_weights(hidden_dim, hidden_dim)
        params[f'b{i}'] = make_bias(hidden_dim)
    params[f'w{n_layers}'] = make_layer_weights(hidden_dim, output_dim)
    params[f'b{n_layers}'] = make_bias(output_dim)
    return params


def get_all_params(params: dict[str, list]) -> list[Value]:
    """Flatten all parameters into a single list for optimizer updates."""
    flat: list[Value] = []
    for key in sorted(params.keys()):
        for row_or_val in params[key]:
            if isinstance(row_or_val, list):
                flat.extend(row_or_val)
            else:
                flat.append(row_or_val)
    return flat


# === LAYER OPERATIONS ===

def linear_forward(x: list[Value], w: list[list[Value]], b: list[Value]) -> list[Value]:
    """Affine transformation: y = Wx + b."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) + b[i] for i, w_row in enumerate(w)]


def relu_forward(x: list[Value]) -> list[Value]:
    return [xi.relu() for xi in x]


def apply_layer(x: list[Value], layer_idx: int, params: dict[str, list]) -> list[Value]:
    """Forward through one layer: linear + ReLU (no ReLU on output layer)."""
    w = params[f'w{layer_idx}']
    b = params[f'b{layer_idx}']
    h = linear_forward(x, w, b)
    if layer_idx < N_LAYERS:
        h = relu_forward(h)
    return h


# === LOSS COMPUTATION ===

def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp to prevent overflow."""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def cross_entropy_loss(logits: list[Value], target: int) -> Value:
    """Cross-entropy loss: -log(softmax(logits)[target])."""
    probs = softmax(logits)
    p = probs[target]
    clamped = max(p.data, 1e-10)
    return Value(math.log(clamped), (p,), (1.0 / clamped,)) * -1


# === STANDARD FORWARD PASS ===
# Stores ALL intermediate activations -- O(n) memory for n layers.

def standard_forward(x: list[Value], params: dict[str, list]) -> list[Value]:
    """Standard forward pass: store every activation for backward.

    Every intermediate result stays alive in the computation graph so backward()
    can use it. For n layers, we store n activation vectors. Memory: O(n).
    """
    h = x
    for i in range(N_LAYERS + 1):
        h = apply_layer(h, i, params)
    return h


# === CHECKPOINTED FORWARD + BACKWARD ===
#
# The core algorithm this script exists to demonstrate.
#
# Standard backprop: forward stores all activations, backward uses them.
#   Memory: O(n), Compute: 1 forward + 1 backward
#
# Checkpointed backprop:
#   Forward: run all layers but only SAVE activations at checkpoint boundaries.
#   Backward: process segments in reverse. For each segment:
#     1. Reload the checkpoint activation at the segment start
#     2. Re-run forward through just that segment (recomputation)
#     3. Run backward through the recomputed segment
#     4. The gradient at the segment input becomes the seed for the next segment
#   Memory: O(sqrt(n)), Compute: ~2x (each layer computed at most twice)

def run_backward_from_seeds(seeds: list[Value]) -> None:
    """Run backward from multiple seed Values that already have .grad set.

    Collects the computation graph from all seeds, topologically sorts,
    and propagates gradients. Used for recomputed segments where the "output"
    is a vector of Values with injected gradients.
    """
    topo: list[Value] = []
    visited: set[int] = set()

    def build_topo(v: Value) -> None:
        vid = id(v)
        if vid not in visited:
            visited.add(vid)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    for s in seeds:
        build_topo(s)

    for v in reversed(topo):
        for child, lg in zip(v._children, v._local_grads):
            child.grad += lg * v.grad


def checkpointed_forward_backward(
    x: list[Value],
    target: int,
    params: dict[str, list],
) -> Value:
    """Complete forward + backward with gradient checkpointing.

    Combines forward and backward because checkpointing fundamentally changes
    how backward works: instead of traversing one connected graph, we process
    independent segments and manually propagate gradients across boundaries.

    Algorithm:
    1. Forward pass using raw floats (no autograd) to compute checkpoint snapshots
    2. Process segments in reverse: recompute each from its checkpoint with autograd,
       then backward. Gradients on shared weight params accumulate correctly since
       each weight appears in exactly one segment.
    """
    total_layers = N_LAYERS + 1

    # Segment boundaries. For 11 layers with CHECKPOINT_EVERY=3:
    # [0,3), [3,6), [6,9), [9,11)
    checkpoint_ids = list(range(0, total_layers, CHECKPOINT_EVERY))
    if checkpoint_ids[-1] != total_layers:
        checkpoint_ids.append(total_layers)

    # --- Phase 1: Forward pass (raw floats, no autograd) ---
    # Compute activations at checkpoint boundaries only. Between checkpoints,
    # activations are computed and immediately discarded. This is where the
    # memory savings come from: we store sqrt(n) snapshots instead of n.
    saved: dict[int, list[float]] = {0: [v.data for v in x]}

    h_data = saved[0][:]
    for layer_i in range(total_layers):
        w = params[f'w{layer_i}']
        b = params[f'b{layer_i}']
        n_out = len(w)
        n_in = len(w[0])
        h_new = [0.0] * n_out
        for r in range(n_out):
            s = b[r].data
            for c in range(n_in):
                s += w[r][c].data * h_data[c]
            if layer_i < N_LAYERS:
                s = max(0.0, s)
            h_new[r] = s
        h_data = h_new
        if (layer_i + 1) in checkpoint_ids:
            saved[layer_i + 1] = list(h_data)

    # --- Phase 2: Backward, segment by segment ---
    # Process segments in reverse. For each segment:
    # 1. Create fresh leaf Values from checkpoint (no graph history)
    # 2. Re-run forward with autograd through this segment
    # 3. Last segment: compute loss + backward from scalar
    #    Earlier segments: inject gradient from next segment, backward from seeds
    # 4. Read gradient on input leaves -> carry to preceding segment
    #
    # Weight parameters are shared across all segments. Each segment's backward
    # accumulates gradients onto the same param Value objects. This is correct
    # because each weight is used in exactly one layer/segment.

    n_segments = len(checkpoint_ids) - 1
    grad_at_boundary: list[float] = []
    loss_data = 0.0

    for seg_idx in range(n_segments - 1, -1, -1):
        seg_start = checkpoint_ids[seg_idx]
        seg_end = checkpoint_ids[seg_idx + 1]

        # Fresh leaf Values from checkpoint snapshot
        seg_input = [Value(v) for v in saved[seg_start]]

        # Recompute this segment with autograd tracking
        h = seg_input
        for layer_i in range(seg_start, seg_end):
            h = apply_layer(h, layer_i, params)

        if seg_idx == n_segments - 1:
            # Last segment: h contains final logits. Compute loss + backward.
            loss = cross_entropy_loss(h, target)
            loss.backward()
            loss_data = loss.data
        else:
            # Inject gradient from next segment and propagate backward
            for hi, g in zip(h, grad_at_boundary):
                hi.grad = g
            run_backward_from_seeds(h)

        # Carry gradient to preceding segment
        grad_at_boundary = [v.grad for v in seg_input]

    return Value(loss_data)


# === TRAINING FUNCTIONS ===

def train_standard(
    raw_xs: list[list[float]],
    ys: list[int],
    params: dict[str, list],
    n_steps: int,
) -> tuple[list[float], float]:
    """Train using standard backpropagation. Returns (loss_history, elapsed_time)."""
    all_params = get_all_params(params)
    loss_history: list[float] = []

    t0 = time.time()
    for step in range(n_steps):
        idx = step % len(raw_xs)
        for p in all_params:
            p.grad = 0.0

        x = [Value(raw_xs[idx][0]), Value(raw_xs[idx][1])]
        logits = standard_forward(x, params)
        loss = cross_entropy_loss(logits, ys[idx])
        loss.backward()

        for p in all_params:
            p.data -= LEARNING_RATE * p.grad

        loss_history.append(loss.data)
        if (step + 1) % 20 == 0 or step == 0:
            print(f"  [standard]   step {step + 1:>3}/{n_steps} | loss: {loss.data:.4f}")

    return loss_history, time.time() - t0


def train_checkpointed(
    raw_xs: list[list[float]],
    ys: list[int],
    params: dict[str, list],
    n_steps: int,
) -> tuple[list[float], float]:
    """Train using checkpointed backpropagation. Returns (loss_history, elapsed_time)."""
    all_params = get_all_params(params)
    loss_history: list[float] = []

    t0 = time.time()
    for step in range(n_steps):
        idx = step % len(raw_xs)
        for p in all_params:
            p.grad = 0.0

        x = [Value(raw_xs[idx][0]), Value(raw_xs[idx][1])]
        loss = checkpointed_forward_backward(x, ys[idx], params)

        for p in all_params:
            p.data -= LEARNING_RATE * p.grad

        loss_history.append(loss.data)
        if (step + 1) % 20 == 0 or step == 0:
            print(f"  [checkpoint] step {step + 1:>3}/{n_steps} | loss: {loss.data:.4f}")

    return loss_history, time.time() - t0


# === GRADIENT VERIFICATION ===

def verify_gradients(raw_x: list[float], target: int) -> tuple[float, bool]:
    """Verify that standard and checkpointed produce identical gradients.

    Both methods compute exact gradients (no approximation), so they must
    match to floating-point precision. Any difference signals a bug.
    """
    random.seed(99)
    params_std = init_mlp(2, HIDDEN_DIM, N_LAYERS, N_CLASSES)
    random.seed(99)
    params_ckpt = init_mlp(2, HIDDEN_DIM, N_LAYERS, N_CLASSES)

    all_std = get_all_params(params_std)
    all_ckpt = get_all_params(params_ckpt)

    # Standard
    for p in all_std:
        p.grad = 0.0
    x_std = [Value(raw_x[0]), Value(raw_x[1])]
    logits_std = standard_forward(x_std, params_std)
    loss_std = cross_entropy_loss(logits_std, target)
    loss_std.backward()

    # Checkpointed
    for p in all_ckpt:
        p.grad = 0.0
    x_ckpt = [Value(raw_x[0]), Value(raw_x[1])]
    _ = checkpointed_forward_backward(x_ckpt, target, params_ckpt)

    max_diff = 0.0
    for p_s, p_c in zip(all_std, all_ckpt):
        diff = abs(p_s.grad - p_c.grad)
        if diff > max_diff:
            max_diff = diff

    return max_diff, max_diff < 1e-6


# === MEMORY ACCOUNTING ===

def measure_memory(raw_x: list[float], target: int, method: str) -> int:
    """Count Value objects created during a forward+backward pass."""
    random.seed(42)
    params = init_mlp(2, HIDDEN_DIM, N_LAYERS, N_CLASSES)
    for p in get_all_params(params):
        p.grad = 0.0

    reset_value_count()
    if method == "standard":
        x = [Value(raw_x[0]), Value(raw_x[1])]
        logits = standard_forward(x, params)
        loss = cross_entropy_loss(logits, target)
        loss.backward()
    else:
        x = [Value(raw_x[0]), Value(raw_x[1])]
        _ = checkpointed_forward_backward(x, target, params)
    return get_value_count()


# === MAIN ===

if __name__ == "__main__":
    print("=" * 65)
    print("Gradient Checkpointing: Trading Compute for Memory")
    print("=" * 65)
    print(f"\nNetwork: {N_LAYERS}-layer MLP, hidden_dim={HIDDEN_DIM}")
    print(f"Dataset: {N_SAMPLES} points, {N_CLASSES} concentric rings")
    print(f"Checkpoint interval: every {CHECKPOINT_EVERY} layers")

    n_checkpoints = len(range(0, N_LAYERS + 1, CHECKPOINT_EVERY))
    print(f"Stored activations: {n_checkpoints} (of {N_LAYERS + 1} total)")
    print(f"Theoretical memory ratio: ~{n_checkpoints / (N_LAYERS + 1):.0%} of standard\n")

    raw_xs, raw_ys = make_rings(N_SAMPLES, N_CLASSES)

    # --- Phase 1: Gradient correctness ---
    print("-" * 65)
    print("Phase 1: Gradient Correctness Verification")
    print("-" * 65)

    n_verify = 5
    all_match = True
    for i in range(n_verify):
        max_diff, match = verify_gradients(raw_xs[i], raw_ys[i])
        status = "PASS" if match else "FAIL"
        print(f"  Sample {i}: max gradient diff = {max_diff:.2e} [{status}]")
        if not match:
            all_match = False

    print(f"\n  Result: {'ALL GRADIENTS MATCH' if all_match else 'GRADIENT MISMATCH DETECTED'}")

    # --- Phase 2: Memory measurement ---
    print(f"\n{'-' * 65}")
    print("Phase 2: Memory Measurement (Value objects created)")
    print("-" * 65)

    mem_std = measure_memory(raw_xs[0], raw_ys[0], "standard")
    mem_ckpt = measure_memory(raw_xs[0], raw_ys[0], "checkpointed")

    print(f"\n  Standard:      {mem_std:>6} Value objects")
    print(f"  Checkpointed:  {mem_ckpt:>6} Value objects")
    # Signpost: checkpointing creates MORE total objects (recomputation allocates
    # new Values) but keeps FEWER alive simultaneously. In a real framework with
    # tensor memory management, peak resident memory drops to O(sqrt(n)). Our
    # counter measures total allocations, not peak residency -- the gradient
    # correctness test is what validates the technique.

    # --- Phase 3: Training comparison ---
    print(f"\n{'-' * 65}")
    print("Phase 3: Training Comparison")
    print("-" * 65)

    print("\n  --- Standard Backpropagation ---")
    random.seed(42)
    params_std = init_mlp(2, HIDDEN_DIM, N_LAYERS, N_CLASSES)
    std_losses, std_time = train_standard(raw_xs, raw_ys, params_std, NUM_STEPS)

    print("\n  --- Checkpointed Backpropagation ---")
    random.seed(42)
    params_ckpt = init_mlp(2, HIDDEN_DIM, N_LAYERS, N_CLASSES)
    ckpt_losses, ckpt_time = train_checkpointed(raw_xs, raw_ys, params_ckpt, NUM_STEPS)

    # --- Results ---
    print(f"\n{'=' * 65}")
    print("RESULTS")
    print("=" * 65)

    print(f"\n  {'Metric':<30} {'Standard':>12} {'Checkpointed':>14}")
    print(f"  {'-' * 56}")
    print(f"  {'Final loss':<30} {std_losses[-1]:>12.4f} {ckpt_losses[-1]:>14.4f}")
    print(f"  {'Training time (s)':<30} {std_time:>12.2f} {ckpt_time:>14.2f}")
    if std_time > 0:
        print(f"  {'Compute overhead':<30} {'1.00x':>12} {ckpt_time / std_time:>13.2f}x")
    print(f"  {'Values created (fwd+bwd)':<30} {mem_std:>12,} {mem_ckpt:>14,}")
    print(f"  {'Checkpoint interval':<30} {'n/a':>12} {'every ' + str(CHECKPOINT_EVERY) + ' layers':>14}")
    print(f"  {'Stored activations':<30} {N_LAYERS + 1:>12} {n_checkpoints:>14}")
    print(f"  {'Gradients match':<30} {'n/a':>12} {'YES' if all_match else 'NO':>14}")

    print(f"""
  Key Insight:
  Gradient checkpointing reduces peak memory from O(n) to O(sqrt(n))
  at the cost of ~2x compute (one extra forward pass per segment).

  For a 96-layer transformer (GPT-3 scale):
  - Standard:      96 activation tensors resident in GPU memory
  - Checkpointed:  ~10 activation tensors (checkpoint every 10 layers)
  - Enables training models that would otherwise exceed GPU memory

  This script demonstrated the mechanism with a {N_LAYERS}-layer MLP.
  PyTorch: torch.utils.checkpoint.checkpoint()
  JAX: jax.checkpoint (formerly jax.remat)
""")
