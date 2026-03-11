"""
How normalizing activations within each mini-batch stabilizes training and enables deeper networks.
"""
# Reference: Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training
# by Reducing Internal Covariate Shift" (2015). The core insight: each layer's input
# distribution shifts as the preceding layers' weights change during training ("internal
# covariate shift"). Normalizing per-layer activations to zero mean and unit variance
# removes this coupling, allowing higher learning rates and faster convergence.

# === TRADEOFFS ===
# + Enables higher learning rates by stabilizing activation distributions
# + Acts as implicit regularization (batch noise prevents overfitting)
# + Reduces sensitivity to weight initialization
# - Behavior differs between training and inference (running stats vs. batch stats)
# - Breaks down with small batch sizes (noisy statistics)
# - Introduces cross-sample dependency within a batch (problematic for some tasks)
# WHEN TO USE: Deep CNNs and MLPs where training instability or vanishing
#   gradients are limiting depth. Standard for computer vision architectures.
# WHEN NOT TO: Sequence models (use LayerNorm), batch size < 8, or online
#   learning where single-sample updates are required.

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Dataset
N_CLASSES = 5          # number of concentric rings
N_SAMPLES_PER_CLASS = 60   # points per ring (300 total — scalar autograd is O(params²) per sample)
NOISE_STD = 0.15       # radial noise (higher = harder classification)

# Architecture — 5-layer MLP, deep enough that BN matters
HIDDEN_DIM = 16        # neurons per hidden layer (kept small for scalar autograd tractability)
N_HIDDEN_LAYERS = 5    # depth where vanishing/exploding gradients become real

# Training
BATCH_SIZE = 16        # mini-batch size (BN needs ≥8 for stable statistics)
LEARNING_RATE = 0.05   # SGD learning rate (BN allows higher LR — that's the point)
NUM_EPOCHS = 15        # training epochs
BN_MOMENTUM = 0.1      # exponential moving average decay for running stats
BN_EPSILON = 1e-5       # numerical stability in variance normalization


# === SYNTHETIC DATA GENERATION ===

# Concentric rings: each class is a circle at radius r = class_idx + 1, with
# Gaussian noise added to the radius. This creates a non-linearly-separable
# problem that requires multiple layers to solve — a single hyperplane can't
# separate nested rings.

def generate_rings(n_classes: int, n_per_class: int, noise: float) -> tuple[list[list[float]], list[int]]:
    """Generate 2D concentric ring dataset.

    Each ring has radius proportional to its class index. Points are placed
    uniformly around the circle with Gaussian radial noise. The result is
    n_classes nested annuli that require nonlinear decision boundaries.
    """
    xs: list[list[float]] = []
    ys: list[int] = []
    for cls in range(n_classes):
        radius = 1.0 + cls * 0.8  # rings at r=1.0, 1.8, 2.6, 3.4, 4.2
        for _ in range(n_per_class):
            angle = random.uniform(0, 2 * math.pi)
            r = radius + random.gauss(0, noise)
            xs.append([r * math.cos(angle), r * math.sin(angle)])
            ys.append(cls)
    return xs, ys


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """Scalar with reverse-mode automatic differentiation. See microgpt.py for detailed docs."""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple[Value, ...] = (), local_grads: tuple[float, ...] = ()):
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
        """Reverse-mode autodiff via topological sort then chain rule."""
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


# === BATCH NORMALIZATION LAYER ===

# The key idea: during training, each mini-batch provides an estimate of the
# activation distribution's mean and variance. We normalize using these batch
# statistics, then apply learnable scale (gamma) and shift (beta) so the layer
# can recover the identity transform if that's optimal.
#
# Why this works: without BN, a layer's output distribution depends on ALL
# preceding layers' weights. A small weight update in layer 1 cascades through
# layers 2-5, changing their input distributions. Each layer must constantly
# re-adapt to a moving target. BN decouples layers by fixing the input
# distribution to N(0,1) (modulo the learnable gamma/beta).

class BatchNormLayer:
    """Batch normalization for a single feature dimension across a mini-batch.

    Forward (training):
        mu_B = (1/m) * sum(x_i)
        var_B = (1/m) * sum((x_i - mu_B)^2)
        x_hat_i = (x_i - mu_B) / sqrt(var_B + epsilon)
        y_i = gamma * x_hat_i + beta

    Forward (eval):
        Uses running_mean and running_var instead of batch statistics.

    Running stats update:
        running_mean = (1 - momentum) * running_mean + momentum * mu_B
        running_var  = (1 - momentum) * running_var  + momentum * var_B
    """

    def __init__(self, n_features: int):
        # Learnable parameters: gamma (scale) and beta (shift).
        # Initialized to gamma=1, beta=0 so BN starts as an identity-like
        # transform (just normalization, no rescaling).
        self.gamma: list[Value] = [Value(1.0) for _ in range(n_features)]
        self.beta: list[Value] = [Value(0.0) for _ in range(n_features)]

        # Running statistics for eval mode — exponential moving average of
        # batch statistics seen during training.
        self.running_mean: list[float] = [0.0] * n_features
        self.running_var: list[float] = [1.0] * n_features

        self.n_features = n_features

    def forward(self, batch: list[list[Value]], training: bool = True) -> list[list[Value]]:
        """Apply batch normalization across a mini-batch.

        Args:
            batch: list of m samples, each a list of n_features Value nodes
            training: if True, use batch statistics; if False, use running stats

        Returns:
            Normalized batch with same shape as input
        """
        m = len(batch)
        out: list[list[Value]] = [[] for _ in range(m)]

        # Normalize each feature independently across the batch.
        # This per-feature normalization is what makes BN different from
        # normalizing the entire activation vector (that's LayerNorm).
        for j in range(self.n_features):
            if training:
                # Compute batch mean: mu_B = (1/m) * sum_i(x_ij)
                batch_mean = sum(batch[i][j] for i in range(m)) * (1.0 / m)

                # Compute batch variance: var_B = (1/m) * sum_i((x_ij - mu_B)^2)
                # Using population variance (1/m not 1/(m-1)) per the original paper.
                batch_var = sum((batch[i][j] - batch_mean) ** 2 for i in range(m)) * (1.0 / m)

                # Update running statistics for eval mode.
                # Exponential moving average ensures recent batches have more influence
                # than older ones, adapting to distribution drift during training.
                self.running_mean[j] = (1 - BN_MOMENTUM) * self.running_mean[j] + BN_MOMENTUM * batch_mean.data
                self.running_var[j] = (1 - BN_MOMENTUM) * self.running_var[j] + BN_MOMENTUM * batch_var.data

                # Normalize: x_hat = (x - mu) / sqrt(var + eps)
                # The epsilon prevents division by zero when all inputs in the batch
                # are identical (variance = 0). Without it, gradients become inf.
                inv_std = (batch_var + BN_EPSILON) ** -0.5

                for i in range(m):
                    x_hat = (batch[i][j] - batch_mean) * inv_std
                    # Scale and shift: y = gamma * x_hat + beta
                    # These learnable parameters let BN represent any affine transform
                    # of the normalized output. If gamma = sqrt(var) and beta = mean,
                    # BN becomes a no-op — so BN never reduces representational power.
                    y = self.gamma[j] * x_hat + self.beta[j]
                    out[i].append(y)
            else:
                # Eval mode: use accumulated running statistics instead of batch stats.
                # At test time, we may process single samples (m=1), so batch statistics
                # are meaningless. Running stats represent the training distribution.
                inv_std_eval = (self.running_var[j] + BN_EPSILON) ** -0.5
                for i in range(m):
                    x_hat = (batch[i][j] - self.running_mean[j]) * inv_std_eval
                    y = self.gamma[j] * x_hat + self.beta[j]
                    out[i].append(y)

        return out

    def parameters(self) -> list[Value]:
        return self.gamma + self.beta


# === MODEL DEFINITION ===

# Architecture: Input(2) → [Linear → BN → ReLU] × 5 → Linear → Output(N_CLASSES)
#
# Placement: BN goes between Linear and ReLU (Linear → BN → ReLU).
# This is the original paper's recommendation. The linear layer's output is
# pre-activation — BN normalizes it before the nonlinearity. This matters
# because ReLU kills negative values: if the pre-activation distribution
# drifts negative, most neurons die. BN centers activations around zero,
# ensuring roughly half survive ReLU.

def make_layer_weights(n_in: int, n_out: int) -> list[list[Value]]:
    """Xavier/Glorot initialization: std = sqrt(2 / (n_in + n_out)).

    Keeps variance roughly constant across layers, preventing the forward pass
    from exploding or vanishing. Without this, a 5-layer MLP's activations
    would either blow up (if weights are too large) or shrink to zero
    (if too small). BN mitigates this, but good initialization still helps.
    """
    std = math.sqrt(2.0 / (n_in + n_out))
    return [[Value(random.gauss(0, std)) for _ in range(n_in)] for _ in range(n_out)]


def make_bias(n: int) -> list[Value]:
    return [Value(0.0) for _ in range(n)]


class MLP:
    """5-layer MLP with optional batch normalization."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, use_bn: bool):
        self.use_bn = use_bn
        self.weights: list[list[list[Value]]] = []
        self.biases: list[list[Value]] = []
        self.bn_layers: list[BatchNormLayer] = []

        # Build hidden layers
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        for i in range(len(dims) - 1):
            self.weights.append(make_layer_weights(dims[i], dims[i + 1]))
            self.biases.append(make_bias(dims[i + 1]))
            # BN on hidden layers only (not the output layer)
            if use_bn and i < len(dims) - 2:
                self.bn_layers.append(BatchNormLayer(dims[i + 1]))

    def forward(self, batch: list[list[Value]], training: bool = True) -> list[list[Value]]:
        """Forward pass for a batch of inputs. training controls BN behavior."""
        n_hidden = len(self.weights) - 1  # all layers except output

        for layer_idx in range(len(self.weights)):
            w = self.weights[layer_idx]
            b = self.biases[layer_idx]

            # Linear transform: y = Wx + b for each sample in the batch
            batch = [
                [sum(w[j][k] * x[k] for k in range(len(x))) + b[j] for j in range(len(w))]
                for x in batch
            ]

            # Hidden layers get BN (if enabled) + ReLU; output layer is raw logits
            if layer_idx < n_hidden:
                if self.use_bn:
                    batch = self.bn_layers[layer_idx].forward(batch, training=training)
                batch = [[v.relu() for v in sample] for sample in batch]

        return batch

    def parameters(self) -> list[Value]:
        params: list[Value] = []
        for w, b in zip(self.weights, self.biases):
            for row in w:
                params.extend(row)
            params.extend(b)
        if self.use_bn:
            for bn in self.bn_layers:
                params.extend(bn.parameters())
        return params


# === LOSS AND ACCURACY ===

def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp to prevent overflow."""
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy_loss(logits_batch: list[list[Value]], targets: list[int]) -> Value:
    """Average cross-entropy loss over the batch.

    Loss = -(1/m) * sum_i log(softmax(logits_i)[target_i])
    """
    m = len(logits_batch)
    total_loss = Value(0.0)
    for logits, target in zip(logits_batch, targets):
        probs = softmax(logits)
        # Clamp to prevent log(0)
        p = probs[target]
        clamped = max(p.data, 1e-10)
        log_p = Value(math.log(clamped), (p,), (1.0 / clamped,))
        total_loss = total_loss + (-log_p)
    return total_loss * (1.0 / m)


def accuracy(logits_batch: list[list[Value]], targets: list[int]) -> float:
    """Classification accuracy: fraction of correct predictions."""
    correct = 0
    for logits, target in zip(logits_batch, targets):
        pred = max(range(len(logits)), key=lambda k: logits[k].data)
        if pred == target:
            correct += 1
    return correct / len(targets)


# === TRAINING ===

def train_model(
    model: MLP,
    x_train: list[list[float]],
    y_train: list[int],
    num_epochs: int,
    batch_size: int,
    lr: float,
    label: str,
) -> list[tuple[float, float]]:
    """Train a model with SGD + LR decay, return per-epoch (loss, accuracy) history."""
    params = model.parameters()
    history: list[tuple[float, float]] = []
    n = len(x_train)
    indices = list(range(n))

    for epoch in range(num_epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        epoch_correct = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            m = len(batch_idx)

            # Skip batches that are too small for meaningful BN statistics.
            # With m=1, batch variance is always 0, making normalization degenerate.
            if m < 4:
                continue

            # Convert raw floats to Value nodes for autograd tracking
            x_batch = [[Value(x_train[i][d]) for d in range(len(x_train[i]))] for i in batch_idx]
            y_batch = [y_train[i] for i in batch_idx]

            # Forward pass (training mode — uses batch statistics for BN)
            logits = model.forward(x_batch, training=True)

            # Compute loss
            loss = cross_entropy_loss(logits, y_batch)

            # Backward pass
            loss.backward()

            # SGD update with linear learning rate decay
            lr_t = lr * (1.0 - epoch / num_epochs)
            for p in params:
                p.data -= lr_t * p.grad
                p.grad = 0.0

            # Track statistics
            epoch_loss += loss.data * m
            for logit_vec, target in zip(logits, y_batch):
                pred = max(range(len(logit_vec)), key=lambda k: logit_vec[k].data)
                if pred == target:
                    epoch_correct += 1

        avg_loss = epoch_loss / n
        avg_acc = epoch_correct / n
        history.append((avg_loss, avg_acc))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{label}] epoch {epoch + 1:>3}/{num_epochs} | loss: {avg_loss:.4f} | acc: {avg_acc:.3f}")

    return history


# === EVAL MODE DEMONSTRATION ===

def eval_model(model: MLP, x_data: list[list[float]], y_data: list[int]) -> float:
    """Evaluate model accuracy in inference mode (running stats for BN)."""
    x_batch = [[Value(x_data[i][d]) for d in range(len(x_data[i]))] for i in range(len(x_data))]
    logits = model.forward(x_batch, training=False)
    return accuracy(logits, y_data)


# === LAYER NORMALIZATION (COMPARISON) ===

# BN normalizes across the batch for each feature:  stats over samples, per feature
# LN normalizes across features for each sample:    stats over features, per sample
# Transformers use LN because token positions have different semantic roles, making
# batch statistics across positions meaningless. CNNs use BN because spatial features
# (e.g. edge detectors) have consistent meaning across samples.

def layer_norm_forward(x: list[Value], gamma: list[Value], beta: list[Value]) -> list[Value]:
    """LayerNorm: x_hat_j = (x_j - mu) / sqrt(var + eps), y_j = gamma_j * x_hat_j + beta_j"""
    n = len(x)
    mean = sum(x) * (1.0 / n)
    var = sum((xi - mean) ** 2 for xi in x) * (1.0 / n)
    inv_std = (var + BN_EPSILON) ** -0.5
    return [gamma[j] * (x[j] - mean) * inv_std + beta[j] for j in range(n)]


# === MAIN ===

if __name__ == "__main__":
    print("=" * 65)
    print("  Batch Normalization: Stabilizing Deep Network Training")
    print("=" * 65)

    # --- Generate dataset ---
    print(f"\nGenerating concentric rings dataset: {N_CLASSES} classes, "
          f"{N_SAMPLES_PER_CLASS} samples/class")
    x_all, y_all = generate_rings(N_CLASSES, N_SAMPLES_PER_CLASS, NOISE_STD)
    n_total = len(x_all)

    # Train/test split (80/20)
    split = int(0.8 * n_total)
    perm = list(range(n_total))
    random.shuffle(perm)
    train_idx, test_idx = perm[:split], perm[split:]
    x_train = [x_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
    x_test = [x_all[i] for i in test_idx]
    y_test = [y_all[i] for i in test_idx]
    print(f"Train: {len(x_train)}, Test: {len(x_test)}")

    # --- Train WITHOUT batch normalization ---
    print(f"\n{'─' * 65}")
    print("Training 5-layer MLP WITHOUT batch normalization")
    print(f"{'─' * 65}")
    random.seed(42)  # reset seed for fair comparison
    model_no_bn = MLP(2, HIDDEN_DIM, N_CLASSES, N_HIDDEN_LAYERS, use_bn=False)
    t0 = time.time()
    history_no_bn = train_model(model_no_bn, x_train, y_train, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, "no BN")
    time_no_bn = time.time() - t0

    # --- Train WITH batch normalization ---
    print(f"\n{'─' * 65}")
    print("Training 5-layer MLP WITH batch normalization")
    print(f"{'─' * 65}")
    random.seed(42)  # same initialization for fair comparison
    model_bn = MLP(2, HIDDEN_DIM, N_CLASSES, N_HIDDEN_LAYERS, use_bn=True)
    t0 = time.time()
    history_bn = train_model(model_bn, x_train, y_train, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, "BN")
    time_bn = time.time() - t0

    # --- Evaluate on test set ---
    print(f"\n{'─' * 65}")
    print("Evaluation (test set, inference mode)")
    print(f"{'─' * 65}")
    test_acc_no_bn = eval_model(model_no_bn, x_test, y_test)
    test_acc_bn = eval_model(model_bn, x_test, y_test)

    # --- Layer Normalization comparison ---
    # Brief demonstration: apply LayerNorm to a single hidden layer's output
    # to show the API difference and when you'd prefer LN over BN.
    print(f"\n{'─' * 65}")
    print("Layer Normalization (single-sample comparison)")
    print(f"{'─' * 65}")
    ln_gamma = [Value(1.0) for _ in range(HIDDEN_DIM)]
    ln_beta = [Value(0.0) for _ in range(HIDDEN_DIM)]
    sample_input = [Value(random.gauss(0, 2.0)) for _ in range(HIDDEN_DIM)]
    ln_output = layer_norm_forward(sample_input, ln_gamma, ln_beta)

    # Verify normalization: output should have ~zero mean and ~unit variance
    ln_mean = sum(v.data for v in ln_output) / HIDDEN_DIM
    ln_var = sum((v.data - ln_mean) ** 2 for v in ln_output) / HIDDEN_DIM
    print(f"  Input  mean: {sum(v.data for v in sample_input) / HIDDEN_DIM:+.4f}, "
          f"var: {sum((v.data - sum(v2.data for v2 in sample_input) / HIDDEN_DIM) ** 2 for v in sample_input) / HIDDEN_DIM:.4f}")
    print(f"  Output mean: {ln_mean:+.4f}, var: {ln_var:.4f}")
    print("  LayerNorm normalizes across features (no batch dependency).")
    print("  Used in transformers where batch statistics across tokens are meaningless.")

    # === RESULTS AND COMPARISON TABLE ===
    print(f"\n{'=' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"\n  Architecture: {N_HIDDEN_LAYERS}-layer MLP, hidden_dim={HIDDEN_DIM}, "
          f"{N_CLASSES}-class classification")
    print(f"  Dataset: concentric rings, {n_total} samples (train={len(x_train)}, test={len(x_test)})")
    print(f"  Training: {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print()
    print(f"  {'Metric':<30} {'Without BN':>12} {'With BN':>12}")
    print(f"  {'─' * 54}")
    print(f"  {'Final train loss':<30} {history_no_bn[-1][0]:>12.4f} {history_bn[-1][0]:>12.4f}")
    print(f"  {'Final train accuracy':<30} {history_no_bn[-1][1]:>11.1%} {history_bn[-1][1]:>11.1%}")
    print(f"  {'Test accuracy':<30} {test_acc_no_bn:>11.1%} {test_acc_bn:>11.1%}")
    print(f"  {'Training time (s)':<30} {time_no_bn:>12.1f} {time_bn:>12.1f}")
    print(f"  {'Parameters':<30} {len(model_no_bn.parameters()):>12,} {len(model_bn.parameters()):>12,}")

    # Show convergence trajectory
    print(f"\n  Convergence trajectory (accuracy at epoch):")
    print(f"  {'Epoch':<8} {'Without BN':>12} {'With BN':>12}")
    print(f"  {'─' * 32}")
    for ep in [0, 2, 4, 7, 9, 12, min(14, NUM_EPOCHS - 1)]:
        if ep < len(history_no_bn):
            print(f"  {ep + 1:<8} {history_no_bn[ep][1]:>11.1%} {history_bn[ep][1]:>11.1%}")

    # Signpost: What production systems do differently
    print(f"\n{'─' * 65}")
    print("  Production notes:")
    print("  - BN is standard in CNNs (ResNet, EfficientNet) but NOT in transformers.")
    print("  - Transformers use LayerNorm because token positions have different roles;")
    print("    batch statistics across positions are semantically meaningless.")
    print("  - BN's dependence on batch size is a liability: small batches give noisy")
    print("    statistics, and batch size changes between training and inference.")
    print("  - GroupNorm and InstanceNorm are alternatives that avoid batch dependence.")
    print("  - Synchronized BN across GPUs is needed for distributed training to get")
    print("    consistent statistics across the full effective batch.")
    print(f"{'─' * 65}")
