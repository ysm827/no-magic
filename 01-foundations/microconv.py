"""
How a sliding kernel extracts spatial features — the convolution operation, pooling, and feature
maps that powered computer vision before transformers.
"""
# Reference: Convolutional Neural Networks (LeCun et al., 1998) applied to pattern recognition.
# This implementation builds a minimal CNN from scratch using scalar autograd, demonstrating how
# learned kernels become edge detectors and how pooling provides translation invariance.
# Architectural simplifications: single conv layer, 8x8 images, scalar (not tensor) autograd.

# === TRADEOFFS ===
# + Translation invariance: detects features regardless of position in the input
# + Parameter-efficient: shared kernels mean far fewer weights than fully connected layers
# + Hierarchical feature extraction: stacking layers captures increasing abstraction
# - Fixed receptive field per layer limits global context without deep stacking
# - Struggles with variable-length sequences without architectural changes
# - Pooling discards spatial precision (problematic for dense prediction tasks)
# WHEN TO USE: Image classification, object detection, or any grid-structured data
#   where local spatial patterns are the primary signal.
# WHEN NOT TO: Sequential data with long-range dependencies (use attention or RNNs),
#   or graph-structured data where adjacency is irregular.

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Image and dataset
IMAGE_SIZE = 8            # 8x8 binary images — small enough for scalar autograd
NUM_CLASSES = 4           # horizontal, vertical, diagonal, cross
TRAIN_SAMPLES = 200       # training examples per class
TEST_SAMPLES = 40         # test examples per class
NOISE_PROB = 0.05         # probability of flipping a pixel (adds variety)

# Convolution architecture
KERNEL_SIZE = 3           # 3x3 kernels — the standard for edge detection
NUM_KERNELS = 4           # one feature map per kernel — enough to learn 4 pattern types
PADDING = 0               # no padding: output shrinks by (kernel_size - 1)
STRIDE = 1                # slide kernel one pixel at a time
CONV_OUT = IMAGE_SIZE - KERNEL_SIZE + 1  # 8 - 3 + 1 = 6 (output spatial dimension after conv)
POOL_SIZE = 2             # 2x2 max pooling
POOL_OUT = CONV_OUT // POOL_SIZE         # 6 // 2 = 3 (output spatial dimension after pool)
FLAT_DIM = NUM_KERNELS * POOL_OUT * POOL_OUT  # 4 * 3 * 3 = 36 (flattened feature vector)

# Training
LEARNING_RATE = 0.005
BETA1 = 0.9
BETA2 = 0.999
EPS_ADAM = 1e-8
NUM_EPOCHS = 12
BATCH_SIZE = 16           # mini-batch gradient descent

# Signpost: Production CNNs operate on 224x224 images with 64+ kernels per layer and 100+
# layers. This toy scale (8x8, 4 kernels, 1 layer) makes scalar autograd feasible but would
# be absurdly slow at real image sizes — tensor libraries parallelize the entire conv operation.


# === SYNTHETIC DATASET ===

# Generate binary 8x8 images with four distinct pattern classes. Each "image" is a list of
# lists of floats (0.0 or 1.0). The patterns are chosen so that learned 3x3 kernels should
# converge to recognizable edge detectors — horizontal kernels, vertical kernels, etc.

def make_horizontal(noise: float = NOISE_PROB) -> list[list[float]]:
    """Generate an 8x8 image with 2-3 horizontal lines."""
    img = [[0.0] * IMAGE_SIZE for _ in range(IMAGE_SIZE)]
    num_lines = random.randint(2, 3)
    rows = random.sample(range(IMAGE_SIZE), num_lines)
    for r in rows:
        for c in range(IMAGE_SIZE):
            img[r][c] = 1.0
    # Add noise: randomly flip pixels to make patterns less trivial
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            if random.random() < noise:
                img[r][c] = 1.0 - img[r][c]
    return img


def make_vertical(noise: float = NOISE_PROB) -> list[list[float]]:
    """Generate an 8x8 image with 2-3 vertical lines."""
    img = [[0.0] * IMAGE_SIZE for _ in range(IMAGE_SIZE)]
    num_lines = random.randint(2, 3)
    cols = random.sample(range(IMAGE_SIZE), num_lines)
    for c in cols:
        for r in range(IMAGE_SIZE):
            img[r][c] = 1.0
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            if random.random() < noise:
                img[r][c] = 1.0 - img[r][c]
    return img


def make_diagonal(noise: float = NOISE_PROB) -> list[list[float]]:
    """Generate an 8x8 image with a diagonal line (main or anti-diagonal)."""
    img = [[0.0] * IMAGE_SIZE for _ in range(IMAGE_SIZE)]
    # Randomly choose main diagonal or anti-diagonal, with slight offset for variety
    offset = random.randint(-1, 1)
    use_anti = random.random() < 0.5
    for i in range(IMAGE_SIZE):
        if use_anti:
            j = IMAGE_SIZE - 1 - i + offset
        else:
            j = i + offset
        # Draw a 2-pixel-wide diagonal for visibility
        for dj in range(2):
            col = j + dj
            if 0 <= col < IMAGE_SIZE:
                img[i][col] = 1.0
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            if random.random() < noise:
                img[r][c] = 1.0 - img[r][c]
    return img


def make_cross(noise: float = NOISE_PROB) -> list[list[float]]:
    """Generate an 8x8 image with a cross (one horizontal + one vertical line)."""
    img = [[0.0] * IMAGE_SIZE for _ in range(IMAGE_SIZE)]
    # Place the cross at a random position
    center_r = random.randint(1, IMAGE_SIZE - 2)
    center_c = random.randint(1, IMAGE_SIZE - 2)
    for c in range(IMAGE_SIZE):
        img[center_r][c] = 1.0
    for r in range(IMAGE_SIZE):
        img[r][center_c] = 1.0
    for r in range(IMAGE_SIZE):
        for c in range(IMAGE_SIZE):
            if random.random() < noise:
                img[r][c] = 1.0 - img[r][c]
    return img


GENERATORS = [make_horizontal, make_vertical, make_diagonal, make_cross]
CLASS_NAMES = ["horizontal", "vertical", "diagonal", "cross"]


def generate_dataset(
    samples_per_class: int,
) -> tuple[list[list[list[float]]], list[int]]:
    """Generate a balanced dataset of synthetic images and labels."""
    images: list[list[list[float]]] = []
    labels: list[int] = []
    for class_id, gen_fn in enumerate(GENERATORS):
        for _ in range(samples_per_class):
            images.append(gen_fn())
            labels.append(class_id)
    # Shuffle while keeping image-label pairs aligned
    combined = list(zip(images, labels))
    random.shuffle(combined)
    imgs, lbls = zip(*combined)
    return list(imgs), list(lbls)


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
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent: float) -> Value:
        # d(x^n)/dx = n * x^(n-1)
        return Value(
            self.data ** exponent, (self,),
            (exponent * self.data ** (exponent - 1),)
        )

    def __neg__(self) -> Value:
        return self * -1.0

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return Value(other) + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other: float) -> Value:
        return Value(other) * (self ** -1.0)

    def relu(self) -> Value:
        # d(relu(x))/dx = 1 if x > 0 else 0
        return Value(max(0.0, self.data), (self,), (float(self.data > 0),))

    def exp(self) -> Value:
        # d(e^x)/dx = e^x
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        # d(log(x))/dx = 1/x
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def max_with(self, other: Value) -> Value:
        """Differentiable max of two Values — gradient flows to the winner only.

        This is the core operation for max pooling. Unlike relu (max with 0), this
        compares two tracked values and routes the gradient only to whichever was larger.
        The losing value gets zero gradient, which means pooling implicitly selects
        which spatial location matters most for the downstream classification.
        """
        if self.data >= other.data:
            return Value(self.data, (self, other), (1.0, 0.0))
        return Value(other.data, (self, other), (0.0, 1.0))

    def backward(self) -> None:
        """Compute gradients via reverse-mode automatic differentiation.

        Builds a topological ordering of the computation graph, then propagates
        gradients backward using the chain rule: dL/dx = sum(dL/dy * dy/dx)
        for each output y that depends on x.
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

        build_topo(self)
        self.grad = 1.0

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# === CONVOLUTION OPERATIONS ===

# The fundamental insight of convolution: instead of connecting every input pixel to every
# output neuron (fully connected), we slide a small kernel across the image. This enforces
# two priors: (1) locality — features are detected from nearby pixels, and (2) weight sharing
# — the same detector runs at every spatial location. These priors dramatically reduce
# parameters and encode the assumption that visual features can appear anywhere in the image.

def conv2d(
    image: list[list[Value]],
    kernel: list[list[Value]],
    bias: Value,
) -> list[list[Value]]:
    """Apply a single 3x3 convolution kernel to an image.

    Math: out[i,j] = sum_m sum_n kernel[m,n] * image[i+m, j+n] + bias
    where m,n range over the kernel dimensions (0..2 for a 3x3 kernel).

    The kernel slides across the image with stride=1, no padding. Output spatial
    dimensions shrink by (kernel_size - 1) in each direction: an 8x8 image with
    a 3x3 kernel produces a 6x6 feature map.

    Args:
        image: [H, W] grid of Value objects
        kernel: [K, K] grid of learned Value weights
        bias: scalar Value bias added to each output position

    Returns:
        [H-K+1, W-K+1] grid of Value objects (the "feature map")
    """
    h = len(image)
    w = len(image[0])
    k = len(kernel)
    out_h = h - k + 1
    out_w = w - k + 1
    output: list[list[Value]] = []

    for i in range(out_h):
        row: list[Value] = []
        for j in range(out_w):
            # Dot product between kernel and the image patch at position (i, j)
            # This is the "sliding window" operation that gives CNNs their name.
            val = bias
            for m in range(k):
                for n in range(k):
                    val = val + kernel[m][n] * image[i + m][j + n]
            row.append(val)
        output.append(row)
    return output


def max_pool2d(feature_map: list[list[Value]], pool_size: int = 2) -> list[list[Value]]:
    """Downsample a feature map by taking the maximum in each pool_size x pool_size window.

    Max pooling serves two purposes:
    1. Dimensionality reduction — halves spatial dimensions (4x fewer values)
    2. Translation invariance — if a feature shifts by 1 pixel within the pool window,
       the output is unchanged. This makes the model robust to small spatial shifts.

    Math: out[i,j] = max(feature_map[2i:2i+2, 2j:2j+2])

    Signpost: Production CNNs often use stride-2 convolutions instead of pooling.
    Max pooling is pedagogically clearer and historically foundational (LeNet, AlexNet).
    """
    h = len(feature_map)
    w = len(feature_map[0])
    out_h = h // pool_size
    out_w = w // pool_size
    output: list[list[Value]] = []

    for i in range(out_h):
        row: list[Value] = []
        for j in range(out_w):
            # Find the maximum value in the pool_size x pool_size window
            # Using Value.max_with to keep the operation differentiable
            current_max = feature_map[i * pool_size][j * pool_size]
            for m in range(pool_size):
                for n in range(pool_size):
                    if m == 0 and n == 0:
                        continue
                    current_max = current_max.max_with(
                        feature_map[i * pool_size + m][j * pool_size + n]
                    )
            row.append(current_max)
        output.append(row)
    return output


# === MODEL DEFINITION ===

# Architecture: Input(8x8) → Conv(4 kernels, 3x3) → ReLU → MaxPool(2x2) → Flatten → Linear → Softmax
#
# This is a minimal CNN classifier. One conv layer with 4 kernels extracts spatial features
# (edges, lines), ReLU introduces nonlinearity, max pooling reduces spatial dimensions and
# adds translation invariance, and a linear layer maps flattened features to class logits.
#
# Signpost: Real CNNs (VGG, ResNet) stack many conv layers with increasing kernel counts
# (64 → 128 → 256 → 512). Each layer captures increasingly abstract features: edges →
# textures → parts → objects. Our single layer can only capture low-level edge patterns,
# which is enough for the synthetic line detection task.

def init_parameters() -> dict:
    """Initialize all model parameters: conv kernels, biases, and linear layer.

    Returns a dict of named parameters. Kernels are initialized with small Gaussian noise
    (Kaiming-style: std = sqrt(2 / fan_in) where fan_in = kernel_size^2 = 9).
    """
    params: dict = {}

    # Kaiming initialization: std = sqrt(2 / fan_in) helps ReLU networks maintain
    # activation variance across layers. With fan_in = 9 (3x3 kernel), std ≈ 0.47.
    kernel_std = math.sqrt(2.0 / (KERNEL_SIZE * KERNEL_SIZE))

    # Convolution kernels: NUM_KERNELS independent 3x3 filters
    # Each kernel learns to detect a different spatial pattern (horizontal edge,
    # vertical edge, diagonal, etc.)
    for k in range(NUM_KERNELS):
        params[f'conv_kernel_{k}'] = [
            [Value(random.gauss(0, kernel_std)) for _ in range(KERNEL_SIZE)]
            for _ in range(KERNEL_SIZE)
        ]
        params[f'conv_bias_{k}'] = Value(0.0)

    # Linear classifier: maps flattened feature vector to class logits
    # Input: FLAT_DIM = NUM_KERNELS * POOL_OUT * POOL_OUT = 4 * 3 * 3 = 36
    # Output: NUM_CLASSES = 4
    linear_std = math.sqrt(2.0 / FLAT_DIM)
    params['linear_w'] = [
        [Value(random.gauss(0, linear_std)) for _ in range(FLAT_DIM)]
        for _ in range(NUM_CLASSES)
    ]
    params['linear_b'] = [Value(0.0) for _ in range(NUM_CLASSES)]

    return params


def flatten(feature_maps: list[list[list[Value]]]) -> list[Value]:
    """Flatten multiple 2D feature maps into a single 1D vector.

    Converts [NUM_KERNELS, POOL_OUT, POOL_OUT] → [NUM_KERNELS * POOL_OUT * POOL_OUT].
    This bridges the spatial conv layers and the fully-connected classifier.
    """
    flat: list[Value] = []
    for fmap in feature_maps:
        for row in fmap:
            flat.extend(row)
    return flat


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: converts logits to probabilities.

    Subtracting max(logits) before exp() prevents overflow. Without this trick,
    logits > 700 would cause exp() to return inf.

    Math: softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def safe_log(prob: Value) -> Value:
    """Clamped logarithm for numerical stability in cross-entropy loss.

    Prevents log(0) = -inf by clamping to 1e-10. The gradient (1/x) is evaluated
    at the clamped value, but prob remains the child node so gradients flow backward
    through the computation graph.
    """
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


def forward(image_data: list[list[float]], params: dict) -> list[Value]:
    """Full CNN forward pass: Conv → ReLU → Pool → Flatten → Linear.

    Args:
        image_data: raw 8x8 image as floats (not yet wrapped in Value)
        params: model parameters dict

    Returns:
        logits: list of NUM_CLASSES Values (unnormalized log-probabilities)
    """
    # Wrap raw image pixels as Value objects so operations build the autograd graph
    image = [[Value(pixel) for pixel in row] for row in image_data]

    # Apply each conv kernel independently, producing NUM_KERNELS feature maps
    feature_maps: list[list[list[Value]]] = []
    for k in range(NUM_KERNELS):
        # Conv2d: slide kernel across image, producing a 6x6 feature map
        fmap = conv2d(image, params[f'conv_kernel_{k}'], params[f'conv_bias_{k}'])

        # ReLU activation: introduces nonlinearity. Without this, stacking conv layers
        # would be equivalent to a single linear transformation (composition of linear
        # functions is linear). ReLU lets the network learn nonlinear decision boundaries.
        fmap = [[val.relu() for val in row] for row in fmap]

        # Max pooling: 6x6 → 3x3 (reduces spatial dimensions by factor of 2)
        fmap = max_pool2d(fmap, POOL_SIZE)

        feature_maps.append(fmap)

    # Flatten: [4, 3, 3] → [36]
    flat = flatten(feature_maps)

    # Linear classifier: logit_c = sum_j W[c,j] * flat[j] + bias[c]
    logits: list[Value] = []
    for c in range(NUM_CLASSES):
        val = params['linear_b'][c]
        for j in range(FLAT_DIM):
            val = val + params['linear_w'][c][j] * flat[j]
        logits.append(val)

    return logits


def compute_loss(logits: list[Value], target: int) -> Value:
    """Cross-entropy loss for a single example.

    Math: L = -log(softmax(logits)[target])
    This measures how surprised the model is by the correct class. A perfect model
    assigns probability 1.0 to the target, giving loss = -log(1) = 0.
    """
    probs = softmax(logits)
    return -safe_log(probs[target])


# === TRAINING LOOP ===

if __name__ == "__main__":
    start_time = time.time()

    # Generate training and test datasets
    print("Generating synthetic dataset...")
    train_images, train_labels = generate_dataset(TRAIN_SAMPLES)
    test_images, test_labels = generate_dataset(TEST_SAMPLES)
    print(f"  Training: {len(train_images)} images ({TRAIN_SAMPLES} per class)")
    print(f"  Test:     {len(test_images)} images ({TEST_SAMPLES} per class)")

    # Initialize model parameters
    params = init_parameters()

    # Collect all trainable parameters into a flat list for the optimizer
    param_list: list[Value] = []
    for key in params:
        val = params[key]
        if isinstance(val, Value):
            param_list.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, Value):
                    param_list.append(item)
                elif isinstance(item, list):
                    for v in item:
                        if isinstance(v, Value):
                            param_list.append(v)

    print(f"  Parameters: {len(param_list)}")
    print(f"  Architecture: Conv({NUM_KERNELS}x{KERNEL_SIZE}x{KERNEL_SIZE}) → ReLU → "
          f"MaxPool({POOL_SIZE}x{POOL_SIZE}) → Linear({FLAT_DIM}→{NUM_CLASSES})\n")

    # Adam optimizer state (per-parameter momentum and variance)
    m_state = [0.0] * len(param_list)
    v_state = [0.0] * len(param_list)
    adam_t = 0  # global step counter for bias correction

    # Training loop
    print("Training...")
    num_batches = len(train_images) // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        # Shuffle training data each epoch
        combined = list(zip(train_images, train_labels))
        random.shuffle(combined)
        shuffled_images = [c[0] for c in combined]
        shuffled_labels = [c[1] for c in combined]

        epoch_loss = 0.0
        epoch_correct = 0

        for batch_start in range(0, len(shuffled_images) - BATCH_SIZE + 1, BATCH_SIZE):
            adam_t += 1
            batch_loss_val = 0.0

            # Accumulate gradients over the mini-batch before updating weights.
            # This averages out noise in individual gradients and stabilizes training.
            for i in range(BATCH_SIZE):
                idx = batch_start + i
                img = shuffled_images[idx]
                label = shuffled_labels[idx]

                logits = forward(img, params)
                loss = compute_loss(logits, label)
                loss.backward()

                batch_loss_val += loss.data

                # Track accuracy
                predicted = max(range(NUM_CLASSES), key=lambda c: logits[c].data)
                if predicted == label:
                    epoch_correct += 1

            epoch_loss += batch_loss_val

            # Adam optimizer update
            lr_t = LEARNING_RATE * (1.0 - epoch / NUM_EPOCHS)  # linear LR decay

            for i, p in enumerate(param_list):
                # Average gradient over the batch
                g = p.grad / BATCH_SIZE

                m_state[i] = BETA1 * m_state[i] + (1.0 - BETA1) * g
                v_state[i] = BETA2 * v_state[i] + (1.0 - BETA2) * g * g

                # Bias correction: early updates would be biased toward zero without this
                m_hat = m_state[i] / (1.0 - BETA1 ** adam_t)
                v_hat = v_state[i] / (1.0 - BETA2 ** adam_t)

                p.data -= lr_t * m_hat / (math.sqrt(v_hat) + EPS_ADAM)
                p.grad = 0.0

        avg_loss = epoch_loss / len(shuffled_images)
        accuracy = epoch_correct / len(shuffled_images) * 100
        elapsed = time.time() - start_time
        print(f"  epoch {epoch + 1:>2}/{NUM_EPOCHS} | loss: {avg_loss:.4f} | "
              f"train acc: {accuracy:.1f}% | time: {elapsed:.1f}s")

    print(f"\nTraining complete ({time.time() - start_time:.1f}s)\n")

    # === KERNEL VISUALIZATION ===

    # Display the learned 3x3 kernels as ASCII art. After training, kernels should visually
    # resemble edge detectors: horizontal kernels show strong horizontal gradients, vertical
    # kernels show vertical gradients, etc. The kernel weights determine what spatial pattern
    # each feature map responds to — this is the "learned representation."

    print("Learned kernels (ASCII visualization):")
    print("  Darker = more negative, lighter = more positive")
    print("  Effective kernels resemble edge/line detectors\n")

    # Map kernel weights to ASCII brightness characters
    ascii_chars = " .:-=+*#%@"  # dark to bright

    for k in range(NUM_KERNELS):
        kernel = params[f'conv_kernel_{k}']
        # Find weight range for normalization
        weights = [kernel[m][n].data for m in range(KERNEL_SIZE) for n in range(KERNEL_SIZE)]
        w_min = min(weights)
        w_max = max(weights)
        w_range = w_max - w_min if w_max > w_min else 1.0

        print(f"  Kernel {k} (bias={params[f'conv_bias_{k}'].data:+.3f}):")
        for m in range(KERNEL_SIZE):
            row_str = "    "
            for n in range(KERNEL_SIZE):
                # Normalize weight to [0, 1] then map to ASCII character
                normalized = (kernel[m][n].data - w_min) / w_range
                char_idx = min(int(normalized * (len(ascii_chars) - 1)), len(ascii_chars) - 1)
                # Use 2 chars per cell for square aspect ratio in terminal
                row_str += ascii_chars[char_idx] * 3 + " "
            # Show raw weight values alongside ASCII
            raw = " ".join(f"{kernel[m][n].data:+.2f}" for n in range(KERNEL_SIZE))
            row_str += f"  [{raw}]"
            print(row_str)
        print()

    # === INFERENCE AND RESULTS ===

    print("Evaluating on test set...\n")

    correct = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    for img, label in zip(test_images, test_labels):
        logits = forward(img, params)
        predicted = max(range(NUM_CLASSES), key=lambda c: logits[c].data)
        if predicted == label:
            correct += 1
            class_correct[label] += 1
        class_total[label] += 1
        confusion[label][predicted] += 1

    total = len(test_images)
    print(f"Test accuracy: {correct}/{total} ({correct / total * 100:.1f}%)\n")

    print("Per-class accuracy:")
    for c in range(NUM_CLASSES):
        acc = class_correct[c] / class_total[c] * 100 if class_total[c] > 0 else 0
        print(f"  {CLASS_NAMES[c]:>12}: {class_correct[c]}/{class_total[c]} ({acc:.1f}%)")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "             " + "".join(f"{CLASS_NAMES[c]:>12}" for c in range(NUM_CLASSES))
    print(header)
    for r in range(NUM_CLASSES):
        row_str = f"  {CLASS_NAMES[r]:>10}"
        for c in range(NUM_CLASSES):
            row_str += f"{confusion[r][c]:>12}"
        print(row_str)

    # Show a few example classifications
    print("\nSample predictions:")
    for i in range(min(8, len(test_images))):
        img = test_images[i]
        label = test_labels[i]
        logits = forward(img, params)
        probs = softmax(logits)
        predicted = max(range(NUM_CLASSES), key=lambda c: logits[c].data)
        prob_str = ", ".join(f"{CLASS_NAMES[c]}:{probs[c].data:.2f}" for c in range(NUM_CLASSES))
        status = "OK" if predicted == label else "WRONG"
        print(f"  [{status:>5}] true={CLASS_NAMES[label]:>10}, "
              f"pred={CLASS_NAMES[predicted]:>10} | {prob_str}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f}s")
