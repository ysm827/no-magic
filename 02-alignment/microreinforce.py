"""
The simplest policy gradient algorithm -- how log-probability weighting turns reward signals
into gradient updates, and why variance reduction matters.
"""
# Reference: Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist
# Reinforcement Learning" (1992). https://link.springer.com/article/10.1007/BF00992696
# REINFORCE is the foundation of all modern policy gradient methods (PPO, GRPO, RLHF).
# This script implements it from scratch on a synthetic sequence generation task, then
# demonstrates why baseline subtraction is essential for practical use.

# === TRADEOFFS ===
# + Minimal assumptions: works with any differentiable policy and any reward signal
# + Unbiased gradient estimates (the expected gradient equals the true policy gradient)
# + Foundation for all modern policy gradient methods (PPO, GRPO, RLHF)
# - Extremely high variance without baseline subtraction (slow convergence)
# - No credit assignment: entire trajectory gets the same reward signal
# - Sample-inefficient: each trajectory is used once then discarded (on-policy)
# WHEN TO USE: Simple RL problems, prototyping reward functions, or as a baseline
#   before trying more complex algorithms (PPO, GRPO).
# WHEN NOT TO: Production RL systems (use PPO for stability), long-horizon tasks
#   (variance makes learning impractical), or when sample efficiency matters.

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Vocabulary: lowercase letters. Simple enough to train quickly, rich enough to see
# the policy learn non-trivial structure (vowel placement, character repetition).
VOCAB = list("abcdefghijklmnopqrstuvwxyz")
VOCAB_SIZE = len(VOCAB)

# Policy network architecture
HIDDEN_DIM = 32      # single hidden layer width
CONTEXT_DIM = VOCAB_SIZE + 1  # input: one-hot current char + normalized position

# Sequence generation
MAX_SEQ_LEN = 8      # maximum characters per generated sequence
MIN_SEQ_LEN = 2      # minimum before applying reward (avoids degenerate empty outputs)

# Training parameters -- raw REINFORCE
RAW_EPISODES = 300       # episodes for raw REINFORCE
RAW_LR = 0.01            # learning rate
BATCH_SIZE = 8            # sequences per gradient update

# Training parameters -- REINFORCE with baseline
BASELINE_EPISODES = 300   # episodes for baseline version
BASELINE_LR = 0.01

# Adam optimizer
BETA1 = 0.9
BETA2 = 0.999
EPS_ADAM = 1e-8

# Variance tracking: sample gradient norms every N episodes to compare methods
VARIANCE_SAMPLE_INTERVAL = 10


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
# This Value class follows the canonical interface (see docs/autograd-interface.md).
# No extensions needed: REINFORCE only requires log, exp, tanh, and basic arithmetic.


# === REWARD FUNCTION ===
# Deterministic, interpretable rules on generated sequences. The policy must discover
# these rules through trial-and-error -- it never sees the reward function directly,
# only the scalar reward signal after generating a complete sequence.
#
# Signpost: This is a "bandit" setting where reward depends on the full sequence,
# not on individual actions. Credit assignment (which character caused the reward?)
# is the core challenge REINFORCE addresses via log-probability weighting.

VOWELS = set("aeiou")

def compute_reward(sequence: list[int]) -> float:
    """Score a generated character sequence with interpretable, deterministic rules.

    Rules (cumulative):
      +1.0  if the sequence starts with a vowel
      +1.0  if the sequence ends with a consonant
      +0.5  for each distinct vowel present (max +2.5 for all 5)
      -1.0  if any character repeats 3+ times consecutively
      +1.0  if length is 4-6 characters (the "sweet spot")
      -2.0  if length < MIN_SEQ_LEN

    These rules reward name-like sequences: vowel-initial, consonant-final, moderate
    length, varied characters. The maximum achievable reward is ~5.5.
    """
    if len(sequence) < MIN_SEQ_LEN:
        return -2.0

    chars = [VOCAB[idx] for idx in sequence]
    reward = 0.0

    # Rule 1: Starts with a vowel
    if chars[0] in VOWELS:
        reward += 1.0

    # Rule 2: Ends with a consonant
    if chars[-1] not in VOWELS:
        reward += 1.0

    # Rule 3: Reward vowel diversity (encourages varied, pronounceable sequences)
    distinct_vowels = len(set(ch for ch in chars if ch in VOWELS))
    reward += 0.5 * distinct_vowels

    # Rule 4: Penalize stuttering (three consecutive identical characters)
    for i in range(len(chars) - 2):
        if chars[i] == chars[i + 1] == chars[i + 2]:
            reward -= 1.0
            break  # One penalty is enough signal

    # Rule 5: Length bonus for the sweet spot
    if 4 <= len(sequence) <= 6:
        reward += 1.0

    return reward


# === POLICY NETWORK ===
# A simple MLP that takes a context vector (current character + position) and outputs
# a probability distribution over the next character. This is the "actor" in actor-critic
# terminology, though REINFORCE uses Monte Carlo returns rather than a learned critic.
#
# Architecture: context -> linear -> tanh -> linear -> softmax -> sample
# Two layers suffice because the reward rules are simple. The key insight is that the
# network must output *probabilities* (via softmax), not actions directly -- REINFORCE
# works by adjusting these probabilities via the log-probability gradient.

def init_policy(input_dim: int, hidden_dim: int, output_dim: int) -> dict[str, list[list[Value]]]:
    """Initialize policy network weights with small random values."""
    params: dict[str, list[list[Value]]] = {}
    # Xavier-like scaling: std = 1/sqrt(fan_in) prevents activation explosion/vanishing
    std1 = 1.0 / math.sqrt(input_dim)
    std2 = 1.0 / math.sqrt(hidden_dim)
    params['w1'] = [[Value(random.gauss(0, std1)) for _ in range(input_dim)] for _ in range(hidden_dim)]
    params['b1'] = [[Value(0.0)] for _ in range(hidden_dim)]
    params['w2'] = [[Value(random.gauss(0, std2)) for _ in range(hidden_dim)] for _ in range(output_dim)]
    params['b2'] = [[Value(0.0)] for _ in range(output_dim)]
    return params


def flatten_params(params: dict[str, list[list[Value]]]) -> list[Value]:
    """Collect all Value objects from a parameter dict into a flat list."""
    return [p for matrix in params.values() for row in matrix for p in row]


def make_context(char_idx: int, position: int) -> list[Value]:
    """Build the input vector for the policy network.

    Encodes the current character as a one-hot vector and appends normalized position.
    The position signal lets the policy learn position-dependent behavior (e.g., prefer
    vowels at the start, consonants at the end).
    """
    context = [Value(0.0)] * CONTEXT_DIM
    if 0 <= char_idx < VOCAB_SIZE:
        context[char_idx] = Value(1.0)
    # Normalized position: 0.0 at start, 1.0 at end of max sequence
    context[VOCAB_SIZE] = Value(position / MAX_SEQ_LEN)
    return context


def policy_forward(context: list[Value], params: dict[str, list[list[Value]]]) -> list[Value]:
    """Forward pass: context -> hidden (tanh) -> logits -> softmax probabilities.

    Returns a probability distribution over VOCAB_SIZE actions (characters). The softmax
    ensures probabilities sum to 1 and are differentiable, which is critical for the
    log-probability gradient in REINFORCE.
    """
    # Hidden layer: h = tanh(W1 @ x + b1)
    hidden: list[Value] = []
    for i in range(len(params['w1'])):
        h = sum(params['w1'][i][j] * context[j] for j in range(len(context))) + params['b1'][i][0]
        hidden.append(h.tanh())

    # Output layer: logits = W2 @ h + b2
    logits: list[Value] = []
    for i in range(len(params['w2'])):
        logit = sum(params['w2'][i][j] * hidden[j] for j in range(len(hidden))) + params['b2'][i][0]
        logits.append(logit)

    # Numerically stable softmax: subtract max before exp to prevent overflow
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    probs = [e / total for e in exp_vals]

    return probs


def safe_log(prob: Value) -> Value:
    """Clipped log for numerical stability. Prevents log(0) = -inf.

    Builds the node manually with prob as its child so gradients flow through the
    computation graph. Without this, a single zero probability early in training
    would produce -inf loss and NaN gradients, killing the entire training run.
    """
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === TRAJECTORY GENERATION ===

def generate_trajectory(
    params: dict[str, list[list[Value]]],
    build_graph: bool = True,
) -> tuple[list[int], list[Value], float]:
    """Generate a sequence by sampling from the policy, optionally building the autograd graph.

    Returns:
        actions: list of character indices chosen at each step
        log_probs: log π(a_t | s_t) for each action (Value objects if build_graph=True)
        reward: scalar reward for the complete sequence

    The policy generates characters autoregressively: each step conditions on the previous
    character and the current position. A special "stop" signal is not modeled explicitly;
    instead, generation runs for a random length (sampled from the policy's implicit length
    distribution via early stopping when we detect low entropy, or up to MAX_SEQ_LEN).

    Signpost: Production RL systems use an explicit stop token. We use fixed-length
    generation with MAX_SEQ_LEN for simplicity -- the reward function's length bonus
    shapes the effective length distribution.
    """
    actions: list[int] = []
    log_probs: list[Value] = []

    # Start with a "null" character context (position 0, no previous character)
    char_idx = -1  # No previous character at the start

    for pos in range(MAX_SEQ_LEN):
        context = make_context(char_idx, pos)
        probs = policy_forward(context, params)

        # Sample action from the probability distribution
        weights = [p.data for p in probs]
        action = random.choices(range(VOCAB_SIZE), weights=weights)[0]

        # Store log π(a_t | s_t) -- the core quantity REINFORCE needs
        # This log-probability connects the action to the policy parameters:
        # ∇ log π(a|s) tells us how to adjust parameters to make action `a` more
        # or less likely in state `s`.
        if build_graph:
            log_probs.append(safe_log(probs[action]))
        else:
            log_probs.append(Value(math.log(max(weights[action], 1e-10))))

        actions.append(action)
        char_idx = action

    reward = compute_reward(actions)
    return actions, log_probs, reward


# === REINFORCE ALGORITHM ===
#
# The policy gradient theorem (Williams, 1992):
#   ∇J(θ) = E_τ [ Σ_t R(τ) · ∇ log π_θ(a_t | s_t) ]
#
# In words: to increase expected reward, adjust parameters in the direction that makes
# high-reward actions more probable and low-reward actions less probable. The magnitude
# of the adjustment is proportional to the reward.
#
# Implementation as a loss function for autograd:
#   L = -Σ_t R(τ) · log π_θ(a_t | s_t)
#
# The negative sign is because we minimize L but want to maximize J. When we call
# L.backward(), the gradient ∂L/∂θ = -R · ∂log π/∂θ, so gradient descent on L
# is equivalent to gradient ascent on J.
#
# Key insight: R(τ) is a SCALAR that weights the entire trajectory's log-probabilities.
# Good trajectories (high R) get strongly reinforced; bad ones get weakly reinforced
# or discouraged (if R < 0). This is credit assignment through log-probability weighting.

def compute_reinforce_loss(log_probs: list[Value], reward: float) -> Value:
    """REINFORCE loss: L = -R · Σ_t log π(a_t | s_t).

    The reward R weights all log-probabilities equally because in our setting, R depends
    on the entire sequence, not individual characters. With per-step rewards, we would
    use discounted future returns instead of the full trajectory reward.

    Math: ∇_θ L = -R · Σ_t ∇_θ log π_θ(a_t | s_t)
    This IS the policy gradient -- autograd computes it for us.
    """
    total_log_prob = sum(log_probs)
    return -(reward * total_log_prob)


def compute_reinforce_loss_with_baseline(
    log_probs: list[Value],
    reward: float,
    baseline: float,
) -> Value:
    """REINFORCE with baseline: L = -(R - b) · Σ_t log π(a_t | s_t).

    Subtracting a baseline b from the reward does NOT change the expected gradient:
      E[R · ∇log π] = E[(R-b) · ∇log π] + b · E[∇log π]
    and E[∇log π] = 0 (the score function identity).

    But it DOES reduce variance. Intuition: without a baseline, even a "good" reward
    of R=3 reinforces all actions in the trajectory. With baseline b=2.5, only the
    excess (R-b=0.5) matters -- a much smaller, more informative signal. The baseline
    acts as a "how much better than average was this trajectory?" measure.

    The optimal baseline minimizes Var[R · ∇log π], but even a simple running average
    of rewards (what we use) gives dramatic variance reduction.
    """
    advantage = reward - baseline
    total_log_prob = sum(log_probs)
    return -(advantage * total_log_prob)


# === ADAM OPTIMIZER ===

def adam_update(
    param_list: list[Value],
    m: list[float],
    v: list[float],
    step: int,
    lr: float,
) -> None:
    """One step of Adam optimizer with gradient clipping.

    Adam adapts the learning rate per-parameter based on gradient history. The first
    moment (m) provides momentum; the second moment (v) provides per-parameter scaling.
    Bias correction compensates for the zero-initialization of m and v.
    """
    for i, p in enumerate(param_list):
        # Clip gradients to prevent explosion from high-variance REINFORCE updates
        grad = max(-1.0, min(1.0, p.grad))
        m[i] = BETA1 * m[i] + (1 - BETA1) * grad
        v[i] = BETA2 * v[i] + (1 - BETA2) * grad ** 2
        m_hat = m[i] / (1 - BETA1 ** (step + 1))
        v_hat = v[i] / (1 - BETA2 ** (step + 1))
        p.data -= lr * m_hat / (v_hat ** 0.5 + EPS_ADAM)
        p.grad = 0.0


# === TRAINING ===

if __name__ == "__main__":
    start_time = time.time()

    print("=== REINFORCE: Policy Gradient From Scratch ===\n")
    print(f"Vocabulary: {VOCAB_SIZE} characters (a-z)")
    print(f"Sequence length: up to {MAX_SEQ_LEN} characters")
    print(f"Reward rules: vowel start (+1), consonant end (+1), vowel diversity (+0.5 each),")
    print(f"  no triple repeats (-1), length 4-6 bonus (+1), max achievable ~5.5\n")

    # === Phase 1: Raw REINFORCE (high variance) ===
    print("=== Phase 1: Raw REINFORCE (no baseline) ===")

    policy_raw = init_policy(CONTEXT_DIM, HIDDEN_DIM, VOCAB_SIZE)
    param_list_raw = flatten_params(policy_raw)
    print(f"Policy parameters: {len(param_list_raw)}")

    m_raw = [0.0] * len(param_list_raw)
    v_raw = [0.0] * len(param_list_raw)

    raw_rewards: list[float] = []
    raw_grad_norms: list[float] = []

    for episode in range(RAW_EPISODES):
        # Generate a batch of trajectories and accumulate gradients
        batch_loss = Value(0.0)
        batch_reward = 0.0

        for _ in range(BATCH_SIZE):
            actions, log_probs, reward = generate_trajectory(policy_raw, build_graph=True)
            loss = compute_reinforce_loss(log_probs, reward)
            batch_loss = batch_loss + loss
            batch_reward += reward

        # Average over batch (reduces variance by a factor of BATCH_SIZE)
        batch_loss = batch_loss * (1.0 / BATCH_SIZE)
        avg_reward = batch_reward / BATCH_SIZE
        raw_rewards.append(avg_reward)

        batch_loss.backward()

        # Track gradient norm for variance comparison
        if (episode + 1) % VARIANCE_SAMPLE_INTERVAL == 0:
            grad_norm = math.sqrt(sum(p.grad ** 2 for p in param_list_raw))
            raw_grad_norms.append(grad_norm)

        adam_update(param_list_raw, m_raw, v_raw, episode, RAW_LR)

        if (episode + 1) % 50 == 0 or episode == 0:
            recent_avg = sum(raw_rewards[-50:]) / len(raw_rewards[-50:])
            print(f"  episode {episode + 1:>4}/{RAW_EPISODES} | "
                  f"batch_reward: {avg_reward:.2f} | recent_avg: {recent_avg:.2f}")

    # === Phase 2: REINFORCE with baseline (variance-reduced) ===
    print("\n=== Phase 2: REINFORCE with Baseline ===")

    # Fresh policy with same initialization seed for fair comparison
    random.seed(42)
    policy_bl = init_policy(CONTEXT_DIM, HIDDEN_DIM, VOCAB_SIZE)
    param_list_bl = flatten_params(policy_bl)

    m_bl = [0.0] * len(param_list_bl)
    v_bl = [0.0] * len(param_list_bl)

    bl_rewards: list[float] = []
    bl_grad_norms: list[float] = []
    baseline_value = 0.0  # Running average of rewards (exponential moving average)
    baseline_alpha = 0.1  # EMA decay rate -- higher = adapts faster, noisier

    for episode in range(BASELINE_EPISODES):
        batch_loss = Value(0.0)
        batch_reward = 0.0

        for _ in range(BATCH_SIZE):
            actions, log_probs, reward = generate_trajectory(policy_bl, build_graph=True)
            # The only difference from raw REINFORCE: subtract the baseline
            loss = compute_reinforce_loss_with_baseline(log_probs, reward, baseline_value)
            batch_loss = batch_loss + loss
            batch_reward += reward

        batch_loss = batch_loss * (1.0 / BATCH_SIZE)
        avg_reward = batch_reward / BATCH_SIZE
        bl_rewards.append(avg_reward)

        # Update baseline: exponential moving average of observed rewards.
        # This tracks the "expected reward" under the current policy. As the policy
        # improves, the baseline rises, so only trajectories that are *better than
        # the current average* produce positive advantage signals.
        baseline_value = baseline_alpha * avg_reward + (1 - baseline_alpha) * baseline_value

        batch_loss.backward()

        # Track gradient norm for variance comparison
        if (episode + 1) % VARIANCE_SAMPLE_INTERVAL == 0:
            grad_norm = math.sqrt(sum(p.grad ** 2 for p in param_list_bl))
            bl_grad_norms.append(grad_norm)

        adam_update(param_list_bl, m_bl, v_bl, episode, BASELINE_LR)

        if (episode + 1) % 50 == 0 or episode == 0:
            recent_avg = sum(bl_rewards[-50:]) / len(bl_rewards[-50:])
            print(f"  episode {episode + 1:>4}/{BASELINE_EPISODES} | "
                  f"batch_reward: {avg_reward:.2f} | recent_avg: {recent_avg:.2f} | "
                  f"baseline: {baseline_value:.2f}")

    # === VARIANCE COMPARISON ===
    print("\n=== Variance Comparison ===")

    # Compare gradient norm statistics between the two methods.
    # Lower variance = more stable training = faster convergence.
    # The baseline should produce noticeably smaller gradient norms because
    # the advantage (R - b) is centered near zero, unlike the raw reward R
    # which is always positive when the policy is decent.

    if raw_grad_norms and bl_grad_norms:
        raw_mean = sum(raw_grad_norms) / len(raw_grad_norms)
        raw_var = sum((g - raw_mean) ** 2 for g in raw_grad_norms) / len(raw_grad_norms)

        bl_mean = sum(bl_grad_norms) / len(bl_grad_norms)
        bl_var = sum((g - bl_mean) ** 2 for g in bl_grad_norms) / len(bl_grad_norms)

        print(f"Gradient norm statistics (sampled every {VARIANCE_SAMPLE_INTERVAL} episodes):")
        print(f"  Raw REINFORCE  -- mean: {raw_mean:.4f}, variance: {raw_var:.4f}")
        print(f"  With baseline  -- mean: {bl_mean:.4f}, variance: {bl_var:.4f}")

        if raw_var > 0:
            reduction = (1 - bl_var / raw_var) * 100
            print(f"  Variance reduction: {reduction:.1f}%")
        # Signpost: In theory, the optimal baseline (minimizing variance of the gradient
        # estimator) is b* = E[R · ||∇log π||²] / E[||∇log π||²]. Our simple running
        # average is suboptimal but captures most of the benefit and is far simpler.

    # === REWARD COMPARISON ===
    print("\n=== Reward Comparison ===")

    # Compare learning curves: average reward over the last 50 episodes
    raw_final = sum(raw_rewards[-50:]) / len(raw_rewards[-50:])
    bl_final = sum(bl_rewards[-50:]) / len(bl_rewards[-50:])
    print(f"Average reward (last 50 episodes):")
    print(f"  Raw REINFORCE:  {raw_final:.2f}")
    print(f"  With baseline:  {bl_final:.2f}")

    # === INFERENCE: GENERATE SAMPLES ===
    print("\n=== Generated Samples ===")

    print("\nFrom raw REINFORCE policy:")
    for i in range(10):
        actions, _, reward = generate_trajectory(policy_raw, build_graph=False)
        name = ''.join(VOCAB[a] for a in actions)
        print(f"  {i + 1:>2}. {name:10s} (reward: {reward:+.1f})")

    print("\nFrom baseline REINFORCE policy:")
    for i in range(10):
        actions, _, reward = generate_trajectory(policy_bl, build_graph=False)
        name = ''.join(VOCAB[a] for a in actions)
        print(f"  {i + 1:>2}. {name:10s} (reward: {reward:+.1f})")

    # === BRIDGE TO PPO ===
    print("\n=== Bridge to PPO ===")
    print("""
REINFORCE is elegant but has two practical limitations that PPO addresses:

1. HIGH VARIANCE: Even with a baseline, the gradient estimate is noisy because
   it uses Monte Carlo returns (single-trajectory rewards). PPO uses a learned
   value function (critic) for more accurate advantage estimation, plus GAE
   (Generalized Advantage Estimation) to trade off bias and variance.

2. UNBOUNDED UPDATES: REINFORCE has no constraint on how far a single update
   can move the policy. A lucky high-reward trajectory can cause a catastrophically
   large parameter change, collapsing the policy. PPO adds the "proximal" constraint:
     L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
   where ratio = pi_new(a|s) / pi_old(a|s). This bounds the policy change per step.

REINFORCE: ∇J = E[R · ∇log π]           (what this script implements)
PPO:       ∇J = E[A · ∇log π], clipped   (see microppo.py for the full algorithm)

The baseline in REINFORCE is the seed of the "critic" in actor-critic methods.
PPO replaces our simple running average with a neural network that predicts
expected returns, giving much lower variance at the cost of potential bias.
""".rstrip())

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f}s")
