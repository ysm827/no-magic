"""
How position information gets baked into attention through rotation matrices — why RoPE
replaced learned embeddings in every modern LLM.
"""
# Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
# Vaswani et al., "Attention Is All You Need" (2017) for sinusoidal baselines.
# All modern open-weight LLMs (LLaMA, Mistral, Gemma, Qwen, DeepSeek) use RoPE or variants.

# === TRADEOFFS ===
# + Relative position encoding: attention score depends on distance, not absolute position
# + Extrapolates to longer sequences than seen during training (with appropriate scaling)
# + No learned parameters: purely analytical formulation (zero extra memory)
# - Requires even head dimensions (pairs of features rotated together)
# - Performance degrades at extreme extrapolation lengths without NTK/YaRN scaling
# - Rotation computation adds overhead per attention head per layer
# WHEN TO USE: Any transformer-based language model. RoPE is the de facto standard
#   for modern open-weight LLMs (LLaMA, Mistral, Gemma, DeepSeek).
# WHEN NOT TO: Non-sequential data where relative position is meaningless, or
#   encoder-only models where learned absolute embeddings work equally well.

from __future__ import annotations

import math
import random

random.seed(42)

# === CONSTANTS ===

D_MODEL = 64          # embedding dimension
N_HEADS = 4           # number of attention heads
HEAD_DIM = D_MODEL // N_HEADS  # = 16, must be even for RoPE's pairwise rotation
SEQ_LEN = 32          # sequence length for demonstrations
MAX_POS = 64          # maximum position for learned embeddings (training range)
BASE_THETA = 10000.0  # RoPE base frequency from the original paper

# Signpost: production LLMs use d_model=4096+, head_dim=128, seq_len=8192-128K.
# These toy values preserve every algorithmic detail.


# === HELPER FUNCTIONS ===

def rand_vector(d: int) -> list[float]:
    """Random vector with unit-variance initialization."""
    s = 1.0 / math.sqrt(d)
    return [random.gauss(0, s) for _ in range(d)]


def rand_matrix(rows: int, cols: int) -> list[list[float]]:
    """Random matrix with Xavier-like scaling."""
    s = 1.0 / math.sqrt(cols)
    return [[random.gauss(0, s) for _ in range(cols)] for _ in range(rows)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def softmax(row: list[float]) -> list[float]:
    """Numerically stable softmax."""
    mx = max(row)
    exps = [math.exp(x - mx) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


# === SINUSOIDAL POSITION ENCODING (Vaswani 2017) ===
# PE(pos, 2i)   = sin(pos / 10000^(2i/d))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
#
# Each dimension oscillates at a different frequency. Low dimensions cycle fast
# (capturing nearby positions), high dimensions cycle slow (capturing global position).
# The key insight: PE(pos+k) can be expressed as a linear function of PE(pos),
# which lets the model learn to attend to relative offsets. But this linearity
# is approximate — the attention score Q^T K doesn't decompose cleanly into a
# function of (m-n) alone.

def sinusoidal_encoding(pos: int, d: int) -> list[float]:
    """Fixed sinusoidal position encoding for a single position."""
    pe = [0.0] * d
    for i in range(d // 2):
        # theta_i decreases exponentially with dimension index —
        # each pair captures a different frequency band
        theta = pos / (BASE_THETA ** (2 * i / d))
        pe[2 * i] = math.sin(theta)
        pe[2 * i + 1] = math.cos(theta)
    return pe


def apply_sinusoidal(vec: list[float], pos: int) -> list[float]:
    """Add sinusoidal encoding to input vector (additive, like the original Transformer)."""
    pe = sinusoidal_encoding(pos, len(vec))
    return [v + p for v, p in zip(vec, pe)]


# === LEARNED POSITION EMBEDDINGS (GPT-2 style) ===
# A lookup table of trainable vectors, one per position index.
# Simple and effective, but fundamentally limited:
#   1. Cannot generalize beyond max_pos (hard length ceiling)
#   2. Position info is additive noise on the content signal
#   3. No built-in relative position bias — the model must learn it implicitly
# GPT-2 and early GPT-3 used this approach.

def make_learned_embeddings(max_pos: int, d: int) -> list[list[float]]:
    """Initialize a position embedding table (normally trained, here random)."""
    return [rand_vector(d) for _ in range(max_pos)]


def apply_learned(vec: list[float], pos: int, table: list[list[float]]) -> list[float]:
    """Add learned position embedding (additive, same as sinusoidal)."""
    if pos >= len(table):
        # Beyond training range — no valid embedding exists.
        # This is the fundamental length extrapolation failure mode.
        return vec[:]
    return [v + e for v, e in zip(vec, table[pos])]


# === ROTARY POSITION EMBEDDINGS (RoPE) — Su et al. 2021 ===
#
# Core idea: instead of ADDING position info to the embedding, ROTATE the
# query and key vectors by a position-dependent angle. For dimension pair
# (2i, 2i+1), rotate by angle m * theta_i where:
#
#   theta_i = BASE^(-2i/d)    (same frequency schedule as sinusoidal)
#
# The rotation matrix for one pair:
#   R(angle) = [[cos(angle), -sin(angle)],
#               [sin(angle),  cos(angle)]]
#
# Why this is better than additive encodings:
#
# When computing Q_m^T K_n (the attention score between positions m and n):
#   (R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R(n-m) k
#
# The rotation matrices compose: R^T(m) @ R(n) = R(n-m). This means the
# attention score depends ONLY on the relative distance (n-m), not the
# absolute positions m and n individually. This is translation invariance —
# "the cat sat" attends the same way regardless of where it appears in
# the sequence.
#
# Additive encodings (sinusoidal/learned) don't have this property because
# (q + pe_m)^T (k + pe_n) = q^T k + q^T pe_n + pe_m^T k + pe_m^T pe_n
# The cross terms q^T pe_n and pe_m^T k depend on absolute positions.

def rope_frequencies(d: int, base: float = BASE_THETA) -> list[float]:
    """Precompute theta_i = base^(-2i/d) for each dimension pair.

    These are the angular frequencies — low-index pairs rotate fast,
    high-index pairs rotate slow. Identical frequency schedule to
    sinusoidal encoding, but applied multiplicatively (rotation) not
    additively."""
    return [1.0 / (base ** (2 * i / d)) for i in range(d // 2)]


def apply_rope(vec: list[float], pos: int, freqs: list[float]) -> list[float]:
    """Rotate vector by position-dependent angles.

    For each consecutive pair (x_2i, x_{2i+1}), apply the 2D rotation:
      x_2i'    = x_2i * cos(pos * theta_i) - x_{2i+1} * sin(pos * theta_i)
      x_{2i+1}' = x_2i * sin(pos * theta_i) + x_{2i+1} * cos(pos * theta_i)

    This is equivalent to multiplying by the block-diagonal rotation matrix
    R(pos) = diag(R_0(pos*theta_0), R_1(pos*theta_1), ..., R_{d/2-1}(pos*theta_{d/2-1}))
    """
    result = [0.0] * len(vec)
    for i, theta in enumerate(freqs):
        angle = pos * theta
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # 2D rotation applied to the (2i, 2i+1) pair
        result[2 * i] = vec[2 * i] * cos_a - vec[2 * i + 1] * sin_a
        result[2 * i + 1] = vec[2 * i] * sin_a + vec[2 * i + 1] * cos_a
    return result


def rope_attention_score(
    q: list[float], k: list[float], pos_q: int, pos_k: int, freqs: list[float]
) -> float:
    """Compute attention score between q at position pos_q and k at position pos_k.

    score = (R(pos_q) @ q)^T @ (R(pos_k) @ k) / sqrt(d)
    Because R^T(m) R(n) = R(n-m), this equals q^T R(pos_k - pos_q) k / sqrt(d).
    The score depends only on the relative distance, not absolute positions."""
    q_rot = apply_rope(q, pos_q, freqs)
    k_rot = apply_rope(k, pos_k, freqs)
    return dot(q_rot, k_rot) / math.sqrt(len(q))


# === NTK-AWARE SCALING ===
#
# Problem: RoPE trained at context length L fails at length L' > L because
# the high-frequency rotation pairs complete too many full rotations,
# wrapping around and creating position aliasing.
#
# NTK-aware interpolation (Reddit user "bloc97", adopted by Code LLaMA):
# Instead of linearly interpolating positions (which compresses ALL frequencies
# equally, losing fine-grained position info), scale the BASE frequency:
#
#   base' = base * alpha^(d/(d-2))
#
# where alpha = L'/L (the length extension ratio).
#
# This spreads the frequency spectrum: high frequencies (which alias first)
# get slowed down more than low frequencies (which still have room).
# The result is smooth extrapolation without fine-tuning.

def ntk_scaled_frequencies(
    d: int, base: float, scale_factor: float
) -> list[float]:
    """Compute RoPE frequencies with NTK-aware base scaling.

    scale_factor = target_length / training_length (e.g., 4.0 for 4x extension).
    The exponent d/(d-2) ensures the frequency adjustment is distributed
    across the spectrum rather than applied uniformly."""
    scaled_base = base * (scale_factor ** (d / (d - 2)))
    return [1.0 / (scaled_base ** (2 * i / d)) for i in range(d // 2)]


# === COMPARISON AND DEMONSTRATIONS ===

def demonstrate_relative_position_property(freqs: list[float]) -> None:
    """The defining property of RoPE: attention scores depend only on (m-n).

    For the same q and k vectors, score(pos=2, pos=5) should equal
    score(pos=10, pos=13) because both have relative distance 3.
    This does NOT hold for sinusoidal or learned embeddings."""
    print("\n--- Relative Position Property ---\n")
    d = len(freqs) * 2
    q = rand_vector(d)
    k = rand_vector(d)

    test_pairs = [(2, 5, 3), (10, 13, 3), (0, 3, 3), (50, 53, 3), (100, 103, 3)]

    print("RoPE scores for different absolute positions, same relative distance (3):")
    rope_scores = []
    for m, n, _ in test_pairs:
        s = rope_attention_score(q, k, m, n, freqs)
        rope_scores.append(s)
        print(f"  score(pos={m:>3}, pos={n:>3}) = {s:.8f}")

    max_diff = max(abs(s - rope_scores[0]) for s in rope_scores)
    print(f"  Max deviation: {max_diff:.2e} (should be ~0 for exact relative encoding)")

    # Contrast with sinusoidal — same experiment
    print("\nSinusoidal scores for same pairs (additive encoding, NOT purely relative):")
    sin_scores = []
    scale = 1.0 / math.sqrt(d)
    for m, n, _ in test_pairs:
        q_enc = apply_sinusoidal(q, m)
        k_enc = apply_sinusoidal(k, n)
        s = dot(q_enc, k_enc) * scale
        sin_scores.append(s)
        print(f"  score(pos={m:>3}, pos={n:>3}) = {s:.8f}")

    max_diff_sin = max(abs(s - sin_scores[0]) for s in sin_scores)
    print(f"  Max deviation: {max_diff_sin:.2e} (nonzero — sinusoidal is NOT purely relative)")


def demonstrate_length_extrapolation(freqs: list[float]) -> None:
    """Show how each method handles positions beyond training range.

    Learned embeddings fail completely (no vector for unseen positions).
    Sinusoidal degrades gracefully (fixed formula works at any position).
    RoPE extrapolates naturally but accumulates rotation — high-frequency
    pairs may alias at extreme lengths. NTK scaling fixes this."""
    print("\n--- Length Extrapolation ---\n")
    d = len(freqs) * 2
    q = rand_vector(d)
    k = rand_vector(d)
    pos_table = make_learned_embeddings(MAX_POS, d)

    test_positions = [10, 32, 63, 100, 500, 2000]
    scale = 1.0 / math.sqrt(d)

    print(f"Attention score: q at pos=0 attending to k at various positions")
    print(f"(Learned embeddings trained up to pos={MAX_POS})\n")
    print(f"  {'pos_k':>6}  {'Sinusoidal':>12}  {'Learned':>12}  {'RoPE':>12}  {'NTK(4x)':>12}")
    print(f"  {'':->6}  {'':->12}  {'':->12}  {'':->12}  {'':->12}")

    # NTK frequencies for 4x context extension
    ntk_freqs = ntk_scaled_frequencies(d, BASE_THETA, scale_factor=4.0)

    for pos in test_positions:
        # Sinusoidal
        q_sin = apply_sinusoidal(q, 0)
        k_sin = apply_sinusoidal(k, pos)
        s_sin = dot(q_sin, k_sin) * scale

        # Learned (returns unmodified vector beyond MAX_POS)
        q_lrn = apply_learned(q, 0, pos_table)
        k_lrn = apply_learned(k, pos, pos_table)
        s_lrn = dot(q_lrn, k_lrn) * scale
        lrn_str = f"{s_lrn:.6f}" if pos < MAX_POS else "N/A (OOB)"

        # RoPE
        s_rope = rope_attention_score(q, k, 0, pos, freqs)

        # NTK-scaled RoPE
        s_ntk = rope_attention_score(q, k, 0, pos, ntk_freqs)

        print(f"  {pos:>6}  {s_sin:>12.6f}  {lrn_str:>12}  {s_rope:>12.6f}  {s_ntk:>12.6f}")

    print(f"\n  Learned embeddings: hard failure beyond pos={MAX_POS}.")
    print(f"  RoPE: continues to work but high-freq pairs rotate too fast at extreme range.")
    print(f"  NTK scaling: slows high-freq rotations, smoother extrapolation.")


def demonstrate_frequency_spectrum(freqs: list[float]) -> None:
    """Visualize the rotation angle per position for each dimension pair.

    Low-index pairs (fast frequency) distinguish nearby positions.
    High-index pairs (slow frequency) distinguish distant positions.
    This multi-scale encoding is why RoPE captures both local and global
    position relationships."""
    print("\n--- Frequency Spectrum ---\n")
    d = len(freqs) * 2
    print(f"  Rotation angles (radians) at position 1 for each dimension pair:")
    print(f"  {'Pair':>6}  {'theta_i':>12}  {'angle(pos=1)':>14}  {'wavelength':>12}")
    print(f"  {'':->6}  {'':->12}  {'':->14}  {'':->12}")

    for i, theta in enumerate(freqs):
        # Wavelength = how many positions for a full 2*pi rotation
        wavelength = 2 * math.pi / theta
        print(f"  {f'({2*i},{2*i+1})':>6}  {theta:>12.6f}  {theta:>14.6f}  {wavelength:>12.1f}")

    print(f"\n  Pair (0,1) completes a full rotation every {2*math.pi/freqs[0]:.0f} positions.")
    print(f"  Pair ({d-2},{d-1}) completes a full rotation every "
          f"{2*math.pi/freqs[-1]:.0f} positions.")
    print(f"  This geometric spacing gives RoPE multi-scale position sensitivity.")


def demonstrate_ntk_scaling(freqs: list[float]) -> None:
    """Compare standard RoPE frequencies with NTK-scaled frequencies."""
    print("\n--- NTK-Aware Scaling ---\n")
    d = len(freqs) * 2

    print(f"  {'Scale':>6}  {'Scaled Base':>12}  {'theta_0':>12}  {'theta_last':>12}")
    print(f"  {'':->6}  {'':->12}  {'':->12}  {'':->12}")
    print(f"  {'1x':>6}  {BASE_THETA:>12.1f}  {freqs[0]:>12.6f}  {freqs[-1]:>12.10f}")

    for sf in [2.0, 4.0, 8.0]:
        ntk = ntk_scaled_frequencies(d, BASE_THETA, sf)
        scaled_base = BASE_THETA * (sf ** (d / (d - 2)))
        print(f"  {f'{sf:.0f}x':>6}  {scaled_base:>12.1f}  {ntk[0]:>12.6f}  {ntk[-1]:>12.10f}")

    print(f"\n  Higher scale factors slow all frequencies proportionally.")
    print(f"  The d/(d-2) exponent distributes adjustment across the spectrum.")


# === MAIN: RUN ALL DEMONSTRATIONS ===

if __name__ == "__main__":
    print("=== Position Encoding Comparison: Sinusoidal vs Learned vs RoPE ===\n")
    print(f"Config: d_model={D_MODEL}, n_heads={N_HEADS}, head_dim={HEAD_DIM}, "
          f"seq_len={SEQ_LEN}, max_pos={MAX_POS}, base={BASE_THETA}\n")

    freqs = rope_frequencies(HEAD_DIM)

    # --- 1. Relative position property (RoPE's key advantage) ---
    demonstrate_relative_position_property(freqs)

    # --- 2. Frequency spectrum visualization ---
    demonstrate_frequency_spectrum(freqs)

    # --- 3. Length extrapolation comparison ---
    demonstrate_length_extrapolation(freqs)

    # --- 4. NTK-aware scaling ---
    demonstrate_ntk_scaling(freqs)

    # --- 5. RoPE score consistency check ---
    print("\n--- RoPE Score Consistency (Numerical Verification) ---\n")
    q_test = rand_vector(HEAD_DIM)
    k_test = rand_vector(HEAD_DIM)

    # Verify R^T(m) R(n) = R(n-m) numerically
    # score(m, n) should equal score(0, n-m) for any m, n
    test_cases = [(0, 5), (3, 8), (10, 15), (20, 25), (50, 55)]
    print("Verifying score(m, n) == score(0, n-m) for relative distance 5:")
    for m, n in test_cases:
        s_mn = rope_attention_score(q_test, k_test, m, n, freqs)
        s_0d = rope_attention_score(q_test, k_test, 0, n - m, freqs)
        match = abs(s_mn - s_0d) < 1e-10
        print(f"  score({m:>2},{n:>2}) = {s_mn:.10f}  "
              f"score(0,{n-m:>2}) = {s_0d:.10f}  match: {match}")

    # === RESULTS TABLE ===
    print("\n=== Position Encoding Comparison Summary ===\n")
    print(f"  {'Property':<30}  {'Sinusoidal':>12}  {'Learned':>10}  {'RoPE':>10}")
    print(f"  {'':->30}  {'':->12}  {'':->10}  {'':->10}")
    print(f"  {'Relative position bias':<30}  {'Approximate':>12}  {'None':>10}  {'Exact':>10}")
    print(f"  {'Length extrapolation':<30}  {'Yes (fixed)':>12}  {'No':>10}  {'Yes*':>10}")
    print(f"  {'Trainable parameters':<30}  {'0':>12}  {f'{MAX_POS}*d':>10}  {'0':>10}")
    print(f"  {'Application method':<30}  {'Additive':>12}  {'Additive':>10}  {'Rotation':>10}")
    print(f"  {'Used in production':<30}  {'Legacy':>12}  {'GPT-2':>10}  {'All new':>10}")

    print(f"\n  * RoPE extrapolates naturally but benefits from NTK scaling for >2x extension.")
    print(f"\n=== Key Takeaways ===\n")
    print("1. RoPE encodes position through rotation, not addition. This makes attention")
    print("   scores depend on relative distance (m-n), not absolute positions m and n.")
    print("2. The rotation matrix R^T(m) R(n) = R(n-m) is the core mathematical property.")
    print("   Sinusoidal encoding approximates this but doesn't achieve it exactly.")
    print("3. Learned embeddings have zero relative position bias and cannot extrapolate")
    print("   beyond training length — a hard ceiling that RoPE eliminates.")
    print("4. NTK-aware scaling adjusts the base frequency to prevent high-frequency")
    print("   aliasing during context length extension, adopted by Code LLaMA and others.")
    print("5. Every major open-weight LLM (LLaMA, Mistral, Gemma, Qwen, DeepSeek) uses")
    print("   RoPE. It won because relative position bias + zero extra parameters +")
    print("   length extrapolation is strictly better than the alternatives.")
