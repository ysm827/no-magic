"""
Flash Attention computes exact attention identical to standard attention, but processes
Q, K, V in tiles with an online softmax that never materializes the full N*N score matrix.
"""
# Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
# with IO-Awareness" (2022). https://arxiv.org/abs/2205.14135
# Also: Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018).

# === TRADEOFFS ===
# + Exact attention output: mathematically identical to standard implementation
# + O(N) memory instead of O(N^2) by never materializing the full score matrix
# + IO-aware tiling exploits GPU SRAM bandwidth (10-100x faster than HBM)
# - Implementation complexity: online softmax with running max and denominator
# - Block size tuning is hardware-dependent (SRAM size varies across GPUs)
# - Custom CUDA kernel required for real speedups (Python tiling is pedagogical only)
# WHEN TO USE: Any transformer model where attention is the memory or speed
#   bottleneck. Standard in production (used by default in PyTorch 2.0+).
# WHEN NOT TO: When sequence lengths are short enough that O(N^2) memory fits
#   comfortably, or when non-standard attention patterns prevent tiling.

from __future__ import annotations

import math
import random
import time

random.seed(42)

# === CONSTANTS AND CONFIGURATIONS ===

D_HEAD = 16  # head dimension (d_k = d_v)

# Test configurations: (sequence_length, block_size) pairs.
# Multiple configs verify correctness isn't accidental for a single size.
VERIFY_CONFIGS: list[tuple[int, int]] = [
    (32, 8),
    (64, 8),
    (64, 16),
    (48, 12),   # non-power-of-2 to test remainder handling
    (37, 8),    # N not divisible by block_size -- the general case
]

# Sequence lengths for the memory comparison table
MEMORY_SEQ_LENS: list[int] = [16, 32, 64, 128, 256]
MEMORY_BLOCK_SIZES: list[int] = [4, 8, 16]

# Block sizes for the block-size-effect table
BLOCK_EFFECT_N = 64
BLOCK_EFFECT_SIZES: list[int] = [4, 8, 16, 32]

# Signpost: these are tiny dimensions chosen for instant execution and readable output.
# Production Flash Attention operates on d=128, N=8192+, block_size=64-256 (tuned per
# GPU SRAM capacity). The algorithm is identical; only the constants differ.


# === HELPER FUNCTIONS ===
# Plain Python matrix operations. No NumPy, no cleverness -- just explicit loops
# so every memory allocation is visible and countable.

def rand_matrix(rows: int, cols: int) -> list[list[float]]:
    """Random matrix with 1/sqrt(cols) scaling to keep dot products O(1).

    Without scaling, QK^T dot products grow proportional to d, pushing softmax
    into saturation (near-one-hot). Xavier-like init prevents this."""
    s = 1.0 / math.sqrt(cols)
    return [[random.gauss(0.0, s) for _ in range(cols)] for _ in range(rows)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """A[m,k] @ B[k,n] -> C[m,n]."""
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    # Transpose B so inner loop accesses contiguous rows (cache-friendly in C;
    # irrelevant in Python, but mirrors the pattern Flash Attention exploits on GPU)
    bt = [[b[r][c] for r in range(k)] for c in range(n)]
    return [
        [sum(a[i][p] * bt[j][p] for p in range(k)) for j in range(n)]
        for i in range(m)
    ]


def transpose(mat: list[list[float]]) -> list[list[float]]:
    rows = len(mat)
    cols = len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def softmax_rows(mat: list[list[float]]) -> list[list[float]]:
    """Row-wise softmax with numerical stability (subtract row max).

    softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
    Subtracting max(x) prevents exp() overflow while preserving the distribution.
    This is the "two-pass" softmax: pass 1 finds max, pass 2 computes exp and sum."""
    result: list[list[float]] = []
    for row in mat:
        mx = max(row)
        exps = [math.exp(x - mx) for x in row]
        s = sum(exps)
        result.append([e / s for e in exps])
    return result


def max_abs_diff(a: list[list[float]], b: list[list[float]]) -> float:
    """Element-wise maximum absolute difference between two matrices."""
    return max(
        abs(a[i][j] - b[i][j])
        for i in range(len(a))
        for j in range(len(a[0]))
    )


# === STANDARD ATTENTION ===
# The textbook formulation that Flash Attention replaces. Computing this requires
# materializing the full N*N score matrix in memory -- the bottleneck.

def standard_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]]
) -> tuple[list[list[float]], int]:
    """Compute attention by materializing the full N*N score matrix.

    Steps:
      S = Q @ K^T / sqrt(d)    -- score matrix [N, N]
      P = softmax(S, axis=-1)   -- attention weights [N, N], rows sum to 1
      O = P @ V                 -- output [N, d]

    Peak memory: N*N floats for S (or P -- same shape, can overwrite S in-place).
    This O(N^2) memory is the reason standard attention breaks on long sequences.
    At N=128K with float16, the score matrix alone is 32 GB.

    Returns (output, peak_memory_floats)."""
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)

    # S = Q @ K^T -- the N*N matrix we want to avoid materializing
    scores = matmul(q, transpose(k))

    # Scale before softmax (equivalent to scaling Q beforehand, but clearer)
    scores = [[v * scale for v in row] for row in scores]

    # P = softmax(S) -- still N*N
    weights = softmax_rows(scores)

    # O = P @ V -- back to [N, d]
    output = matmul(weights, v)

    # Peak memory: the N*N score/weight matrix
    peak_memory = n * n
    return output, peak_memory


# === FLASH ATTENTION ===
#
# The key insight from Dao et al.: attention can be computed in tiles without ever
# storing the full N*N score matrix. The trick is an "online softmax" that maintains
# running statistics (max and denominator sum) across tiles.
#
# Why tiling matters on GPU (but not in this simulation):
#   GPU memory has two levels: HBM (large, slow) and SRAM (small, fast).
#   Standard attention reads Q,K from HBM, writes N*N scores to HBM, reads them
#   back for softmax, writes weights to HBM, reads them back for P@V.
#   Flash Attention loads tiles of Q,K,V into SRAM, computes attention within SRAM,
#   and writes only the final output to HBM. Total HBM reads drop from O(N^2) to O(N).
#
# This simulation shows the ALGORITHM (tiling + online softmax) that makes this possible.
# It does not show the SPEEDUP, which comes from the GPU memory hierarchy.
#
# === ONLINE SOFTMAX: THE CORE INSIGHT ===
#
# Standard softmax needs ALL scores to compute the denominator:
#   softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
#
# Online softmax processes scores in blocks, maintaining running statistics:
#   m = running maximum (for numerical stability)
#   l = running sum of exp(score - m) (the softmax denominator)
#
# When a new block arrives with local max m_new:
#   1. m_combined = max(m_old, m_new)
#   2. Rescale old sum:    l_old' = l_old * exp(m_old - m_combined)
#   3. Compute new block:  l_new  = sum(exp(scores - m_combined))
#   4. l_combined = l_old' + l_new
#   5. Rescale old output:  O' = O * (l_old / l_combined) * exp(m_old - m_combined)
#      (merge the rescaling of both max and denominator into one step)
#   6. Add new contribution: O' += (1/l_combined) * exp(scores - m_combined) @ V_block
#
# After processing all blocks, O holds the exact same result as standard attention.
# Mathematical proof: the rescaling chain telescopes -- at each step, every previous
# contribution is multiplied by the same correction factor, preserving the ratio of
# exp(score_i) to the total sum across ALL blocks seen so far.

def flash_attention(
    q: list[list[float]], k: list[list[float]], v: list[list[float]],
    block_size: int,
) -> tuple[list[list[float]], int]:
    """Flash Attention: compute exact attention WITHOUT materializing the N*N matrix.

    Process Q, K, V in tiles of size block_size. For each query block, iterate over
    all key/value blocks, accumulating the output using online softmax to maintain
    correct normalization without storing all scores.

    Peak memory: block_size * block_size floats (one tile of scores at a time).
    Compare to N*N for standard attention.

    Returns (output, peak_memory_floats)."""
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)

    # Per-query running statistics. Each query row gets its own max and sum because
    # softmax is applied independently per row (each query attends separately).
    output = [[0.0] * d for _ in range(n)]
    row_max = [float("-inf")] * n   # m_i: running max of scores for query i
    row_sum = [0.0] * n             # l_i: running sum of exp(score - m_i) for query i

    peak_memory = 0

    # Outer loop: iterate over blocks of queries
    for q_start in range(0, n, block_size):
        q_end = min(q_start + block_size, n)
        q_block = q[q_start:q_end]
        bq = q_end - q_start  # actual block size (may be smaller at boundary)

        # Inner loop: for each query block, sweep over ALL key/value blocks.
        # This is the "tiling" -- only one bq * bk tile of scores exists at a time.
        for k_start in range(0, n, block_size):
            k_end = min(k_start + block_size, n)
            k_block = k[k_start:k_end]
            v_block = v[k_start:k_end]
            bk = k_end - k_start

            # Track simulated memory: the score tile is the largest temporary
            peak_memory = max(peak_memory, bq * bk)

            # Step 1: Compute partial scores S_ij = Q_block @ K_block^T / sqrt(d)
            # This is a bq * bk matrix -- NOT N*N.
            scores_tile: list[list[float]] = []
            for qi in range(bq):
                row: list[float] = []
                for ki in range(bk):
                    dot = sum(q_block[qi][c] * k_block[ki][c] for c in range(d))
                    row.append(dot * scale)
                scores_tile.append(row)

            # Step 2: For each query row in this block, apply the online softmax update
            for qi in range(bq):
                global_i = q_start + qi  # index into the full output

                # Local max for this tile row
                #   m_ij = max(S_ij[qi, :])
                m_tile = max(scores_tile[qi])

                # Combined max: max of running max and this tile's max
                #   m_new = max(m_old, m_tile)
                m_old = row_max[global_i]
                m_new = max(m_old, m_tile)

                # Rescale factor for old accumulator:
                #   When the max increases, all previous exp() values were computed
                #   relative to the old max. Multiplying by exp(m_old - m_new) corrects
                #   them to be relative to the new max.
                #   exp(score - m_old) * exp(m_old - m_new) = exp(score - m_new)
                if m_old == float("-inf"):
                    # First block for this query -- no previous accumulator to rescale
                    old_scale = 0.0
                else:
                    old_scale = math.exp(m_old - m_new)

                # Compute exp(score - m_new) for each score in this tile row
                #   P_ij[qi, ki] = exp(S_ij[qi, ki] - m_new)
                exp_scores = [math.exp(s - m_new) for s in scores_tile[qi]]

                # Sum of new exponentiated scores: contribution to the denominator
                new_sum = sum(exp_scores)

                # Update running denominator:
                #   l_new = l_old * exp(m_old - m_new) + sum(exp(scores - m_new))
                l_old = row_sum[global_i]
                l_new = l_old * old_scale + new_sum

                # Update output accumulator:
                #   O_i = O_i * (l_old * old_scale / l_new) + (1/l_new) * P_ij @ V_block
                #
                # The first term rescales previous output (correcting for the new max
                # and reweighting by the updated denominator). The second term adds
                # this tile's weighted contribution, already normalized.
                if l_new > 0.0:
                    # Rescale previous accumulator
                    rescale = (l_old * old_scale) / l_new
                    for c in range(d):
                        output[global_i][c] *= rescale

                    # Add new contribution: (1/l_new) * sum_ki(exp_scores[ki] * V[ki, c])
                    inv_l = 1.0 / l_new
                    for ki in range(bk):
                        w = exp_scores[ki] * inv_l
                        for c in range(d):
                            output[global_i][c] += w * v_block[ki][c]

                # Update running statistics
                row_max[global_i] = m_new
                row_sum[global_i] = l_new

    # No final normalization needed -- output is already correctly normalized at each
    # step because we divide by l (the running denominator) incrementally. This is
    # different from some presentations that defer normalization to the end; here
    # the output is always "fully normalized so far" after each inner iteration.

    return output, peak_memory


# === VERIFICATION ===

def verify(
    n: int, d: int, block_size: int, tolerance: float = 1e-6
) -> tuple[bool, float, int, int]:
    """Run standard and flash attention on identical inputs, check outputs match.

    Returns (passed, max_diff, standard_memory, flash_memory)."""
    q = rand_matrix(n, d)
    k = rand_matrix(n, d)
    v = rand_matrix(n, d)

    out_std, mem_std = standard_attention(q, k, v)
    out_flash, mem_flash = flash_attention(q, k, v, block_size)

    diff = max_abs_diff(out_std, out_flash)
    passed = diff < tolerance
    return passed, diff, mem_std, mem_flash


# === MEMORY ANALYSIS ===
# Standard attention: peak memory = N^2 (the full score matrix).
# Flash attention: peak memory = B^2 (one tile of scores), independent of N.
#
# In practice on GPU, Flash Attention also reduces HBM I/O from O(N^2 * d) to
# O(N * d^2 / SRAM_size), but that's an I/O complexity argument about the memory
# hierarchy -- not simulable in pure Python.

def format_int(n: int) -> str:
    """Format integer with comma separators."""
    return f"{n:,}"


def print_memory_table(seq_lens: list[int], block_sizes: list[int]) -> None:
    """Print comparison of peak memory for standard vs flash across configurations."""
    # Build header
    header = f"{'Seq Length (N)':>14}   {'Standard (floats)':>18}"
    for b in block_sizes:
        header += f"   {'Flash B=' + str(b):>12}"
    print(header)

    separator = "\u2500" * 14 + "   " + "\u2500" * 18
    for _ in block_sizes:
        separator += "   " + "\u2500" * 12
    print(separator)

    for n in seq_lens:
        std_mem = n * n
        row = f"{n:>14}   {format_int(std_mem):>18}"
        for b in block_sizes:
            flash_mem = b * b
            row += f"   {format_int(flash_mem):>12}"
        print(row)


def print_block_effect_table(n: int, d: int, block_sizes: list[int]) -> None:
    """Show how block size affects memory and tile count."""
    header = f"{'Block Size':>10}   {'Memory (floats)':>15}   {'Num Tiles':>9}"
    print(header)
    separator = "\u2500" * 10 + "   " + "\u2500" * 15 + "   " + "\u2500" * 9
    print(separator)

    for b in block_sizes:
        mem = b * b
        # Number of tiles: ceil(N/B) query blocks * ceil(N/B) key blocks
        num_q_blocks = math.ceil(n / b)
        num_k_blocks = math.ceil(n / b)
        num_tiles = num_q_blocks * num_k_blocks
        print(f"{b:>10}   {format_int(mem):>15}   {num_tiles:>9}")


# === MAIN ===

if __name__ == "__main__":
    print("=== Flash Attention: Algorithmic Simulation ===\n")

    # Signpost: make the simulation-vs-optimization distinction immediately clear
    print("Signpost: This is an algorithmic simulation, not a performance benchmark.")
    print("Pure Python is slower than standard attention here. The point is showing WHAT")
    print("Flash Attention does (tiled computation, online softmax), not achieving speedup.")
    print("On GPU, the speedup comes from keeping tiles in SRAM (fast, small) instead of")
    print("reading/writing the N*N matrix from HBM (large, slow).\n")

    # --- Verification ---
    print("--- Verification ---")
    all_passed = True

    for n, block_size in VERIFY_CONFIGS:
        print(f"\nConfig: N={n}, d={D_HEAD}, block_size={block_size}")
        t0 = time.time()
        passed, diff, mem_std, mem_flash = verify(n, D_HEAD, block_size)
        elapsed = time.time() - t0

        print(f"  Standard attention: computed (peak memory: {format_int(mem_std)} floats)")
        print(f"  Flash attention:    computed (peak memory: {format_int(mem_flash)} floats)")
        print(f"  Max element difference: {diff:.2e}")
        print(f"  Time: {elapsed*1000:.1f} ms")

        if passed:
            print(f"  PASS: outputs match within 1e-6 tolerance")
        else:
            print(f"  FAIL: outputs diverge beyond 1e-6 tolerance")
            all_passed = False

    print(f"\nOverall: {'all configurations passed' if all_passed else 'SOME CONFIGURATIONS FAILED'}")

    # --- Memory Comparison ---
    # This table is the core result: standard attention memory grows as O(N^2)
    # while flash attention memory stays at O(B^2) regardless of sequence length.
    print("\n--- Memory Comparison ---")
    print("Peak floats allocated for the score matrix (standard) vs one tile (flash):\n")
    print_memory_table(MEMORY_SEQ_LENS, MEMORY_BLOCK_SIZES)

    print(f"\nStandard attention memory grows as O(N^2) -- doubling N quadruples memory.")
    print(f"Flash attention memory is O(B^2), independent of sequence length N.")
    print(f"At N=128K with B=128, standard needs 16 billion floats; flash needs 16,384.")

    # --- Block Size Effect ---
    # Smaller blocks = less memory but more tiles (more loop iterations).
    # On GPU, smaller blocks mean more SRAM loads -- there's a sweet spot where
    # blocks fill SRAM without wasting capacity.
    print(f"\n--- Block Size Effect ---")
    print(f"For N={BLOCK_EFFECT_N}, d={D_HEAD}:\n")
    print_block_effect_table(BLOCK_EFFECT_N, D_HEAD, BLOCK_EFFECT_SIZES)

    print(f"\nSmaller blocks use less memory but require more tiles (iterations).")
    print(f"On GPU, the optimal block size fills SRAM: A100 has 192KB SRAM,")
    print(f"fitting B~128 for d=128 in float16. Pure Python has no SRAM,")
    print(f"so block size affects only iteration count here.")

    # Signpost: no runtime benchmark section. Flash Attention's speedup comes from
    # GPU memory hierarchy (SRAM vs HBM), not from reducing FLOPs. In pure Python,
    # interpreter overhead per operation dominates, making flash SLOWER than standard.
    # On GPU, Flash Attention is 2-4x faster because it reduces HBM reads from O(N^2)
    # to O(N). The memory comparison tables above are the meaningful result here.
