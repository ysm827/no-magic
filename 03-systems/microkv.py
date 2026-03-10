"""
Why autoregressive generation recomputes redundant work at every step, and how the KV cache
eliminates that redundancy by memoizing key/value projections across the sequence.
"""
# Reference: Pope et al., "Efficiently Scaling Transformer Inference" (2022) for KV cache
# analysis. Kwon et al., "Efficient Memory Management for Large Language Model Serving
# with PagedAttention" (2023) for paged allocation. Architecture follows the microgpt
# pattern (Radford et al., 2019) with pedagogical simplifications.

# === TRADEOFFS ===
# + Eliminates redundant key/value recomputation during autoregressive generation
# + Linear speedup: each generation step processes only the new token
# + Paged allocation reduces memory fragmentation for variable-length sequences
# - Memory grows linearly with sequence length (long contexts exhaust GPU memory)
# - Cache management adds implementation complexity (especially for batched serving)
# - No benefit during training (full sequence is processed in parallel)
# WHEN TO USE: Autoregressive inference with any transformer-based language model.
#   KV-caching is mandatory for practical LLM serving.
# WHEN NOT TO: Training (forward pass is already parallel), or encoder-only models
#   (BERT-style) where all tokens are processed simultaneously.

from __future__ import annotations

import math
import os
import random
import time
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture — minimal viable transformer for demonstrating cache mechanics.
# The model only needs to produce non-random outputs; generation quality is secondary.
N_EMBD = 16
N_HEAD = 2
N_LAYER = 1
BLOCK_SIZE = 32
HEAD_DIM = N_EMBD // N_HEAD  # 8

# Training
LEARNING_RATE = 0.01
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
NUM_STEPS = 300

# Inference comparison
GEN_LEN = 16  # characters to generate for the comparison
PAGE_BLOCK_SIZE = 4  # positions per block in paged attention simulation

# Data
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: production KV caches store thousands of positions across dozens of layers
# with 128-dimensional heads. Our toy dimensions (1 layer, 8-dim heads) preserve the
# algorithmic structure while keeping runtime under a minute.


# === DATA LOADING ===

def load_data(url: str, filename: str) -> list[str]:
    """Download and parse the training corpus."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


# === SCALAR AUTOGRAD ENGINE (compact) ===
# Used only for training. After training completes, we extract weights as plain floats
# and run inference with raw arithmetic — isolating the KV cache comparison from autograd
# overhead and making multiply counts exact.

class Value:
    """Scalar with reverse-mode autodiff. See docs/autograd-interface.md for spec."""
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

    def exp(self) -> Value:
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def relu(self) -> Value:
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self) -> None:
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
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# Compact Value class used solely for training the tiny model. Inference comparison
# operates on extracted plain floats to count multiplies without autograd overhead.
# See docs/autograd-interface.md for the canonical interface.


# === TRAINING HELPERS (Value-based) ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]

def linear_v(x: list[Value], w: list[list[Value]]) -> list[Value]:
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]

def softmax_v(logits: list[Value]) -> list[Value]:
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm_v(x: list[Value]) -> list[Value]:
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def safe_log(prob: Value) -> Value:
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


def gpt_forward_train(
    token_id: int, pos_id: int,
    keys: list[list[list[Value]]], values: list[list[list[Value]]],
    params: dict[str, list[list[Value]]],
) -> list[Value]:
    """Single-token forward pass for training (Value-based)."""
    x = [t + p for t, p in zip(params['wte'][token_id], params['wpe'][pos_id])]
    x = rmsnorm_v(x)
    for li in range(N_LAYER):
        x_res = x
        x = rmsnorm_v(x)
        q = linear_v(x, params[f'l{li}.wq'])
        k = linear_v(x, params[f'l{li}.wk'])
        v = linear_v(x, params[f'l{li}.wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn: list[Value] = []
        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            q_h = q[hs:hs + HEAD_DIM]
            k_h = [kt[hs:hs + HEAD_DIM] for kt in keys[li]]
            v_h = [vt[hs:hs + HEAD_DIM] for vt in values[li]]
            scores = [sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / (HEAD_DIM ** 0.5)
                      for t in range(len(k_h))]
            weights = softmax_v(scores)
            x_attn.extend([sum(weights[t] * v_h[t][j] for t in range(len(v_h)))
                           for j in range(HEAD_DIM)])
        x = linear_v(x_attn, params[f'l{li}.wo'])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x
        x = rmsnorm_v(x)
        x = linear_v(x, params[f'l{li}.fc1'])
        x = [xi.relu() for xi in x]
        x = linear_v(x, params[f'l{li}.fc2'])
        x = [a + b for a, b in zip(x, x_res)]
    return linear_v(x, params['lm_head'])


# === PLAIN-FLOAT INFERENCE HELPERS ===
# After training, we extract Value.data into nested lists of plain floats.
# All inference comparison functions below operate on floats and count multiplies.

def extract(w: list[list[Value]]) -> list[list[float]]:
    """Strip autograd wrappers: list[list[Value]] -> list[list[float]]."""
    return [[v.data for v in row] for row in w]


def linear_f(x: list[float], w: list[list[float]], counter: list[int]) -> list[float]:
    """Matrix-vector multiply on plain floats. Counts every scalar multiply."""
    counter[0] += len(w) * len(x)
    return [sum(w[i][j] * x[j] for j in range(len(x))) for i in range(len(w))]


def softmax_f(logits: list[float]) -> list[float]:
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def rmsnorm_f(x: list[float]) -> list[float]:
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# === INFERENCE WITHOUT KV CACHE ===
# At each generation step, recompute Q/K/V projections for ALL positions from scratch.
# This is how attention would work if we treated every step as independent: feed the
# entire sequence, attend over everything, discard intermediate state, repeat.
# Total work for T steps: sum(t * C_proj + t^2 * C_attn for t in 1..T) ~ O(T^3)

def generate_no_cache(
    prompt_tok: int, wf: dict[str, list[list[float]]],
    vocab_size: int, gen_len: int,
) -> tuple[list[int], list[int]]:
    """Generate tokens WITHOUT KV cache. Returns (tokens, muls_per_step)."""
    tokens = [prompt_tok]
    muls_per_step: list[int] = []

    for step in range(gen_len):
        counter = [0]
        seq = tokens  # reprocess entire sequence from scratch every step
        seq_len = len(seq)

        # Embed all positions
        embeddings: list[list[float]] = []
        for pos, tid in enumerate(seq):
            x = [wf['wte'][tid][j] + wf['wpe'][pos][j] for j in range(N_EMBD)]
            embeddings.append(rmsnorm_f(x))

        # Transformer layers — recompute Q, K, V for EVERY position
        hiddens = [row[:] for row in embeddings]
        for li in range(N_LAYER):
            residuals = [row[:] for row in hiddens]
            normed = [rmsnorm_f(h) for h in hiddens]

            # Project all positions to Q, K, V — this is the redundant work.
            # Positions 0..(t-1) were already projected on previous steps.
            all_q = [linear_f(normed[p], wf[f'l{li}.wq'], counter) for p in range(seq_len)]
            all_k = [linear_f(normed[p], wf[f'l{li}.wk'], counter) for p in range(seq_len)]
            all_v = [linear_f(normed[p], wf[f'l{li}.wv'], counter) for p in range(seq_len)]

            # Causal multi-head attention over the full sequence
            attn_out: list[list[float]] = []
            for pos in range(seq_len):
                head_cat: list[float] = []
                for h in range(N_HEAD):
                    hs = h * HEAD_DIM
                    q_h = all_q[pos][hs:hs + HEAD_DIM]
                    # Causal: only attend to positions 0..pos
                    scores: list[float] = []
                    for t in range(pos + 1):
                        k_h = all_k[t][hs:hs + HEAD_DIM]
                        dot = sum(q_h[j] * k_h[j] for j in range(HEAD_DIM))
                        counter[0] += HEAD_DIM
                        scores.append(dot / (HEAD_DIM ** 0.5))
                    weights = softmax_f(scores)
                    for j in range(HEAD_DIM):
                        val = 0.0
                        for t in range(pos + 1):
                            val += weights[t] * all_v[t][hs + j]
                            counter[0] += 1
                        head_cat.append(val)
                attn_out.append(head_cat)

            # Output projection + residual
            for pos in range(seq_len):
                projected = linear_f(attn_out[pos], wf[f'l{li}.wo'], counter)
                hiddens[pos] = [a + b for a, b in zip(projected, residuals[pos])]

            # MLP + residual
            for pos in range(seq_len):
                res2 = hiddens[pos][:]
                h = rmsnorm_f(hiddens[pos])
                h = linear_f(h, wf[f'l{li}.fc1'], counter)
                h = [max(0.0, v) for v in h]
                h = linear_f(h, wf[f'l{li}.fc2'], counter)
                hiddens[pos] = [a + b for a, b in zip(h, res2)]

        # Logits from last position only
        logits = linear_f(hiddens[-1], wf['lm_head'], counter)
        probs = softmax_f(logits)
        next_tok = max(range(vocab_size), key=lambda i: probs[i])
        tokens.append(next_tok)
        muls_per_step.append(counter[0])

    return tokens[1:], muls_per_step


# === INFERENCE WITH KV CACHE ===
# At each step, compute Q/K/V for ONLY the new token. Append K and V to the cache.
# Attention: Q_new attends to all cached K (0..t), V (0..t).
# Work per step: C_proj + t * C_attn ~ O(t). Total for T steps: O(T^2) — one order better.
# The insight: K and V projections for past tokens never change in autoregressive decoding.
# Recomputing them is pure waste — the KV cache is memoization of linear projections.

def generate_with_cache(
    prompt_tok: int, wf: dict[str, list[list[float]]],
    vocab_size: int, gen_len: int,
) -> tuple[list[int], list[int], list[int]]:
    """Generate tokens WITH KV cache. Returns (tokens, muls_per_step, cache_sizes)."""
    tokens: list[int] = []
    muls_per_step: list[int] = []
    cache_sizes: list[int] = []

    # KV cache: stores projected K and V vectors for each layer and position.
    # Shape: kv_cache[layer] = {'k': list of vectors, 'v': list of vectors}
    kv_cache: list[dict[str, list[list[float]]]] = [
        {'k': [], 'v': []} for _ in range(N_LAYER)
    ]

    current_tok = prompt_tok
    for step in range(gen_len):
        counter = [0]
        pos = step

        # Embed only the NEW token — previous embeddings don't need recomputation
        x = [wf['wte'][current_tok][j] + wf['wpe'][pos][j] for j in range(N_EMBD)]
        x = rmsnorm_f(x)

        for li in range(N_LAYER):
            x_res = x[:]
            x = rmsnorm_f(x)

            # Project ONLY the new token — this is where the cache saves work.
            # Without cache: project all t tokens. With cache: project 1 token.
            q = linear_f(x, wf[f'l{li}.wq'], counter)
            k = linear_f(x, wf[f'l{li}.wk'], counter)
            v = linear_f(x, wf[f'l{li}.wv'], counter)

            # Append new K, V to cache (the cache grows by one entry per step)
            kv_cache[li]['k'].append(k)
            kv_cache[li]['v'].append(v)

            # Attention: Q from new token attends to ALL cached K/V
            head_cat: list[float] = []
            cached_len = len(kv_cache[li]['k'])
            for h in range(N_HEAD):
                hs = h * HEAD_DIM
                q_h = q[hs:hs + HEAD_DIM]
                scores: list[float] = []
                for t in range(cached_len):
                    k_h = kv_cache[li]['k'][t][hs:hs + HEAD_DIM]
                    dot = sum(q_h[j] * k_h[j] for j in range(HEAD_DIM))
                    counter[0] += HEAD_DIM
                    scores.append(dot / (HEAD_DIM ** 0.5))
                weights = softmax_f(scores)
                for j in range(HEAD_DIM):
                    val = 0.0
                    for t in range(cached_len):
                        val += weights[t] * kv_cache[li]['v'][t][hs + j]
                        counter[0] += 1
                    head_cat.append(val)

            x = linear_f(head_cat, wf[f'l{li}.wo'], counter)
            x = [a + b for a, b in zip(x, x_res)]
            x_res = x[:]
            x = rmsnorm_f(x)
            x = linear_f(x, wf[f'l{li}.fc1'], counter)
            x = [max(0.0, v) for v in x]
            x = linear_f(x, wf[f'l{li}.fc2'], counter)
            x = [a + b for a, b in zip(x, x_res)]

        logits = linear_f(x, wf['lm_head'], counter)
        probs = softmax_f(logits)
        next_tok = max(range(vocab_size), key=lambda i: probs[i])
        tokens.append(next_tok)
        current_tok = next_tok
        muls_per_step.append(counter[0])

        # Cache memory: 2 (K+V) * n_layer * n_embd floats per cached position
        total_cached_floats = 2 * N_LAYER * N_EMBD * len(kv_cache[0]['k'])
        cache_sizes.append(total_cached_floats)

    return tokens, muls_per_step, cache_sizes


# === PAGED ATTENTION SIMULATION ===
# Production systems (vLLM) can't pre-allocate contiguous memory for every sequence's
# KV cache because sequence lengths are unknown and variable. Paged attention borrows
# the OS virtual memory idea: allocate fixed-size blocks on demand, map logical positions
# to physical blocks through a page table. This eliminates fragmentation from over-
# allocation and enables sharing physical blocks across sequences (e.g., shared prefixes).

def simulate_paged_attention(seq_len: int, block_size: int) -> None:
    """Demonstrate how paged attention allocates and maps cache blocks."""
    print(f"\n=== Paged Attention Simulation ===")
    print(f"Block size: {block_size} positions | Sequence length: {seq_len}")
    print(f"Each block holds {block_size} positions of KV data\n")

    # Page table: maps logical block index -> physical block index
    # Physical blocks are allocated from a pool as needed
    page_table: list[int] = []
    next_physical_block = 0

    print("Allocation trace:")
    for pos in range(seq_len):
        logical_block = pos // block_size
        slot_in_block = pos % block_size

        # Allocate a new physical block when we enter a new logical block
        if logical_block >= len(page_table):
            page_table.append(next_physical_block)
            print(f"  Position {pos:>2}: new block needed -> "
                  f"logical block {logical_block} -> physical block {next_physical_block}")
            next_physical_block += 1
        else:
            print(f"  Position {pos:>2}: slot {slot_in_block} in "
                  f"logical block {logical_block} (physical {page_table[logical_block]})")

    print(f"\nPage table (logical -> physical):")
    for i, phys in enumerate(page_table):
        start = i * block_size
        end = min(start + block_size - 1, seq_len - 1)
        status = "FULL" if (i + 1) * block_size <= seq_len else f"{seq_len - i * block_size}/{block_size}"
        print(f"  Logical block {i} -> Physical block {phys} "
              f"[positions {start}-{end}] {status}")

    # Signpost: in production, physical blocks are shared across sequences. Two prompts
    # starting with the same system message reuse the same physical blocks for the shared
    # prefix — vLLM's copy-on-write avoids duplicating KV data. We simulate single-
    # sequence allocation; the multi-sequence sharing is the real memory win at scale.
    wasted = len(page_table) * block_size - seq_len
    print(f"\nBlocks allocated: {len(page_table)} ({len(page_table) * block_size} slots)")
    print(f"Slots used: {seq_len} | Wasted: {wasted} "
          f"({100 * wasted / (len(page_table) * block_size):.0f}% internal fragmentation)")


# === MAIN ===

if __name__ == "__main__":
    # -- Load data and build vocabulary --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    VOCAB_SIZE = len(unique_chars) + 1
    print(f"Loaded {len(docs)} documents, vocab size: {VOCAB_SIZE}")

    # -- Train tiny model --
    # Training is not the focus — we just need a model that produces deterministic,
    # non-random outputs so the with-cache and without-cache comparison is meaningful.
    print(f"\nTraining tiny model ({NUM_STEPS} steps)...")
    params: dict[str, list[list[Value]]] = {}
    params['wte'] = make_matrix(VOCAB_SIZE, N_EMBD)
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)
    for li in range(N_LAYER):
        params[f'l{li}.wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'l{li}.wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'l{li}.wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'l{li}.wo'] = make_matrix(N_EMBD, N_EMBD)
        params[f'l{li}.fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'l{li}.fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)
    params['lm_head'] = make_matrix(VOCAB_SIZE, N_EMBD)

    param_list = [p for w in params.values() for row in w for p in row]
    m_state = [0.0] * len(param_list)
    v_state = [0.0] * len(param_list)

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        seq_len = min(BLOCK_SIZE, len(tokens) - 1)
        keys_t = [[] for _ in range(N_LAYER)]
        vals_t = [[] for _ in range(N_LAYER)]
        losses: list[Value] = []
        for pos in range(seq_len):
            logits = gpt_forward_train(tokens[pos], pos, keys_t, vals_t, params)
            probs = softmax_v(logits)
            losses.append(-safe_log(probs[tokens[pos + 1]]))
        loss = (1.0 / seq_len) * sum(losses)
        loss.backward()

        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
        for i, p in enumerate(param_list):
            m_state[i] = BETA1 * m_state[i] + (1 - BETA1) * p.grad
            v_state[i] = BETA2 * v_state[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_state[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_state[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS} | loss: {loss.data:.4f}")

    print(f"Training complete. Final loss: {loss.data:.4f}")

    # -- Extract weights as plain floats --
    wf: dict[str, list[list[float]]] = {k: extract(v) for k, v in params.items()}

    # -- Run both inference methods on the same prompt --
    prompt_tok = BOS
    print(f"\n=== KV-Cache Comparison ===")
    print(f"Generating {GEN_LEN}-character sequence from BOS token\n")

    t0 = time.time()
    toks_no_cache, muls_no = generate_no_cache(prompt_tok, wf, VOCAB_SIZE, GEN_LEN)
    time_no = time.time() - t0

    t0 = time.time()
    toks_cached, muls_yes, cache_sizes = generate_with_cache(prompt_tok, wf, VOCAB_SIZE, GEN_LEN)
    time_cached = time.time() - t0

    # -- Verify identical outputs --
    # Both methods compute the exact same mathematical function. The KV cache is a
    # computational shortcut, not an approximation — outputs must match exactly.
    assert toks_no_cache == toks_cached, (
        f"Output mismatch: no-cache produced {toks_no_cache}, cache produced {toks_cached}"
    )

    # -- Print step-by-step comparison --
    name_no = [unique_chars[t] if t != BOS else '.' for t in toks_no_cache]
    name_yes = [unique_chars[t] if t != BOS else '.' for t in toks_cached]

    header = f"{'Step':>4}  {'No Cache (muls)':>16}  {'With Cache (muls)':>18}  {'Speedup':>8}  {'Match':>5}"
    print(header)
    print("-" * len(header))
    for i in range(GEN_LEN):
        ratio = muls_no[i] / muls_yes[i] if muls_yes[i] > 0 else 0
        match = "yes" if toks_no_cache[i] == toks_cached[i] else "NO"
        print(f"{i + 1:>4}  {muls_no[i]:>16,}  {muls_yes[i]:>18,}  {ratio:>7.1f}x  {match:>5}")

    total_no = sum(muls_no)
    total_yes = sum(muls_yes)
    overall_ratio = total_no / total_yes if total_yes > 0 else 0
    print(f"\nTotal multiplies -- No cache: {total_no:,} | With cache: {total_yes:,} | "
          f"Ratio: {overall_ratio:.1f}x")
    print(f"Wall time -- No cache: {time_no:.3f}s | With cache: {time_cached:.3f}s")

    generated_str = ''.join(name_yes)
    print(f"\nGenerated: \"{generated_str}\" (both methods identical)")

    # -- Memory growth analysis --
    # KV cache stores 2 vectors (K and V) per layer per position, each of size n_embd.
    # Memory growth is strictly linear in sequence length — no quadratic blowup.
    # This linear growth is WHY long-context models (100K+ tokens) are memory-bound:
    # at d_model=4096, 40 layers, 100K tokens, the cache is ~32GB in float16.
    floats_per_pos = 2 * N_LAYER * N_EMBD
    print(f"\n=== Memory Growth ===")
    print(f"{'Position':>8}   {'Cache Size (floats)':>20}   {'Cache Size (bytes, float32)':>28}")
    print("-" * 62)
    for i in range(GEN_LEN):
        n_floats = cache_sizes[i]
        n_bytes = n_floats * 4
        print(f"{i + 1:>8}   {n_floats:>20,}   {n_bytes:>28,}")

    print(f"\nGrowth: linear O(n) -- {floats_per_pos} floats per position "
          f"(2 * {N_LAYER} layer * {N_EMBD} embd)")
    print(f"Signpost: LLaMA-2 70B with 80 layers, 8192 embd, 4K context = "
          f"~5.2 GB KV cache in float16.")
    print(f"This is why KV cache memory, not compute, is the bottleneck for long sequences.")

    # -- Paged attention --
    simulate_paged_attention(GEN_LEN, PAGE_BLOCK_SIZE)
