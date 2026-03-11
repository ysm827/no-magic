"""
The other half of the transformer — how BERT learns bidirectional representations by predicting
masked tokens, and why encoders complement decoders.
"""
# Reference: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for
# Language Understanding" (2018). https://arxiv.org/abs/1810.04805
# Architecture mirrors microgpt.py (N_EMBD=16, N_HEAD=4, N_LAYER=1) with one critical
# difference: no causal mask. Every token attends to every other token, enabling the model
# to use BOTH left and right context when predicting masked positions.

# === TRADEOFFS ===
# + Bidirectional context lets every token attend to both left and right neighbors
# + Excels at classification, NER, and other understanding tasks (not generation)
# + Pre-train once, fine-tune cheaply on many downstream tasks
# - Cannot generate text autoregressively (no causal structure)
# - Masked-token training is sample-inefficient (only ~15-25% of tokens contribute loss)
# - Fixed context window with no streaming capability
# WHEN TO USE: Text classification, named entity recognition, sentence similarity,
#   and any task where understanding matters more than generation.
# WHEN NOT TO: Open-ended text generation, dialogue systems, or any task
#   requiring autoregressive output (use GPT-style decoder instead).

from __future__ import annotations

import math
import os
import random
import urllib.request

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture — deliberately identical to microgpt for controlled comparison.
# The ONLY architectural difference is causal vs. bidirectional attention.
N_EMBD = 16         # embedding dimension (d_model, same as microgpt)
N_HEAD = 4          # number of attention heads
N_LAYER = 1         # transformer blocks
BLOCK_SIZE = 16     # maximum sequence length (names are short, 16 is sufficient)
HEAD_DIM = N_EMBD // N_HEAD  # 4 dimensions per head

# Training parameters
LEARNING_RATE = 0.02
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8
NUM_STEPS = 3000

# Signpost: BERT needs more steps than microgpt (1000) because only ~25% of tokens
# contribute to the loss per step. With names averaging ~6 chars, that's 1-2 masked
# tokens per example — the model gets far fewer gradient signals per step.

# MLM parameters
MASK_PROB = 0.25    # fraction of tokens to mask per training example
# Signpost: BERT's original uses 15% masking with an 80/10/10 split (80% [MASK], 10% random
# token, 10% unchanged). We use 25% pure [MASK] — higher than standard to give the tiny model
# more gradient signals per step. At toy scale, 15% of a 6-char name is often just 1 token.

# Data parameters
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "names.txt"

# Signpost: Same ~4,200 parameters as microgpt. Production BERT-base has 110M parameters
# across 12 layers. The architecture scales identically — this toy version proves the
# mechanism works before you burn a GPU cluster.


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
    its local derivative (∂out/∂input) as a closure, then backward() replays
    the computation graph in reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

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
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """Compute gradients via reverse-mode automatic differentiation."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# No per-script extensions needed — BERT uses the same operations as GPT
# (attention, softmax, cross-entropy). The difference is architectural
# (bidirectional attention), not operational.
# See docs/autograd-interface.md for the full specification.


# === PARAMETER INITIALIZATION ===

def make_matrix(nrows: int, ncols: int, std: float = 0.08) -> list[list[Value]]:
    """Initialize a weight matrix with Gaussian noise."""
    return [[Value(random.gauss(0, std)) for _ in range(ncols)] for _ in range(nrows)]


def init_parameters(vocab_size: int) -> dict:
    """Initialize all model parameters: embeddings, attention, MLP, and MLM head.

    Identical to microgpt's parameter structure EXCEPT:
    - No language model head projecting to vocab (that's microgpt's autoregressive head)
    - Instead, an MLM head that projects masked positions to vocab predictions
    - In practice, these are the same matrix shape — the difference is conceptual:
      microgpt predicts the NEXT token, BERT predicts the MASKED token.
    """
    params = {}

    # Token embeddings: [vocab_size, n_embd]
    # vocab_size includes regular chars + [BOS] + [MASK] special tokens
    params['wte'] = make_matrix(vocab_size, N_EMBD)

    # Position embeddings: [block_size, n_embd]
    # Same as GPT — encodes absolute position in the sequence
    params['wpe'] = make_matrix(BLOCK_SIZE, N_EMBD)

    # Transformer layer weights (identical structure to microgpt)
    for layer_idx in range(N_LAYER):
        params[f'layer{layer_idx}.attn_wq'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wk'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wv'] = make_matrix(N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.attn_wo'] = make_matrix(N_EMBD, N_EMBD)

        params[f'layer{layer_idx}.mlp_fc1'] = make_matrix(4 * N_EMBD, N_EMBD)
        params[f'layer{layer_idx}.mlp_fc2'] = make_matrix(N_EMBD, 4 * N_EMBD)

    # MLM prediction head: projects hidden states at masked positions to vocabulary logits
    # Same shape as microgpt's lm_head, but applied only at masked positions
    params['mlm_head'] = make_matrix(vocab_size, N_EMBD)

    return params


# === CORE OPERATIONS ===

def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    """Matrix-vector multiplication: y = W @ x (no bias)."""
    return [sum(w_row[j] * x[j] for j in range(len(x))) for w_row in w]


def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: converts logits to probabilities.

    Subtract max before exp to prevent overflow. softmax is translation-invariant
    so this doesn't change the result: softmax(x - c) = softmax(x) for any c.
    """
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def rmsnorm(x: list[Value]) -> list[Value]:
    """Root Mean Square normalization: scale vector to unit RMS magnitude."""
    mean_sq = sum(xi * xi for xi in x) / len(x)
    scale = (mean_sq + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def safe_log(prob: Value) -> Value:
    """Clipped logarithm — prevents log(0) from breaking gradient computation."""
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


# === BERT FORWARD PASS ===

def bert_forward(
    token_ids: list[int],
    params: dict,
) -> list[list[Value]]:
    """Full-sequence forward pass through the BERT encoder.

    THE CRITICAL DIFFERENCE FROM MICROGPT:
    microgpt processes tokens one-at-a-time with an incremental KV cache, building
    causal attention implicitly (each token only sees past tokens because future keys
    don't exist yet). BERT processes ALL tokens simultaneously and computes attention
    over the FULL sequence — every token attends to every other token, including those
    that come after it.

    This bidirectional attention is why BERT excels at understanding tasks (classification,
    NER, question answering) — it can use full context to understand each token. But it
    can't generate text autoregressively because seeing the full sequence during training
    means it has no concept of "left-to-right" generation.

    Args:
        token_ids: Full sequence of integers [seq_len], may include [MASK] tokens
        params: Model weight matrices

    Returns:
        Hidden states for ALL positions, shape [seq_len][n_embd]
    """
    seq_len = len(token_ids)

    # -- Embedding layer --
    # Same as GPT: token embedding + position embedding
    x_seq = []
    for pos, token_id in enumerate(token_ids):
        tok_emb = params['wte'][token_id]
        pos_emb = params['wpe'][pos]
        x_seq.append([t + p for t, p in zip(tok_emb, pos_emb)])

    # Normalize each position's embedding
    x_seq = [rmsnorm(x) for x in x_seq]

    # -- Transformer layers --
    for layer_idx in range(N_LAYER):
        residuals = [list(x) for x in x_seq]

        # Pre-norm
        x_normed = [rmsnorm(x) for x in x_seq]

        # Project ALL positions to Q, K, V simultaneously
        # This is where BERT diverges from microgpt's incremental approach:
        # we have the full sequence available, so we project everything at once.
        all_q = [linear(x, params[f'layer{layer_idx}.attn_wq']) for x in x_normed]
        all_k = [linear(x, params[f'layer{layer_idx}.attn_wk']) for x in x_normed]
        all_v = [linear(x, params[f'layer{layer_idx}.attn_wv']) for x in x_normed]

        # -- Bidirectional multi-head self-attention --
        # For each position, attend to ALL positions (not just past ones)
        attn_output = []
        for pos_i in range(seq_len):
            head_outputs = []
            for head in range(N_HEAD):
                head_start = head * HEAD_DIM

                # This position's query
                q_head = all_q[pos_i][head_start : head_start + HEAD_DIM]

                # ALL positions' keys and values — the bidirectional part
                # In microgpt, this loop only covers positions 0..pos_i (causal).
                # In BERT, it covers ALL positions 0..seq_len-1.
                k_heads = [
                    all_k[pos_j][head_start : head_start + HEAD_DIM]
                    for pos_j in range(seq_len)
                ]
                v_heads = [
                    all_v[pos_j][head_start : head_start + HEAD_DIM]
                    for pos_j in range(seq_len)
                ]

                # Attention scores: score(q_i, k_j) = (q_i · k_j) / √d_head
                # Every position j is included — no masking of future positions
                attn_logits = [
                    sum(q_head[d] * k_heads[j][d] for d in range(HEAD_DIM))
                    / (HEAD_DIM ** 0.5)
                    for j in range(seq_len)
                ]

                attn_weights = softmax(attn_logits)

                # Weighted sum of ALL value vectors
                head_out = [
                    sum(attn_weights[j] * v_heads[j][d] for j in range(seq_len))
                    for d in range(HEAD_DIM)
                ]
                head_outputs.extend(head_out)

            attn_output.append(head_outputs)

        # Project attention output and add residual
        x_seq = [
            [a + r for a, r in zip(
                linear(attn_out, params[f'layer{layer_idx}.attn_wo']),
                residual
            )]
            for attn_out, residual in zip(attn_output, residuals)
        ]

        # MLP with residual connection
        residuals = [list(x) for x in x_seq]
        x_seq_normed = [rmsnorm(x) for x in x_seq]
        x_seq = [
            [mlp_out + r for mlp_out, r in zip(
                linear(
                    [xi.relu() for xi in linear(x, params[f'layer{layer_idx}.mlp_fc1'])],
                    params[f'layer{layer_idx}.mlp_fc2']
                ),
                residual
            )]
            for x, residual in zip(x_seq_normed, residuals)
        ]

    return x_seq


# === MASKING STRATEGY ===

def apply_masking(
    token_ids: list[int],
    mask_token_id: int,
    mask_prob: float,
) -> tuple[list[int], list[int]]:
    """Replace a fraction of tokens with [MASK] and return masked positions.

    BERT's training signal comes ONLY from predicting the original tokens at masked
    positions. Unmasked positions contribute no loss — the model must learn good
    representations for ALL tokens to predict the few that are masked.

    This is the key insight: by forcing the model to predict missing tokens using
    bidirectional context, BERT learns contextual representations that capture
    meaning from both directions.

    Returns:
        masked_ids: token sequence with some positions replaced by mask_token_id
        masked_positions: indices where masking was applied (for loss computation)
    """
    masked_ids = list(token_ids)
    masked_positions = []

    # Only mask content tokens (positions 1..n-2), never the BOS delimiters.
    # BOS tokens are structural markers, not content — masking them would teach
    # the model to predict delimiters instead of learning character patterns.
    for i in range(1, len(token_ids) - 1):
        if random.random() < mask_prob:
            masked_ids[i] = mask_token_id
            masked_positions.append(i)

    # Ensure at least one content position is masked (otherwise no training signal)
    if not masked_positions and len(token_ids) > 2:
        pos = random.randint(1, len(token_ids) - 2)
        masked_ids[pos] = mask_token_id
        masked_positions.append(pos)

    return masked_ids, masked_positions


# === TRAINING LOOP ===

if __name__ == "__main__":
    # -- Prepare vocabulary and data --
    print("Loading data...")
    docs = load_data(DATA_URL, DATA_FILE)
    random.shuffle(docs)

    # Build vocabulary from unique characters
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)          # beginning/end-of-sequence token
    MASK_TOKEN = len(unique_chars) + 1  # [MASK] special token
    VOCAB_SIZE = len(unique_chars) + 2  # chars + BOS + MASK

    print(f"Loaded {len(docs)} documents")
    print(f"Vocabulary: {VOCAB_SIZE} tokens ({len(unique_chars)} chars + [BOS] + [MASK])")

    # Initialize parameters
    params = init_parameters(VOCAB_SIZE)

    # Flatten parameters for optimizer
    param_list = [p for matrix in params.values() for row in matrix for p in row]
    print(f"Parameters: {len(param_list):,}")

    # Signpost: Same parameter count as microgpt (~4,200, plus one extra embedding for [MASK]).
    # The only structural difference is the lack of causal masking. This controlled
    # comparison isolates the effect of bidirectional vs. unidirectional attention.

    # -- Initialize Adam optimizer state --
    m_adam = [0.0] * len(param_list)
    v_adam = [0.0] * len(param_list)

    # -- Training --
    print("\nTraining BERT (masked language modeling)...")
    print("=" * 60)

    for step in range(NUM_STEPS):
        doc = docs[step % len(docs)]

        # Tokenize: [BOS] + characters + [BOS]
        original_tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]

        # Truncate to block_size
        if len(original_tokens) > BLOCK_SIZE:
            original_tokens = original_tokens[:BLOCK_SIZE]

        seq_len = len(original_tokens)

        # Apply masking — replace ~15% of tokens with [MASK]
        # The model sees the masked sequence and must predict the original tokens
        # at masked positions. This is the MLM pretraining objective.
        masked_tokens, masked_positions = apply_masking(
            original_tokens, MASK_TOKEN, MASK_PROB
        )

        # Forward pass: process the ENTIRE masked sequence at once
        # Every position gets a hidden state informed by the full bidirectional context
        hidden_states = bert_forward(masked_tokens, params)

        # Compute loss ONLY at masked positions
        # This is the core MLM loss: -log P(original_token | masked_context)
        # Unmasked positions don't contribute to the loss — but they DO contribute
        # to the attention computation, providing context for masked predictions.
        losses = []
        for pos in masked_positions:
            # Project hidden state at masked position to vocabulary logits
            logits = linear(hidden_states[pos], params['mlm_head'])
            probs = softmax(logits)

            # Cross-entropy: how well does the model predict the original token?
            target = original_tokens[pos]
            loss_t = -safe_log(probs[target])
            losses.append(loss_t)

        if not losses:
            continue

        loss = (1.0 / len(losses)) * sum(losses)

        # -- Backward pass --
        loss.backward()

        # -- Adam optimizer step --
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)

        for i, param in enumerate(param_list):
            m_adam[i] = BETA1 * m_adam[i] + (1 - BETA1) * param.grad
            v_adam[i] = BETA2 * v_adam[i] + (1 - BETA2) * param.grad ** 2

            m_hat = m_adam[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_adam[i] / (1 - BETA2 ** (step + 1))

            param.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            param.grad = 0.0

        # Print progress
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS:>4} | loss: {loss.data:.4f}"
                  f" | masked {len(masked_positions)}/{seq_len} tokens")

    print(f"\nTraining complete. Final loss: {loss.data:.4f}")

    # === INFERENCE: FILL-IN-THE-BLANK ===
    # BERT's natural inference mode: given a sequence with [MASK] tokens,
    # predict what goes in the blanks using full bidirectional context.
    #
    # This is fundamentally different from microgpt's inference:
    # - microgpt generates LEFT-TO-RIGHT, one token at a time
    # - BERT fills in MASKED POSITIONS, using context from both sides
    #
    # You can't do open-ended generation with BERT because it needs a fixed-length
    # input with specific [MASK] positions. It's a "fill-in-the-blank" model,
    # not a "continue-the-sentence" model.

    print("\n" + "=" * 60)
    print("INFERENCE: Fill-in-the-blank predictions")
    print("=" * 60)

    # -- Test 1: Evaluate on training data samples --
    # The pedagogical goal is showing the model learns character patterns in context,
    # not memorizing specific names. We evaluate on a sample of training names.
    print("\n--- Masked prediction on training data ---")

    eval_correct = 0
    eval_total = 0
    eval_top3_correct = 0
    num_eval = 50

    for eval_idx in range(num_eval):
        name = docs[eval_idx]
        tokens = [BOS] + [unique_chars.index(ch) for ch in name] + [BOS]
        if len(tokens) > BLOCK_SIZE:
            tokens = tokens[:BLOCK_SIZE]
        if len(name) < 3:
            continue

        # Mask a middle character — requires bidirectional context
        mask_char_pos = len(name) // 2
        mask_seq_pos = mask_char_pos + 1
        original_token = tokens[mask_seq_pos]

        masked_tokens = list(tokens)
        masked_tokens[mask_seq_pos] = MASK_TOKEN

        hidden_states = bert_forward(masked_tokens, params)
        logits = linear(hidden_states[mask_seq_pos], params['mlm_head'])
        probs = softmax(logits)

        prob_data = [(i, p.data) for i, p in enumerate(probs) if i < len(unique_chars)]
        prob_data.sort(key=lambda x: x[1], reverse=True)

        if prob_data[0][0] == original_token:
            eval_correct += 1
        if original_token in [p[0] for p in prob_data[:3]]:
            eval_top3_correct += 1
        eval_total += 1

    print(f"  Top-1 accuracy: {eval_correct}/{eval_total}"
          f" ({eval_correct / max(eval_total, 1):.1%})")
    print(f"  Top-3 accuracy: {eval_top3_correct}/{eval_total}"
          f" ({eval_top3_correct / max(eval_total, 1):.1%})")

    # -- Test 2: Demonstrate bidirectional context with specific examples --
    # Show that changing surrounding characters changes the prediction —
    # proof the model uses context from BOTH sides, not just left context.
    print("\n--- Bidirectional context demonstration ---")
    print("  Same [MASK] position, different surrounding context:\n")

    # Construct pairs where the same position is masked but context differs
    context_pairs = [
        ("ma_y", "mary"),    # common name patterns
        ("ma_k", "mark"),
        ("sa_a", "sara"),
        ("sa_m", "sam" + "m"),  # padding
        ("an_a", "anna"),
        ("an_e", "ane" + "e"),
    ]

    for display, name in context_pairs:
        tokens = [BOS] + [unique_chars.index(ch) for ch in name] + [BOS]
        if len(tokens) > BLOCK_SIZE:
            tokens = tokens[:BLOCK_SIZE]

        # Find the mask position (the underscore position in display)
        mask_char_pos = display.index('_')
        mask_seq_pos = mask_char_pos + 1

        masked_tokens = list(tokens)
        masked_tokens[mask_seq_pos] = MASK_TOKEN

        hidden_states = bert_forward(masked_tokens, params)
        logits = linear(hidden_states[mask_seq_pos], params['mlm_head'])
        probs = softmax(logits)

        prob_data = [(i, p.data) for i, p in enumerate(probs) if i < len(unique_chars)]
        prob_data.sort(key=lambda x: x[1], reverse=True)
        top3 = prob_data[:3]

        top3_str = ", ".join(
            f"'{unique_chars[idx]}' ({prob:.3f})" for idx, prob in top3
        )
        print(f"  {display:>8} → top-3: {top3_str}")

    # === COMPARISON WITH MICROGPT ===
    # Summarize the key architectural and behavioral differences

    print("\n" + "=" * 60)
    print("BERT vs. GPT COMPARISON")
    print("=" * 60)
    print(f"""
  Property              | BERT (this script)      | GPT (microgpt.py)
  ──────────────────────┼─────────────────────────┼──────────────────────────
  Attention direction   | Bidirectional (full)    | Unidirectional (causal)
  Training objective    | Masked LM (fill-blank)  | Next-token prediction
  Masking               | 25% random [MASK]       | Causal mask (future hidden)
  Inference mode        | Fill-in-the-blank       | Left-to-right generation
  Good at               | Understanding, NLU      | Generation, completion
  Token processing      | All tokens at once      | One token at a time
  KV cache needed?      | No (full sequence)      | Yes (incremental)
  Parameters            | ~{len(param_list):,}             | ~4,200
  Dimensions            | d={N_EMBD}, h={N_HEAD}, L={N_LAYER}         | d=16, h=4, L=1

  Key insight: The ONLY difference is causal masking. Same attention mechanism,
  same projections, same MLP — but removing the causal constraint transforms a
  text generator into a text understander. This is why modern systems often use
  BOTH: an encoder (BERT-like) for understanding and a decoder (GPT-like) for
  generation (e.g., T5, the original Transformer).
""")
