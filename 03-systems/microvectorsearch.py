"""
Vector search from first principles: exact brute-force nearest neighbors vs approximate
search via locality-sensitive hashing — demonstrating the accuracy-speed tradeoff that
underpins every retrieval-augmented generation system.
"""
# Reference: Indyk & Motwani, "Approximate Nearest Neighbors: Towards Removing the
# Curse of Dimensionality" (1998). Also: Charikar, "Similarity Estimation Techniques
# from Rounding Algorithms" (2002) — the random hyperplane LSH family.

# === TRADEOFFS ===
# + Exact search guarantees finding the true nearest neighbor every time
# + LSH provides sublinear query time O(n^(1/c)) for c-approximate neighbors
# + No training required — index construction is deterministic
# - Brute-force is O(n*d) per query — prohibitive for million-scale datasets
# - LSH accuracy depends heavily on hash function count and table count
# - Curse of dimensionality: distance concentration makes all methods degrade
# WHEN TO USE: Retrieval-augmented generation, recommendation systems,
#   semantic search, duplicate detection, clustering initialization.
# WHEN NOT TO: When exact results are required with strict latency
#   constraints on very large datasets (use specialized libraries like FAISS).

from __future__ import annotations

import math
import random
import time
from collections import defaultdict

random.seed(42)

# === CONSTANTS AND HYPERPARAMETERS ===

VECTOR_DIM = 64           # embedding dimensionality
NUM_VECTORS = 5000        # database size — large enough to show speedup
NUM_QUERIES = 50          # queries to average over for timing
TOP_K = 10                # number of nearest neighbors to retrieve

# LSH parameters — these are the primary accuracy-speed knobs.
# More tables = higher recall (each table is an independent chance to find a neighbor).
# More hash bits = more precise buckets but sparser (fewer collisions per bucket).
NUM_TABLES = 20           # L: independent hash tables (recall scales as 1-(1-p^k)^L)
NUM_HASH_BITS = 5         # k: bits per hash (precision per table, each bit is one hyperplane)

# Signpost: production vector search systems (Pinecone, Weaviate, Qdrant) use HNSW
# (hierarchical navigable small world graphs) or IVF (inverted file index) rather than
# LSH. LSH has strong theoretical guarantees but HNSW dominates in practice due to
# better recall-speed tradeoffs. We use LSH here because it is the simplest approximate
# method with provable properties.

# Cluster parameters for synthetic data generation
NUM_CLUSTERS = 20         # semantic clusters (simulating word embedding neighborhoods)
CLUSTER_SPREAD = 0.3      # controls intra-cluster variance


# === DATA GENERATION ===
# Synthetic word embeddings: clustered random vectors that mimic the structure of real
# embeddings where semantically related words form neighborhoods in vector space.
# This clustering matters — uniform random vectors in high dimensions are all roughly
# equidistant (curse of dimensionality), making nearest-neighbor search trivial to
# approximate but meaningless. Clustered data creates the "needle in a haystack"
# structure where search quality actually matters.

def generate_clustered_vectors(
    num_vectors: int,
    dim: int,
    num_clusters: int,
    spread: float,
) -> list[list[float]]:
    """Generate vectors clustered around random centroids.

    Each centroid is a random unit vector; points are sampled by adding Gaussian
    noise scaled by `spread`. Lower spread = tighter clusters = easier search
    (neighbors are more distinct from non-neighbors)."""
    # Generate cluster centroids on the unit sphere
    centroids: list[list[float]] = []
    for _ in range(num_clusters):
        raw = [random.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw))
        centroids.append([x / norm for x in raw])

    vectors: list[list[float]] = []
    for i in range(num_vectors):
        # Round-robin cluster assignment ensures balanced clusters
        centroid = centroids[i % num_clusters]
        # Gaussian perturbation around the centroid
        vec = [c + random.gauss(0.0, spread) for c in centroid]
        vectors.append(vec)
    return vectors


def generate_query_vectors(
    database: list[list[float]],
    num_queries: int,
    noise_scale: float = 0.1,
) -> list[list[float]]:
    """Generate queries by perturbing randomly chosen database vectors.

    Perturbation ensures queries aren't exact database entries (trivial to find)
    but are close enough that meaningful nearest neighbors exist."""
    queries: list[list[float]] = []
    for _ in range(num_queries):
        base = database[random.randint(0, len(database) - 1)]
        query = [x + random.gauss(0.0, noise_scale) for x in base]
        queries.append(query)
    return queries


# === DISTANCE FUNCTIONS ===
# Three standard distance/similarity measures. Each captures a different notion of
# "closeness" and is appropriate for different use cases.

def dot_product(a: list[float], b: list[float]) -> float:
    """Inner product: sum(a_i * b_i). The fundamental operation underlying all three
    measures. O(d) time, no way around it — this is the irreducible cost of comparing
    two d-dimensional vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """L2 distance: ||a - b||_2 = sqrt(sum((a_i - b_i)^2)).

    Sensitive to vector magnitude — a vector [2,0] is far from [1,0] even though
    they point in the same direction. Use when magnitude carries meaning (e.g.,
    TF-IDF vectors where magnitude reflects document length)."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(theta) = (a . b) / (||a|| * ||b||).

    Projects vectors onto the unit sphere before comparing — only direction matters,
    not magnitude. This is why cosine similarity dominates in NLP: a sentence embedding
    scaled by 2x should still mean the same thing. Range: [-1, 1] where 1 = identical
    direction, 0 = orthogonal, -1 = opposite."""
    # Math-to-code: cos(θ) = Σ(aᵢbᵢ) / (√Σaᵢ² · √Σbᵢ²)
    ab = dot_product(a, b)
    norm_a = math.sqrt(dot_product(a, a))
    norm_b = math.sqrt(dot_product(b, b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return ab / (norm_a * norm_b)


# === BRUTE-FORCE EXACT SEARCH ===
# The baseline: compute similarity to every vector in the database, return top-k.
# O(n*d) per query with no way to avoid it — every vector must be touched.
# This is the method that all approximate algorithms benchmark against.

def brute_force_search(
    query: list[float],
    database: list[list[float]],
    top_k: int,
) -> list[tuple[int, float]]:
    """Return the top_k most similar vectors by cosine similarity.

    Returns list of (index, similarity) tuples sorted by descending similarity.
    This is the ground truth that LSH approximates."""
    similarities: list[tuple[int, float]] = []
    for idx, vec in enumerate(database):
        sim = cosine_similarity(query, vec)
        similarities.append((idx, sim))
    # Full sort is O(n log n); could use a heap for O(n log k) but clarity wins here.
    # Production systems use partial sort / selection algorithms.
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# === LSH INDEX ===
# Random hyperplane LSH for cosine similarity (SimHash).
#
# Core insight: pick a random hyperplane through the origin. Two vectors on the same
# side of the hyperplane have angular distance < 90°; on opposite sides, > 90°.
# The probability of two vectors hashing to the same bit equals:
#
#   Pr[h(a) = h(b)] = 1 - θ(a,b)/π
#
# where θ(a,b) is the angle between them. Similar vectors (small θ) agree on most
# bits; dissimilar vectors (large θ) disagree.
#
# This is a locality-sensitive hash because it satisfies:
#   - If sim(a,b) ≥ s₁, then Pr[h(a)=h(b)] ≥ p₁  (similar → likely same bucket)
#   - If sim(a,b) ≤ s₂, then Pr[h(a)=h(b)] ≤ p₂  (dissimilar → likely different bucket)
#
# Connection to Johnson-Lindenstrauss: random projections preserve distances in
# expectation. Each hyperplane is a 1-bit random projection. Multiple bits give a
# low-distortion embedding from R^d to {0,1}^k, preserving angular relationships.

def generate_random_hyperplanes(
    num_planes: int,
    dim: int,
) -> list[list[float]]:
    """Generate random unit vectors as hash hyperplanes.

    Each hyperplane partitions R^d into two half-spaces. The hash bit for vector v
    is sign(dot(v, plane)): 1 if v is on the positive side, 0 if negative.
    Normal distribution ensures uniform random directions (rotation invariance)."""
    planes: list[list[float]] = []
    for _ in range(num_planes):
        raw = [random.gauss(0.0, 1.0) for _ in range(dim)]
        # Normalization isn't strictly necessary (sign is scale-invariant) but keeps
        # the geometry clean and avoids numerical issues with extreme magnitudes.
        norm = math.sqrt(sum(x * x for x in raw))
        planes.append([x / norm for x in raw])
    return planes


def compute_hash(
    vector: list[float],
    hyperplanes: list[list[float]],
) -> int:
    """Hash a vector to an integer bucket using random hyperplane projections.

    Each hyperplane contributes one bit: sign(dot(vector, plane)).
    k hyperplanes → k-bit hash → 2^k possible buckets.

    Math-to-code: h(v) = Σᵢ sign(v · rᵢ) · 2ⁱ
    where rᵢ is the i-th random hyperplane normal vector."""
    hash_val = 0
    for i, plane in enumerate(hyperplanes):
        # sign(dot product) determines which side of the hyperplane v falls on
        if dot_product(vector, plane) >= 0.0:
            hash_val |= (1 << i)
    return hash_val


class LSHIndex:
    """Locality-Sensitive Hashing index using multiple hash tables.

    Multiple tables boost recall: if a true neighbor falls in a different bucket in
    one table (due to an unlucky hyperplane), it may land in the same bucket in another.

    Recall probability for a neighbor at angle θ:
      P(found) = 1 - (1 - (1 - θ/π)^k)^L
    where k = bits per table, L = number of tables.

    More tables (L↑): higher recall, more memory (L copies of all vectors).
    More bits (k↑): fewer candidates per bucket (faster), but lower per-table recall.
    The product k*L controls total hash computations per query."""

    def __init__(
        self,
        dim: int,
        num_tables: int,
        num_hash_bits: int,
    ) -> None:
        self.dim = dim
        self.num_tables = num_tables
        self.num_hash_bits = num_hash_bits

        # Each table has its own set of random hyperplanes — independence is critical.
        # Shared hyperplanes would make tables correlated, defeating the purpose of
        # multiple tables (which is to get independent chances at finding neighbors).
        self.hyperplanes: list[list[list[float]]] = [
            generate_random_hyperplanes(num_hash_bits, dim)
            for _ in range(num_tables)
        ]

        # hash_tables[t][bucket_hash] = list of (index, vector) pairs
        self.hash_tables: list[dict[int, list[tuple[int, list[float]]]]] = [
            defaultdict(list) for _ in range(num_tables)
        ]

    def build(self, vectors: list[list[float]]) -> None:
        """Index all vectors into each hash table.

        O(n * L * k * d) total: n vectors, L tables, k hyperplanes per table,
        d-dimensional dot product per hyperplane. This is a one-time cost;
        queries amortize it over many lookups."""
        for idx, vec in enumerate(vectors):
            for table_idx in range(self.num_tables):
                bucket = compute_hash(vec, self.hyperplanes[table_idx])
                self.hash_tables[table_idx][bucket].append((idx, vec))

    def query(
        self,
        query_vec: list[float],
        database: list[list[float]],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Find approximate nearest neighbors via LSH.

        1. Hash the query in each table → get candidate buckets
        2. Union all candidates across tables (dedup by index)
        3. Re-rank candidates by exact cosine similarity
        4. Return top-k

        The speedup comes from step 2: instead of scanning all n vectors, we only
        compute exact similarity for the (much smaller) candidate set. The re-ranking
        step is exact — LSH only prunes the search space, it doesn't approximate
        the similarity computation itself."""
        # Collect candidate indices from all tables. Using a set for O(1) dedup —
        # the same vector often appears in matching buckets across multiple tables.
        candidate_indices: set[int] = set()
        for table_idx in range(self.num_tables):
            bucket = compute_hash(query_vec, self.hyperplanes[table_idx])
            for idx, _ in self.hash_tables[table_idx].get(bucket, []):
                candidate_indices.add(idx)

        # Re-rank candidates by exact cosine similarity.
        # This is the same computation as brute-force, but over |candidates| << n vectors.
        scored: list[tuple[int, float]] = []
        for idx in candidate_indices:
            sim = cosine_similarity(query_vec, database[idx])
            scored.append((idx, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def bucket_stats(self) -> dict[str, float]:
        """Report hash table statistics for diagnostics.

        Healthy LSH: buckets should be neither too full (no pruning benefit) nor
        too sparse (neighbors split across empty buckets)."""
        total_buckets = 0
        total_entries = 0
        max_bucket_size = 0
        for table in self.hash_tables:
            total_buckets += len(table)
            for bucket_entries in table.values():
                size = len(bucket_entries)
                total_entries += size
                max_bucket_size = max(max_bucket_size, size)
        avg_bucket_size = total_entries / max(total_buckets, 1)
        return {
            "total_buckets": total_buckets,
            "avg_bucket_size": avg_bucket_size,
            "max_bucket_size": max_bucket_size,
        }


# === EVALUATION METRICS ===

def recall_at_k(
    predicted: list[tuple[int, float]],
    ground_truth: list[tuple[int, float]],
    k: int,
) -> float:
    """Fraction of true top-k neighbors found by the approximate method.

    recall@k = |predicted_top_k ∩ true_top_k| / k

    This is THE metric for approximate nearest neighbor search. A system with
    recall@10 = 0.8 finds 8 of the 10 true nearest neighbors on average."""
    true_set = {idx for idx, _ in ground_truth[:k]}
    pred_set = {idx for idx, _ in predicted[:k]}
    return len(true_set & pred_set) / k


# === MAIN: BUILD INDEX, SEARCH, AND COMPARE ===

def main() -> None:
    print("=" * 70)
    print("VECTOR SEARCH: Exact vs Approximate (LSH)")
    print("=" * 70)

    # --- Data generation ---
    print(f"\nGenerating {NUM_VECTORS} vectors of dimension {VECTOR_DIM} "
          f"in {NUM_CLUSTERS} clusters...")
    database = generate_clustered_vectors(NUM_VECTORS, VECTOR_DIM, NUM_CLUSTERS,
                                          CLUSTER_SPREAD)
    queries = generate_query_vectors(database, NUM_QUERIES)
    print(f"Generated {NUM_QUERIES} query vectors (perturbed database samples).")

    # --- Build LSH index ---
    print(f"\nBuilding LSH index: {NUM_TABLES} tables, {NUM_HASH_BITS} bits/table "
          f"({2**NUM_HASH_BITS} buckets/table)...")
    build_start = time.time()
    lsh = LSHIndex(VECTOR_DIM, NUM_TABLES, NUM_HASH_BITS)
    lsh.build(database)
    build_time = time.time() - build_start
    print(f"Index built in {build_time:.3f}s")

    stats = lsh.bucket_stats()
    print(f"  Total buckets across all tables: {stats['total_buckets']:.0f}")
    print(f"  Average bucket size: {stats['avg_bucket_size']:.1f}")
    print(f"  Max bucket size: {stats['max_bucket_size']:.0f}")

    # --- Run searches and collect metrics ---
    print(f"\nRunning {NUM_QUERIES} queries (top-{TOP_K})...")

    brute_times: list[float] = []
    lsh_times: list[float] = []
    recalls: list[float] = []
    candidate_counts: list[int] = []

    for query in queries:
        # Brute-force (ground truth)
        t0 = time.time()
        bf_results = brute_force_search(query, database, TOP_K)
        brute_times.append(time.time() - t0)

        # LSH (approximate)
        t0 = time.time()
        lsh_results = lsh.query(query, database, TOP_K)
        lsh_times.append(time.time() - t0)

        # Recall: how many true neighbors did LSH find?
        recalls.append(recall_at_k(lsh_results, bf_results, TOP_K))

        # Track candidate set size to understand the pruning ratio
        # Re-compute candidates to count them (query method doesn't expose this)
        candidates = set()
        for table_idx in range(NUM_TABLES):
            bucket = compute_hash(query, lsh.hyperplanes[table_idx])
            for idx, _ in lsh.hash_tables[table_idx].get(bucket, []):
                candidates.add(idx)
        candidate_counts.append(len(candidates))

    # --- Results ---
    avg_brute_ms = sum(brute_times) / len(brute_times) * 1000
    avg_lsh_ms = sum(lsh_times) / len(lsh_times) * 1000
    avg_recall = sum(recalls) / len(recalls)
    avg_candidates = sum(candidate_counts) / len(candidate_counts)
    speedup = avg_brute_ms / avg_lsh_ms if avg_lsh_ms > 0 else float("inf")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<35} {'Brute-Force':>15} {'LSH':>15}")
    print("-" * 65)
    print(f"{'Avg query time (ms)':<35} {avg_brute_ms:>15.3f} {avg_lsh_ms:>15.3f}")
    print(f"{'Vectors examined per query':<35} {NUM_VECTORS:>15} {avg_candidates:>15.0f}")
    print(f"{'Pruning ratio':<35} {'100%':>15} "
          f"{avg_candidates/NUM_VECTORS*100:>14.1f}%")
    print(f"{'Recall@{TOP_K}':<35} {'1.000 (exact)':>15} {avg_recall:>15.3f}")
    print(f"{'Speedup':<35} {'1.00x':>15} {speedup:>14.2f}x")

    # --- Recall distribution ---
    recall_buckets: dict[str, int] = {"1.0": 0, "0.8-0.9": 0, "0.6-0.7": 0, "<0.6": 0}
    for r in recalls:
        if r >= 1.0 - 1e-9:
            recall_buckets["1.0"] += 1
        elif r >= 0.8:
            recall_buckets["0.8-0.9"] += 1
        elif r >= 0.6:
            recall_buckets["0.6-0.7"] += 1
        else:
            recall_buckets["<0.6"] += 1

    print(f"\nRecall distribution across {NUM_QUERIES} queries:")
    for bucket_name, count in recall_buckets.items():
        bar = "#" * int(count / NUM_QUERIES * 40)
        print(f"  {bucket_name:>8}: {count:>3} ({count/NUM_QUERIES*100:5.1f}%) {bar}")

    # --- Parameter sensitivity: vary hash bits ---
    # Demonstrates the key tradeoff knob: more bits = more precise buckets = fewer
    # candidates (faster) but lower recall (more true neighbors in adjacent buckets).
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY: Effect of hash bits (k) on recall vs speed")
    print("=" * 70)
    print(f"\nFixed: {NUM_TABLES} tables, {NUM_VECTORS} vectors, dim={VECTOR_DIM}")
    print(f"\n{'Bits (k)':<10} {'Buckets':<12} {'Avg Candidates':<18} "
          f"{'Recall@{TOP_K}':<15} {'Speedup':<10}")
    print("-" * 65)

    # Precompute brute-force results for the first 10 queries (speed)
    sample_queries = queries[:10]
    bf_sample_results = [brute_force_search(q, database, TOP_K) for q in sample_queries]

    for bits in [4, 6, 8, 10, 12]:
        test_lsh = LSHIndex(VECTOR_DIM, NUM_TABLES, bits)
        test_lsh.build(database)

        test_recalls: list[float] = []
        test_candidates: list[int] = []
        test_times: list[float] = []

        for i, query in enumerate(sample_queries):
            t0 = time.time()
            results = test_lsh.query(query, database, TOP_K)
            test_times.append(time.time() - t0)
            test_recalls.append(recall_at_k(results, bf_sample_results[i], TOP_K))

            candidates = set()
            for table_idx in range(NUM_TABLES):
                bucket = compute_hash(query, test_lsh.hyperplanes[table_idx])
                for idx, _ in test_lsh.hash_tables[table_idx].get(bucket, []):
                    candidates.add(idx)
            test_candidates.append(len(candidates))

        avg_r = sum(test_recalls) / len(test_recalls)
        avg_c = sum(test_candidates) / len(test_candidates)
        avg_t = sum(test_times) / len(test_times) * 1000
        spd = avg_brute_ms / avg_t if avg_t > 0 else float("inf")
        print(f"{bits:<10} {2**bits:<12} {avg_c:<18.0f} {avg_r:<15.3f} {spd:<10.2f}x")

    # --- Parameter sensitivity: vary number of tables ---
    print(f"\n{'Tables (L)':<12} {'Avg Candidates':<18} "
          f"{'Recall@{TOP_K}':<15} {'Speedup':<10}")
    print("-" * 55)

    for tables in [1, 4, 8, 12, 20]:
        test_lsh = LSHIndex(VECTOR_DIM, tables, NUM_HASH_BITS)
        test_lsh.build(database)

        test_recalls = []
        test_candidates = []
        test_times = []

        for i, query in enumerate(sample_queries):
            t0 = time.time()
            results = test_lsh.query(query, database, TOP_K)
            test_times.append(time.time() - t0)
            test_recalls.append(recall_at_k(results, bf_sample_results[i], TOP_K))

            candidates = set()
            for table_idx in range(tables):
                bucket = compute_hash(query, test_lsh.hyperplanes[table_idx])
                for idx, _ in test_lsh.hash_tables[table_idx].get(bucket, []):
                    candidates.add(idx)
            test_candidates.append(len(candidates))

        avg_r = sum(test_recalls) / len(test_recalls)
        avg_c = sum(test_candidates) / len(test_candidates)
        avg_t = sum(test_times) / len(test_times) * 1000
        spd = avg_brute_ms / avg_t if avg_t > 0 else float("inf")
        print(f"{tables:<12} {avg_c:<18.0f} {avg_r:<15.3f} {spd:<10.2f}x")

    # --- Distance metric comparison ---
    # Show that cosine similarity and euclidean distance can disagree when vectors
    # have different magnitudes, but agree on normalized vectors.
    print("\n" + "=" * 70)
    print("DISTANCE METRIC COMPARISON")
    print("=" * 70)

    # Pick a query and show top-5 by each metric
    demo_query = queries[0]

    # Cosine similarity ranking
    cos_ranking = []
    euc_ranking = []
    for idx, vec in enumerate(database):
        cos_ranking.append((idx, cosine_similarity(demo_query, vec)))
        euc_ranking.append((idx, euclidean_distance(demo_query, vec)))

    cos_ranking.sort(key=lambda x: x[1], reverse=True)   # higher = more similar
    euc_ranking.sort(key=lambda x: x[1], reverse=False)   # lower = closer

    cos_top5 = {idx for idx, _ in cos_ranking[:5]}
    euc_top5 = {idx for idx, _ in euc_ranking[:5]}
    overlap = len(cos_top5 & euc_top5)

    print(f"\nTop-5 overlap between cosine and euclidean: {overlap}/5")
    print("(Disagreement arises when vectors differ in magnitude — cosine ignores")
    print(" magnitude while euclidean is sensitive to it.)")

    # --- Connection to RAG ---
    print("\n" + "=" * 70)
    print("CONNECTION TO RAG (microrag.py)")
    print("=" * 70)
    # This is the same retrieval step that microrag.py uses: given a query embedding,
    # find the most relevant document chunks by vector similarity. In microrag.py, the
    # search is brute-force (fine for small document sets). For production RAG with
    # millions of chunks, LSH or HNSW replaces the linear scan — the tradeoff shown
    # above is exactly the one production RAG systems navigate.
    print("\nIn RAG, the retrieval step IS vector search:")
    print("  1. Embed the query → vector")
    print("  2. Search the document chunk index → nearest vectors")
    print("  3. Feed retrieved chunks + query to the LLM")
    print(f"\nWith {NUM_VECTORS} chunks, brute-force takes {avg_brute_ms:.1f}ms/query.")
    print(f"LSH reduces this to {avg_lsh_ms:.1f}ms/query "
          f"({speedup:.1f}x faster, {avg_recall:.1%} recall).")
    print("At 1M chunks, brute-force becomes seconds per query — LSH or HNSW is essential.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
