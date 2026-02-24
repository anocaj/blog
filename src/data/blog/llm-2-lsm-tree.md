---
title: "LSM Trees in Disguise: How Vector Databases Handle a Million Embeddings Per Second"
author: Alban
pubDatetime: 2026-02-24T10:00:00Z
slug: llm-2-lsm-tree
featured: false
draft: false
tags:
  - llm-systems
  - ai-agents
  - db-internals
description: "LSM Trees are write-buffering structures that turn random mutations into sequential I/O — and they're hiding inside every vector database, LLM embedding pipeline, and model training data loader you've used."
---

*This is part of a companion series to [DB Internals from Scratch](/blog/posts/week-2-lsm-tree/). Each post takes a data structure we implemented in C++ and maps it to where it reappears — often unacknowledged — in modern LLM and AI agent systems.*

---

## The Mapping

In [Week 2](/blog/posts/week-2-lsm-tree/) we built an LSM Tree from scratch. The core insight: **random writes are slow; sequential writes are fast; therefore, buffer all writes in memory and flush as sorted runs**. Read by checking layers from newest to oldest, using Bloom filters to skip irrelevant files.

That same insight is hiding in half the AI infrastructure stack:

| DB Concept | AI Equivalent |
|---|---|
| MemTable (in-memory sorted buffer) | Qdrant's in-memory HNSW segment, Pinecone's write buffer |
| SSTable (immutable sorted file) | Immutable HNSW graph segment, frozen embedding store |
| Bloom filter (probabilistic skip) | Approximate nearest neighbour pre-filter, IVF coarse quantiser |
| Compaction (merge sorted runs) | HNSW graph rebuild, incremental index merging |
| Tombstone (lazy delete) | Soft deletion in vector stores, deferred HNSW node removal |
| L0 → L1 → L2 levels | Qdrant's segment hierarchy, Weaviate's segment tree |
| Write amplification | Embedding re-indexing cost, HNSW construction amortisation |

The mapping isn't metaphorical — it's structural. When you call `qdrant.upsert(vectors)` in production, a component that is architecturally identical to an LSM MemTable is receiving your vectors.

---

## The Write Problem for Vector Databases

Let's start with the same problem the LSM tree solved for RocksDB, but restate it in AI terms.

You have an LLM-powered product: a document assistant, a semantic search engine, a recommendation system. Your users are uploading documents, and your pipeline:

1. Chunks documents into passages
2. Embeds each passage with `text-embedding-3-small` (1536 dimensions, float32 = 6 KB per embedding)
3. Inserts the embedding into a vector database for nearest-neighbour search

The ingest rate: 10,000 documents/hour × 20 chunks/document = 200,000 embeddings/hour ≈ **55 inserts/second**.

Sounds manageable. But HNSW — the dominant index structure in vector databases — has a problem: **inserting a single vector into an HNSW graph is O(log n) work that requires random memory access across the entire graph structure**. For an index with 100 million vectors:
- Graph traversal touches ~O(log n) = ~27 layers of nodes
- Each node access is a random pointer chase
- At 100M vectors × 1536 × 4 bytes = ~600 GB, most of the graph doesn't fit in RAM

Sound familiar? This is the random write problem for B-trees, translated to vector space. And the solution is the same one the LSM tree invented: **buffer writes in a fast in-memory structure, merge lazily to disk**.

---

## Qdrant: An LSM Tree for HNSW Graphs

Qdrant is the most architecturally transparent about this. Its [source code](https://github.com/qdrant/qdrant) explicitly uses the term "segments" for what are functionally SSTables, and its segment management is a direct LSM-inspired design.

### Qdrant's Segment Architecture

```
Writes → [ Appendable Segment (in-memory/mutable) ]
                    │
             (segment full)
                    ▼
          [ Immutable Segment 0 ] (sealed, on disk)
          [ Immutable Segment 1 ] (sealed, on disk)
          [ Immutable Segment 2 ] (sealed, on disk)
                    │
              (optimizer runs)
                    ▼
          [ Merged Segment ] (compacted HNSW graph)
```

**Appendable segment**: accepts new vectors via an in-memory structure. Queries against this segment use brute-force search (no index yet) — analogous to scanning the MemTable linearly.

**Immutable segments**: sealed segments with a built HNSW graph. Queries use proper graph traversal. Read-only, like SSTables.

**Optimizer**: Qdrant's background process that merges small immutable segments into larger ones, rebuilding the HNSW graph across the merged segment. This is **compaction**.

### The Analogy in Code

Qdrant's `Collection` struct manages a `SegmentHolder` — a collection of segments very similar to our `levels_` in `LSMTree`:

```rust
// Qdrant (conceptually):
struct SegmentHolder {
    segments: HashMap<SegmentId, LockedSegment>,
    // "LockedSegment" = either appendable (MemTable) or immutable (SSTable)
}
```

When a write arrives:
1. It goes to the appendable segment (MemTable path)
2. When the appendable segment fills, a new appendable segment is created and the old one is sealed (flush path)
3. The optimizer detects too many small immutable segments and merges them (compaction path)

The optimizer's merge strategy: select segments to merge by minimising the ratio of dead vectors (deleted) to live vectors — exactly the "tombstone density" heuristic that drives SSTable compaction in RocksDB.

### Bloom Filters → IVF Pre-filtering

In our LSM implementation, before reading any SSTable, we check its Bloom filter:

```cpp
if (!bloom_.possibly_contains(key)) return std::nullopt;  // skip this file
```

In Qdrant, before searching a segment, it checks whether the segment's payload index could contain any vectors matching the filter predicate:

```python
# Qdrant query with filter
client.search(
    collection_name="documents",
    query_vector=[0.2, 0.1, ...],
    query_filter=Filter(
        must=[FieldCondition(key="year", match=MatchValue(value=2024))]
    )
)
```

Internally, Qdrant uses a **payload index** (essentially an inverted index) to determine whether a segment contains any vectors with `year=2024`. If not, the segment is skipped entirely — same semantics as a Bloom filter, different implementation (exact rather than probabilistic).

---

## Pinecone: The Black-Box LSM

Pinecone is more opaque about its internals, but its observed behavior reveals LSM-like architecture:

**Write buffering**: Pinecone's API acknowledges that freshly upserted vectors may not be immediately searchable. From their docs: *"There is a brief period of time where new vectors may not be visible to queries."* This is the MemTable flush delay — vectors are in the write buffer but the index (SSTable equivalent) hasn't been built yet.

**Eventual consistency of deletes**: Pinecone soft-deletes vectors by marking them as deleted, with actual removal happening asynchronously — identical to LSM tombstones.

**Segment-based architecture**: Pinecone's internal architecture (partially revealed in their engineering blog) shows that indexes are composed of multiple immutable segments that are periodically merged. The `describe_index_stats()` API returns a `fullness` metric — the equivalent of the L0 SSTable count in our LSM implementation.

The mathematical tradeoff is identical:
- More segments = more write parallelism (lower WA) but higher query latency (more segments to search)
- Fewer segments (after merge) = lower query latency but higher merge cost (higher WA)

This is the RUM conjecture applied to vector search: you cannot simultaneously minimise read overhead (single-segment search), update overhead (no rebuilding), and memory overhead (compact representation).

---

## Weaviate: Time-Stamped Segments

Weaviate's HNSW implementation (Go, open source) reveals even more explicit LSM thinking.

### Write Path

Weaviate maintains an **in-memory HNSW graph** for the "active" segment. All writes go here — analogous to our SkipList MemTable. The in-memory graph is flushed to disk when it exceeds a size threshold (default: when the segment reaches ~200MB or after a time interval).

### Tombstones in HNSW

Deleting a vector from an HNSW graph is expensive: the graph is a connected structure, and removing a node can disconnect the graph, degrading search quality. Weaviate's solution: **tombstones**.

```go
// Weaviate HNSW (conceptually)
type hnsw struct {
    nodes       []vertex
    tombstoned  map[uint64]bool  // deleted but not yet removed
}
```

A deleted node is marked in `tombstoned` but remains in the graph. During search, tombstoned nodes are skipped in results — identical to how our `SSTable::get` handles tombstones:

```cpp
if (entries_[i].tombstone) {
    if (found_tombstone) *found_tombstone = true;
    return std::nullopt;
}
```

Tombstoned nodes are garbage-collected during **tombstone cleanup**, a background process that rebuilds the HNSW graph without the deleted nodes. This is the vector database equivalent of LSM compaction eliminating tombstones at the last level.

The operational implication is identical to Cassandra's tombstone problem: if you delete many vectors and don't trigger cleanup, tombstoned nodes accumulate, search quality degrades (the graph becomes sparse and the search path traverses many dead nodes), and memory usage stays high. Weaviate exposes `flat_search_cut_off` to switch to brute-force search when too many tombstones exist — a circuit breaker for the tombstone problem.

---

## The Write Amplification Parallel

In our LSM implementation we track write amplification:

```cpp
double write_amplification() const {
    return static_cast<double>(bytes_written_disk) / total_writes;
}
```

For leveled compaction with ratio R=10 and L levels: WA ≈ R × L ≈ 30–60×.

**HNSW construction has an analogous amplification factor**.

When a vector is inserted into HNSW:
1. Search the graph to find the M nearest neighbours (O(log n) graph traversal)
2. Add bidirectional edges to those M neighbours

Each edge addition potentially triggers a **neighbourhood pruning** step: if a node already has M_max connections, one must be removed to make room. This "shrink connections" operation re-evaluates all neighbours.

The construction cost per vector: O(M × log n) graph traversal + O(M²) connection updates in the worst case.

For batch ingestion of N vectors, the total construction cost is:

```
C_construct = N × M × log(N) × C_search
```

When you later **rebuild** an index (after accumulating many writes or after merging segments), you pay this cost again for all N vectors in the merged segment. This is **compaction write amplification**: the cost of re-building the HNSW graph is the vector equivalent of merging SSTables.

For RocksDB with WA=30×: 1 GB of data → 30 GB of disk writes over its lifetime.
For a vector database with 10M vectors, merged 3 times: 10M embeddings computed once + ~30M neighbourhood updates per merge × 3 merges = effective WA of ~10×.

The mathematics are structurally identical. The substrate is different (sorted strings vs graph edges), but the engineering tradeoff — **how often to merge, how large to make segments, how much write cost to pay for read performance** — is the same problem.

---

## LLM Training Data: The LSM Tree in the Dataloader

The LSM pattern appears in a surprising place: **LLM training data pipelines**.

Training a large language model requires streaming hundreds of billions of tokens. The dataset is too large to fit in memory (GPT-3's training data: ~570GB; Llama-3's: ~15TB). The naïve approach — shuffle all tokens and stream them — requires a full sort of terabytes of data.

**The solution: multi-level shuffling**, which is structurally an LSM-inspired merge.

1. **Level 0 (MemTable)**: load a "shard" (~1GB) into memory and shuffle in-place. This is random within the shard but not globally random.

2. **Level 1 (L0 SSTable)**: write the shuffled shard to disk as an immutable file. Many shards accumulate.

3. **Compaction**: when training, read from M random shards simultaneously, interleaving their tokens. This is an M-way merge — not sorted, but randomly interleaved — which approximates global shuffle.

HuggingFace's `datasets` library implements this as `IterableDataset` with `shuffle(buffer_size=N)`:

```python
from datasets import load_dataset

dataset = load_dataset("c4", "en", streaming=True)
shuffled = dataset.shuffle(seed=42, buffer_size=10_000)
```

Internally, `shuffle` maintains a buffer of `buffer_size` examples (the MemTable), yields one randomly chosen example from the buffer, and refills with the next example from the stream. This is **reservoir sampling** — statistically equivalent to a random shuffle without loading the full dataset into memory.

The quality tradeoff: larger `buffer_size` = better approximation of true global shuffle (less correlation between consecutive batches) = better training convergence. Smaller `buffer_size` = lower memory usage. This is the LSM read/space amplification tradeoff, re-expressed as shuffle quality vs memory.

---

## SGLang's RadixAttention: Prefix Cache as SSTable

[SGLang](https://github.com/sgl-project/sglang)'s RadixAttention (2024) is a KV cache management system that bears a striking resemblance to SSTable design.

Traditional KV caching hashes the **full token sequence** prefix to look up cached KV blocks. SGLang uses a **radix tree** (trie) instead: the trie keys are token sequences, and each node stores a pointer to the KV block for that prefix.

```
Radix Tree:
"You are a helpful"  → KV block A
"You are a helpful assistant"  → KV block B (shares prefix with A)
"You are an expert"  → KV block C (shares "You are a" with A and B)
```

This enables **prefix sharing at the sub-block level** — different requests that share only partial prefixes can still reuse KV computation for the shared part.

The LSM parallel:
- Each KV block is an immutable chunk of cached data, like an SSTable
- The radix tree is the index, like the SSTable's sparse index
- When a request extends a shared prefix, a new KV block is appended — immutable, like SSTable creation
- Old KV blocks that are no longer referenced are evicted (tombstoned, then GC'd)
- The eviction policy (LRU on the radix tree leaves) is analogous to SSTable compaction priority

SGLang reports **5× KV cache utilization** compared to naive prefix hashing — the benefit of the trie structure is that it exploits the hierarchical structure of prefixes, exactly as LSM levels exploit the size-tiered structure of data.

---

## vLLM's Block Manager: Compaction on GPU

vLLM's PagedAttention (which we referenced in Week 1 for its buffer-pool analogy) also exhibits LSM-like write patterns in its **block manager**.

When a new token is generated:
1. The KV for the new token is written to the next free position in the current block (MemTable append)
2. When the block is full (e.g., 16 tokens), it becomes immutable — `block.is_full = True`
3. Full blocks are referenced by a block table and never modified (SSTable semantics)

vLLM's `preemption` (when GPU memory is full and a sequence must be paused):

```python
# vLLM block manager preemption:
# 1. Free all blocks for the preempted sequence (mark as tombstones)
# 2. When the sequence is resumed, re-allocate blocks and re-compute KV
#    (= re-generate from scratch, no swap)
```

This is the equivalent of **dropping an L0 SSTable** to reclaim memory — the data isn't durably stored on disk, so preemption means losing the KV computation entirely. The "durability" cost of LSM (data is not persistent until flushed to SSTable) becomes the "recomputation" cost of vLLM (preempted sequences must regenerate their KV from scratch).

vLLM 0.4+ added **swap** — the ability to offload KV blocks from GPU to CPU RAM. This is **two-level storage**: GPU ≈ L0 (fast, small), CPU ≈ L1 (slower, larger). The swap mechanism mirrors LSM's flush: when GPU blocks are full, move blocks to CPU (flush to L1). On resume, bring them back (prefetch from L1 to L0). The latency cost of swapping KV from CPU to GPU (PCIe bandwidth ~32 GB/s) is the exact analogue of L0 compaction latency in RocksDB.

---

## Where the Analogy Breaks Down

The LSM analogy is strong but not perfect. Here are the places it fails:

### 1. The "sorted" invariant is meaningless in vector space

LSM trees work because they can sort by key. Merging two sorted runs into one is O(N) by merge sort. All the compaction analysis depends on sortability.

HNSW graphs are not sorted in any useful sense. Merging two HNSW segments isn't O(N) — it requires rebuilding the entire neighbour graph from scratch, which is O(N log N). This makes vector store "compaction" far more expensive than LSM compaction, and it's why vector databases batch writes aggressively and compact infrequently.

**Implication**: the write amplification for vector stores is dominated by graph construction cost, not by sequential I/O. RocksDB's WA is measured in bytes written; Qdrant's "WA" is measured in CPU-hours spent rebuilding HNSW.

### 2. Bloom filters don't generalise to approximate search

In an LSM tree, a Bloom filter answers "is this exact key in this SSTable?" — a binary, exact question. False negatives are forbidden.

In vector search, the query is "what are the 10 nearest vectors to this query?" — an approximate, ranked question. There is no Bloom filter for this. IVF pre-filtering (checking which Voronoi cells to search) is the closest analogue, but it has false negatives by design — vectors in a cell adjacent to the query cell might be closer than vectors in the queried cell.

This fundamental difference (exact vs approximate) means that vector databases cannot use Bloom filters to safely skip segments. They must either search all segments (full recall, high latency) or accept recall degradation (fast but approximate). The LSM tree's Bloom filter is the mechanism that gives it O(1) expected read complexity; vector stores have no equivalent.

### 3. No natural partial ordering for tombstones

In an LSM tree, a tombstone at sequence number *t* shadows all versions of the key with sequence number < *t*. This works because there's a total order on writes.

In a vector database, "deleting" a vector means removing it from the graph and all its edge connections. When segments are merged, there's no "sequence number" that lets you automatically drop a deleted vector's contribution — you have to explicitly check a deleted set and exclude those vectors from the new graph. This is why tombstone GC in vector stores (Weaviate's tombstone cleanup, Qdrant's dead vector optimizer) requires a full graph rebuild, not just a skip in a merge sort.

### 4. The MemTable doesn't need to be sorted

Our SkipList MemTable is sorted because LSM requires sorted SSTables (to enable binary search and merge sort during compaction). A vector database's write buffer doesn't need to be sorted — it just needs to support fast approximate nearest-neighbour search. Qdrant's appendable segment uses **brute-force search** (scan all vectors, compute all distances) rather than maintaining a sorted structure, because for small buffers (< 20,000 vectors), brute force is faster than graph construction.

This is the inversion: LSM uses a sorted in-memory structure (SkipList) so that the flush to disk is cheap (O(N)). Vector databases use an unsorted in-memory structure (array of embeddings) because the flush to disk requires an expensive graph build regardless.

---

## Why Understanding LSM Makes You a Better AI Systems Engineer

**1. You understand why writes are expensive in RAG systems.**

When you insert 1 million documents into Pinecone and it takes 20 minutes, that's not network latency — it's the HNSW construction cost that's the vector equivalent of LSM compaction. Knowing this tells you: batch your inserts, use `upsert` (not `insert` one-by-one), and don't query during heavy ingestion.

**2. You can reason about freshness vs performance tradeoffs.**

Every vector database has a parameter controlling how often to compact (rebuild the merged HNSW graph). Higher compaction frequency → better search quality (fewer small segments, more connected graph) but higher CPU cost. Lower frequency → more segments, higher recall degradation, but lower operational cost. This is the LSM l0_compaction_trigger parameter. You can now reason about it quantitatively.

**3. You understand why streaming ingestion and batch ingestion require different architectures.**

Streaming ingestion (1 doc/second continuously) → many small segments, frequent compaction, write amplification dominates. Batch ingestion (1M docs at midnight) → write to a temporary in-memory structure, build one large HNSW graph, swap it in atomically. The second approach has WA ≈ 1×; the first has WA >> 1×. Most production RAG systems use a hybrid: batch the incoming stream into hourly segments, rebuild each segment as a single HNSW graph, then merge across segments periodically.

**4. You can debug "my index is getting slow after many updates."**

Symptom: a vector database that was fast at launch gets slower over time despite no change in query load. Diagnosis: tombstone accumulation. Too many deletes and re-inserts have left dead nodes in the HNSW graph. The search path traverses dead nodes, increasing path length and reducing recall. Fix: trigger a manual compaction / tombstone cleanup / full index rebuild.

**5. You know when NOT to use vector databases.**

If your workload is mostly reads (search) with occasional bulk inserts, a vector database is great — it's optimised for the read path. If your workload is a continuous stream of single-vector inserts with concurrent queries (e.g., real-time personalisation), the write-buffer/compaction cycle creates either high latency (frequent compaction) or degraded search quality (infrequent compaction). Sometimes a simpler structure — a flat in-memory FAISS index, rebuilt hourly — has better end-to-end performance.

---

## The Deeper Point

The LSM tree solved a problem that was considered intractable in 1996: how to make a storage system that's simultaneously fast to write, fast to read, and space-efficient under mixed workloads with deletes. The solution — buffer in memory, sort, flush sequentially, merge periodically — turns out to be a **general pattern for any system where:**

1. Incoming data is random in key space
2. Underlying storage is fast for sequential access but slow for random access
3. Deletions must be handled lazily
4. The index structure is expensive to build incrementally

GPU memory, HNSW graphs, KV caches, training data pipelines — they all satisfy these conditions. The engineers who built Qdrant, vLLM, and SGLang reinvented LSM-inspired designs, sometimes explicitly and sometimes independently.

Understanding the original design — why the MemTable must be sorted, why Bloom filters are positioned at the SSTable level rather than the key level, why tombstone GC requires knowing the deepest level — gives you the conceptual vocabulary to design and debug these systems rather than treating them as black boxes.

---

## What's in This Series

| DB Internals | AI Systems Equivalent |
|---|---|
| Hash Index | KV Cache, MQA/GQA, prefix caching |
| **LSM Tree** | **Vector store ingestion, HNSW write buffering, PagedAttention swap** |
| Columnar Storage | Tensor memory layouts, FlashAttention tiling |
| Buffer Pool Manager | PagedAttention, context window management |
| Query Engine | RAG pipelines, query planning over embeddings |
| MVCC | Multi-agent state consistency |
| WAL & Recovery | Agent action logs, LangSmith traces, rollback |

---

## Next in This Series

**Columnar Storage → Tensor Memory Layouts and FlashAttention**

When ClickHouse scans 1 billion rows to compute `avg(revenue)`, it only reads the `revenue` column — one sequential I/O instead of reading all columns. The layout decision (row-oriented vs column-oriented) changes I/O by 100× for analytics.

FlashAttention applies the same insight to GPU memory: instead of materialising the full N×N attention matrix (which is column-oriented and doesn't fit in SRAM), it tiles the computation to process blocks of rows and columns that fit in on-chip memory. The "tiling" strategy in FlashAttention 2 is directly analogous to the block-level I/O optimisation in columnar storage engines.

We'll implement a columnar storage engine in C++ with run-length encoding, delta encoding, and dictionary encoding — and map each technique to its FlashAttention counterpart.

---

*DB implementations: [github.com/anocaj/db-internals](https://github.com/anocaj/db-internals)*
