---
title: "Hash Tables in Disguise: How Transformers Cache Attention"
author: Alban
pubDatetime: 2026-02-23T10:00:00Z
slug: llm-1-kv-cache
featured: false
draft: false
tags:
  - llm-systems
  - ai-agents
  - db-internals
description: "The KV cache is just a hash table. Understanding hash index design — load factors, eviction, open addressing — explains most of what makes modern LLM inference fast or slow."
---

*This is part of a companion series to [DB Internals from Scratch](/blog/posts/week-1-hash-index/). Each post takes a data structure we implemented in C++ and maps it to where it reappears — often unacknowledged — in modern LLM and AI agent systems.*

---

## The Mapping

In [Week 1](/blog/posts/week-1-hash-index/) we built two hash indexes from scratch. The core insight: instead of *searching* for a value, a hash function lets you *calculate* exactly where it lives. O(1) lookup at the cost of ordering.

That same idea is doing critical work inside every transformer inference call you've ever made. It just goes by a different name: **the KV cache**.

---

## What the KV Cache Actually Is

In a transformer, every token in the context attends to every previous token via a dot-product attention mechanism:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

For each layer, every token produces three vectors: a **query** (Q), a **key** (K), and a **value** (V). During autoregressive generation — producing one token at a time — the new token needs to attend to *all previous tokens*.

Without caching, generating token *t* requires recomputing K and V for all *t-1* previous tokens. That's O(t) compute per token, O(t²) total for a sequence of length *t*.

With a KV cache: compute K and V once per token, store them, and retrieve them for every subsequent token. Lookup cost: **O(1) per token per layer**.

The KV cache is a table mapping (layer, position) → (K vector, V vector). That's a hash map.

---

## Load Factor: The Same Problem, The Same Math

In our hash index, load factor α = n/m drives everything. When α approaches 1, performance degrades — either probe chains lengthen (open addressing) or linked lists grow (chaining).

The KV cache has an identical constraint. Each (K, V) pair for a single token, at a single layer, in a typical 7B-parameter model (e.g. LLaMA-2 7B) occupies:

```
2 vectors × d_kv dimensions × bytes_per_element
= 2 × 128 × 2 bytes (fp16)
= 512 bytes per layer per token
```

With 32 layers:

```
32 × 512 bytes = 16 KB per token
```

An A100 GPU has 80 GB HBM. For a batch of 1 request:

```
max_tokens = 80 GB / 16 KB ≈ 5,000,000 tokens
```

But you're not serving one request. With a batch of 100 concurrent requests:

```
max_tokens_per_request = 80 GB / (100 × 16 KB) = 50,000 tokens
```

At 50k tokens per request you're fine. But real-world deployments want batches of 1,000+ with 128k context windows — and suddenly you're doing the same load factor arithmetic as a hash table designer.

**This is why KV cache memory is the primary bottleneck in LLM serving**, not compute. It's a storage problem.

---

## Multi-Query Attention: Reducing Bucket Count

In standard **Multi-Head Attention (MHA)**, every head has its own K and V matrices. With *h* heads, you store *h* KV pairs per token per layer.

For LLaMA-2 7B: 32 heads × 16 KB = 512 KB per token per layer. That's a lot of buckets.

**Multi-Query Attention (MQA)**, introduced by Shazeer (2019), collapses all heads to share a single K and V:

```
MHA: h queries, h keys, h values    → h KV pairs stored
MQA: h queries, 1 key,  1 value     → 1 KV pair stored
```

KV cache size drops by a factor of *h* — 32× for a 32-head model. This is directly analogous to **reducing the number of hash table buckets**: you save memory at the cost of some "resolution" (expressiveness per head).

**Grouped Query Attention (GQA)**, used in LLaMA-3, Mistral, and Gemma, is the middle ground: *g* groups, each sharing one KV pair across *h/g* query heads:

```
GQA: h queries, g keys, g values    → g KV pairs stored  (1 ≤ g ≤ h)
```

When *g=1*: MQA. When *g=h*: MHA. GQA is the **load factor tuning knob** of attention — trade memory for model quality.

The empirical finding: *g = h/8* (e.g. 4 KV heads for 32 query heads) recovers ~99% of MHA quality at 8× memory reduction. That's a remarkably good tradeoff — analogous to finding the sweet spot load factor that maximizes cache utilisation without degrading performance.

---

## Eviction: The Page Replacement Problem

Our hash table rehashes when it's full — doubles capacity and redistributes everything. But a KV cache can't grow arbitrarily: GPU memory is fixed. When the cache fills, something must be **evicted**.

This is exactly the buffer pool manager's page replacement problem (which we'll implement in Week 4), applied to attention caches.

### Sliding Window Attention

The simplest eviction policy: **LRU applied globally** — drop the oldest tokens. This is "sliding window attention": only attend to the last *w* tokens.

Mistral 7B uses a window size of 4,096. Every token older than 4,096 positions is evicted from cache. O(1) cache size, O(1) eviction — identical to a circular buffer LRU cache.

The cost: you lose long-range context. For many tasks (coding, summarisation) this doesn't matter. For tasks requiring reasoning over a 100k-token document, it's catastrophic.

### StreamingLLM

MIT's StreamingLLM (2023) made an interesting discovery: you don't need to keep all recent tokens — just the first few ("attention sinks") plus a sliding window. Transformers disproportionately attend to initial tokens regardless of content, so keeping them prevents distribution shift when the window slides.

```
KV cache = [sink tokens 0..3] + [sliding window last w tokens]
```

This is LRU with a **pinned cold set** — the same pattern as buffer pools that pin frequently-accessed pages regardless of recency.

### H2O (Heavy Hitter Oracle)

H2O (ICML 2023) applies a more sophisticated policy: track cumulative attention scores for each token, evict the lowest-scoring ones. This is **LFU (Least Frequently Used)** for the KV cache.

Empirically, a small fraction of tokens receive the majority of attention — the heavy hitters. Keeping them yields much better quality than LRU at the same cache budget.

The paper showed 20× KV cache compression with <5% quality degradation on many benchmarks. The tradeoff is O(n) tracking overhead vs O(1) for sliding window.

---

## Prefix Caching: Shared Hash Buckets

Here's another direct mapping. When serving many requests with the same system prompt (e.g. "You are a helpful assistant..."), recomputing KV for the prompt on every request is wasteful. **Prefix caching** (implemented in vLLM, TensorRT-LLM, SGLang) hashes the token sequence prefix and reuses cached KV tensors across requests.

```
h(token_sequence_prefix) → cached KV block
```

This is a **shared hash index** — multiple queries hitting the same bucket and reusing the result. In vLLM's implementation, KV blocks are managed like database pages: fixed-size chunks (e.g. 16 tokens), with a hash over the token IDs as the key.

The copy-on-write problem: if one request extends the shared prefix, you can't mutate the cached block (other requests are reading it). vLLM uses **block-level reference counting** — identical to how database systems handle shared buffer pool pages. Modify-on-write increments a new block's reference count; the original stays cached.

---

## PagedAttention: The Buffer Pool Manager

vLLM's key innovation is **PagedAttention** (SOSP 2023): instead of allocating a contiguous KV cache per sequence (which causes fragmentation), it manages KV cache in fixed-size **pages** (called "blocks"), exactly like an OS virtual memory system.

```
Physical KV memory:  [block 0][block 1][block 2]...
Logical sequence:    block_table[seq_id] = [2, 7, 3, 11, ...]
```

A sequence's KV cache is stored in non-contiguous blocks, mapped via a block table — identical to a page table in virtual memory, or a buffer pool manager's frame table.

Benefits:
- **No fragmentation**: blocks are fixed-size, allocated on demand
- **Copy-on-write for beam search**: branching sequences share blocks until they diverge
- **Efficient batching**: sequences of different lengths pack efficiently into the same physical memory

This is exactly what we'll build in Week 4 (Buffer Pool Manager), applied verbatim to GPU memory.

---

## Hashing for Retrieval: RAG as a Hash Join

Retrieval-Augmented Generation (RAG) — fetching relevant documents before generating — is a **hash join** at inference time.

In query engines (Week 5), a hash join works by:
1. Build phase: hash the smaller relation into a hash table keyed by join attribute
2. Probe phase: for each row in the larger relation, look up its join key

In RAG:
1. Build phase (indexing): embed documents, store vectors in a vector database
2. Probe phase (retrieval): embed the query, find nearest neighbours

The "hash function" here is the embedding model. The "bucket" is a region of embedding space. Approximate nearest neighbour search (HNSW, IVF) trades the exactness of a perfect hash for speed — exactly the same tradeoff as open addressing vs chaining.

FAISS's IVF (Inverted File Index) literally partitions the embedding space into *m* Voronoi cells (clusters) — structurally identical to hash buckets — and only searches the cells nearest to the query. Adjusting the number of cells probed is adjusting the probe sequence length in open addressing.

---

## The Deeper Point

What this series is really about is that **the problems don't change, only the substrate does**.

Database engineers in the 1970s-90s solved the fundamental problems of managing bounded fast memory in front of slow large storage, handling concurrent access, and making reads and writes fast under adversarial access patterns. They developed formal tools — amortised analysis, the buffer pool abstraction, MVCC, WAL — to reason about these problems precisely.

LLM systems are hitting the same walls:
- Bounded GPU memory (= buffer pool)
- Expensive recomputation (= the case for caching)
- Concurrent requests with shared state (= MVCC)
- Replay and auditability (= WAL)

The engineers who understand both layers — who know *why* the 0.75 load factor threshold exists and can apply that reasoning to KV cache sizing — will design better systems than those who treat the LLM stack as a black box.

---

## What's in This Series

| DB Internals | AI Systems Equivalent |
|---|---|
| Hash Index | KV Cache, MQA/GQA, prefix caching |
| LSM Tree | Vector store ingestion, HNSW with write buffering |
| Columnar Storage | Tensor memory layouts, FlashAttention tiling |
| Buffer Pool Manager | PagedAttention, context window management |
| Query Engine | RAG pipelines, query planning over embeddings |
| MVCC | Multi-agent state consistency |
| WAL & Recovery | Agent action logs, LangSmith traces, rollback |

---

## Next in This Series

**LSM Trees → Vector Store Ingestion**

When you insert a million documents into Pinecone or Weaviate, the system can't rebuild the HNSW graph on every write — it would be O(n log n) per insert. The solution is the same one RocksDB uses: buffer writes in memory, merge lazily. We'll look at how Qdrant's HNSW index construction works and why it's an LSM tree in disguise.

---

*DB implementations: [github.com/anocaj/db-internals](https://github.com/anocaj/db-internals)*
