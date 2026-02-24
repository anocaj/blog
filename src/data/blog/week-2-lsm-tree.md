---
title: "LSM Trees from Scratch: How RocksDB Writes a Million Keys Per Second"
author: Alban
pubDatetime: 2026-02-24T09:00:00Z
slug: week-2-lsm-tree
featured: false
draft: false
tags:
  - db-internals
  - c++
  - data-structures
description: "Building an LSM Tree from first principles — SkipList MemTable, Bloom filters, SSTable compaction, and the RUM conjecture. Week 2 of the db-internals series."
---

## The Problem With Random Writes

In [Week 1](/blog/posts/week-1-hash-index/) we built a hash index and noted that it answers point queries in O(1). But databases aren't just key-value stores — they also need to handle **writes**, and lots of them.

Consider what happens when you write a row to a B+ tree stored on disk. The B+ tree is sorted, so your new row must land in the right leaf page. But "the right leaf page" is determined by the key value — it could be anywhere on disk. If you're inserting sequential IDs, you're always at the end of the file (sequential). If you're inserting UUIDs, phone numbers, or hash values, you're writing to random disk locations.

The problem isn't the CPU time — it's the I/O:
- **Spinning HDD sequential I/O**: ~200 MB/s (the disk head moves predictably)
- **Spinning HDD random I/O**: ~100–200 *operations/s* at ~8 KB per op ≈ **1–2 MB/s**
- **NVMe SSD sequential I/O**: ~7 GB/s
- **NVMe SSD random I/O**: ~1 million IOPS × ~4 KB = **4 GB/s**

For HDDs, random vs sequential is a **100–200× gap**. For SSDs the gap is smaller but still significant — and SSDs have write amplification concerns that make frequent random writes degrade the device lifetime.

The LSM Tree (Log-Structured Merge-Tree, O'Neil et al. 1996) was invented to close this gap: **turn all disk writes into sequential writes**, at the cost of more complexity in the read path.

---

## The Core Insight: Batch and Sort

The fundamental LSM insight is almost embarrassingly simple:

> Don't write to disk immediately. Collect writes in memory, sort them, then write everything at once as a single sequential file.

That's it. Every other complexity in LSM trees is either a consequence of this decision or a refinement of it.

### The Write Path

```
Write("user:alice", "30")
Write("user:bob", "25")
Write("user:charlie", "31")
...N more writes...
MemTable full → sort → write ONE file to disk
```

The single file written to disk is called an **SSTable** (Sorted String Table) — an immutable, sorted sequence of key-value pairs. Immutable because once written, we never go back and modify it. We only append, and periodically merge files together.

### Why This Changes Everything

Instead of `N` random writes (one per `put`), we get:
- One sequential scan of an in-memory data structure (O(N log N) for sort)
- One sequential write to disk (O(N) bytes)

On an HDD: N=100,000 writes of 100-byte values → 10 MB. At 100 IOPS random, this takes **16 minutes**. Sequential: 10 MB / 200 MB/s = **0.05 seconds**. Twenty thousand times faster.

---

## Architecture Overview

```
Writes ──► [ MemTable (SkipList, in memory) ]
                          │
                    (flush when full)
                          ▼
               [ L0 SSTables (on disk) ]
                          │
                  (compact when ≥ 4 files)
                          ▼
               [ L1 SSTables (larger) ]
                          │
                  (compact when ≥ 4 files)
                          ▼
               [ L2 SSTables (larger still) ]

Reads: MemTable → L0 → L1 → L2 → ...
       (Bloom filters short-circuit most levels)
```

**MemTable**: an in-memory sorted structure (we use a SkipList). All writes go here first.

**SSTable**: when the MemTable exceeds a size threshold, it's flushed to disk as an immutable sorted file — one large sequential write.

**Compaction**: periodically, multiple SSTables at a level are merged into one larger SSTable at the next level. This reclaims space and keeps the number of files bounded, which bounds read amplification.

**Bloom filters**: each SSTable has a Bloom filter that answers "is this key definitely NOT in this file?" in O(1), allowing most SSTables to be skipped during reads.

---

## Component 1: The Bloom Filter

A Bloom filter is a space-efficient probabilistic data structure that answers **set membership queries** with no false negatives and a tunable false positive rate.

### The Mathematics

A Bloom filter consists of:
- A bit array of size *m*, initially all zeros
- *k* independent hash functions h₁, h₂, …, hₖ, each mapping a key to {0, …, m−1}

**Insert(key)**: set bits h₁(key), h₂(key), …, hₖ(key) to 1.

**Contains(key)**: return true iff ALL of h₁(key), h₂(key), …, hₖ(key) are 1.

The guarantee: if a key was inserted, all its bits are set → no false negatives. But multiple keys can set overlapping bits → false positives are possible.

**False Positive Rate (FPR)**

After inserting *n* keys into a filter with *m* bits, the probability that any given bit is still 0 is:

```
P(bit = 0) = (1 - 1/m)^{kn} ≈ e^{-kn/m}
```

The probability a non-inserted key passes all *k* checks (false positive):

```
FPR = P(all k bits set for a random key)
    = (1 - P(bit = 0))^k
    ≈ (1 - e^{-kn/m})^k
```

**Optimal number of hash functions**

Taking the derivative of FPR with respect to *k* and setting it to zero:

```
k* = (m/n) × ln(2) ≈ 0.693 × (m/n)
```

At optimal *k*, the FPR simplifies to:

```
FPR* = (0.5)^k* = (0.6185)^{m/n}
```

| Bits per key (m/n) | Optimal k | FPR |
|---|---|---|
| 6 | 4 | ~5.6% |
| 8 | 6 | ~2.1% |
| 10 | 7 | ~0.8% |
| 14 | 10 | ~0.1% |
| 20 | 14 | ~0.006% |

RocksDB uses 10 bits/key by default (`NewBloomFilterPolicy(10)`) — the sweet spot between memory cost and 0.8% FPR.

**Why this matters for reads**

Consider an LSM tree with 5 levels. Reading a missing key requires checking all 5 levels. With Bloom filters at FPR=1%:

```
Expected SSTable reads = 1 + 0.01 × 5 ≈ 1.05 files
```

Without Bloom filters: 5+ files. The filter eliminates 95%+ of unnecessary disk reads.

### Implementation

```cpp
class BloomFilter {
public:
    explicit BloomFilter(size_t expected_keys, size_t bits_per_key = 10)
        : num_bits_(std::max(expected_keys * bits_per_key, size_t(64)))
        , bits_(num_bits_, false)
    {
        // k = (m/n) * ln(2) — optimal number of hash functions
        double optimal_k = static_cast<double>(bits_per_key) * std::log(2.0);
        num_hash_functions_ = static_cast<size_t>(
            std::max(1.0, std::min(30.0, std::round(optimal_k)))
        );
    }

    void insert(const std::string& key) {
        for (size_t i = 0; i < num_hash_functions_; ++i) {
            size_t bit = hash_string(key, i) % num_bits_;
            bits_[bit] = true;
        }
    }

    bool possibly_contains(const std::string& key) const {
        for (size_t i = 0; i < num_hash_functions_; ++i) {
            size_t bit = hash_string(key, i) % num_bits_;
            if (!bits_[bit]) return false;  // definite miss
        }
        return true;  // all bits set — probably present
    }
```

We use different seed values (0, 1, 2, …) for the same FNV-1a function to simulate *k* independent hash functions. This is a well-known trick: given a single good hash function, `h_i(key) = fnv1a(key, seed=i)` gives effectively independent functions.

---

## Component 2: The SkipList MemTable

The MemTable must support:
1. O(log n) insert and point lookup
2. O(n) sorted scan (for flush)
3. Lock-free concurrent access (in production systems)

A balanced BST (red-black tree, AVL tree) gives O(log n) insert/lookup and O(n) scan, but:
- Concurrent lock-free implementations are notoriously complex due to tree rotations that must touch multiple nodes atomically
- In-order traversal requires a stack or Morris traversal

A **SkipList** gives the same asymptotic bounds with:
- Trivially O(n) scan: just follow the level-0 pointers, which form a sorted linked list
- Lock-free concurrent implementations via CAS on forward pointers (used by both LevelDB and RocksDB)

### SkipList Structure

A SkipList is a multi-level linked list. Level 0 is a complete sorted linked list. Level *k* is a sublist of level *k-1*, where each element is included independently with probability *p*.

```
Level 3: ──────────────────────────────── [50] ─────────────────── NIL
Level 2: ────────────── [20] ───────────── [50] ─── [80] ───────── NIL
Level 1: ──── [10] ───── [20] ───── [40] ─ [50] ─── [80] ─── [90] NIL
Level 0: [5] ─ [10] ─── [20] ─ [30] ─ [40] ─ [50] ─ [60] ─ [80] ─ [90] NIL
```

**Search complexity**: at each level, advance until the next node exceeds the target, then drop to the level below. This skips O(1/p) nodes per level, and there are O(log_{1/p} n) levels. Total: O(log n) expected.

For p=0.25 and n=1,000,000: expected levels ≈ log₄(10⁶) ≈ 10.

**Height distribution**: the probability a node reaches level *k* is p^{k-1}. Expected height = 1/(1-p). For p=0.25: expected height ≈ 1.33 levels per node, same asymptotic bounds as a balanced BST.

**Why p=0.25?** The level-promotion probability controls the space/time tradeoff. p=0.5 gives a factor-of-2 skip at each level (better search), but doubles the memory for pointers. p=0.25 gives a factor-of-4 skip at each level with 25% fewer pointers on average. In practice, p=0.25 (used by LevelDB and RocksDB) is the standard choice.

### Implementation

```cpp
static constexpr int SKIPLIST_MAX_LEVEL = 16;
static constexpr double SKIPLIST_P      = 0.25;

struct SkipNode {
    std::string key;
    std::string value;
    bool tombstone;
    std::array<SkipNode*, SKIPLIST_MAX_LEVEL> forward;
};

class SkipList {
public:
    void put(const std::string& key, const std::string& value,
             bool tombstone = false) {
        // update[i] = rightmost node at level i with key < target
        std::array<SkipNode*, SKIPLIST_MAX_LEVEL> update;
        SkipNode* cur = header_;

        for (int i = level_ - 1; i >= 0; --i) {
            while (cur->forward[i] && cur->forward[i]->key < key)
                cur = cur->forward[i];
            update[i] = cur;
        }

        SkipNode* next = cur->forward[0];
        if (next && next->key == key) {
            // Overwrite in-place — no new allocation
            next->value = value;
            next->tombstone = tombstone;
            return;
        }

        int new_level = random_level();  // geometric(p=0.25)
        SkipNode* node = new SkipNode{key, value, tombstone};
        for (int i = 0; i < new_level; ++i) {
            node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = node;
        }
        ++size_;
    }

private:
    int random_level() {
        int lvl = 1;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        while (dist(rng_) < SKIPLIST_P && lvl < SKIPLIST_MAX_LEVEL)
            ++lvl;
        return lvl;
    }
};
```

Key observation: the sorted scan (`scan_all`) just follows `header_->forward[0]` — a single pointer-chase pass. This is what makes SkipList flush O(n) and cache-friendly during the final linear pass.

---

## Component 3: SSTable with Sparse Index

An SSTable is an immutable, sorted sequence of key-value pairs. Once written, it is never modified. It may be read many times, and eventually merged with other SSTables during compaction.

### Real SSTable File Format (RocksDB/LevelDB)

```
┌──────────────────────────────────────────┐
│ Data Block 0     (4 KB)                  │
│  key₀:val₀, key₁:val₁, ..., key₃₁:val₃₁ │
├──────────────────────────────────────────┤
│ Data Block 1     (4 KB)                  │
│  key₃₂:val₃₂, ..., key₆₃:val₆₃          │
├──────────────────────────────────────────┤
│ ...                                      │
├──────────────────────────────────────────┤
│ Index Block                              │
│  (last_key_in_block → block_offset) ×N  │
├──────────────────────────────────────────┤
│ Bloom Filter Block                       │
├──────────────────────────────────────────┤
│ Metaindex Block                          │
├──────────────────────────────────────────┤
│ Footer (magic, index ptr, metaindex ptr) │
└──────────────────────────────────────────┘
```

### Sparse Index

Storing one index entry per key would require O(n) memory for the index — same as storing all the data. Instead, store one entry per **block** (every ~32–128 keys). To find a key:

1. Binary search the sparse index to find the block that might contain the key: O(log(n/B))
2. Load that block and scan linearly: O(B)

With B=32 and n=10⁶ keys:
- Dense index: 10⁶ entries to binary-search
- Sparse index: 31,250 entries to binary-search, then 32 comparisons

Total comparisons: log₂(31,250) + 32 ≈ 15 + 32 = 47 vs log₂(10⁶) ≈ 20 for dense. The sparse index is actually slower in raw comparisons, but it has far better cache behavior: the entire sparse index often fits in L2 cache, while a dense index is too large to cache.

```cpp
class SSTable {
public:
    std::optional<std::string> get(const std::string& key,
                                   bool* found_tombstone = nullptr) const {
        // Step 1: Bloom filter check — O(k), eliminates ~99% of misses
        if (!bloom_.possibly_contains(key)) return std::nullopt;

        // Step 2: Sparse index binary search
        size_t start = sparse_index_lower_bound(key);

        // Step 3: Linear scan of the block
        size_t end = std::min(start + 2 * SPARSE_INDEX_INTERVAL, entries_.size());
        for (size_t i = start; i < end; ++i) {
            if (entries_[i].key == key) {
                if (entries_[i].tombstone) {
                    if (found_tombstone) *found_tombstone = true;
                    return std::nullopt;
                }
                return entries_[i].value;
            }
            if (entries_[i].key > key) break;  // sorted — early exit
        }
        return std::nullopt;
    }
```

---

## Component 4: Compaction — The Heart of LSM

Compaction is what keeps LSM trees from degrading into an unbounded pile of SSTables. It's an **N-way external merge sort**, run continuously in the background.

### Why Compaction is Necessary

Without compaction:
- Every flush creates a new L0 SSTable
- A point read must check ALL L0 files (they may have overlapping key ranges)
- Space usage grows indefinitely (old values are never reclaimed)
- Tombstones never get garbage collected

With compaction:
- The number of files per level is bounded
- At levels L1+, files have non-overlapping key ranges (in leveled compaction), so a point read touches at most 1 file per level
- Old versions of keys are eliminated, reclaiming space
- Tombstones at the last level are garbage collected (no older copy can exist below)

### Compaction Strategies

**Size-Tiered Compaction (Cassandra default)**

When *N* SSTables of similar size accumulate at a level, merge them all into one larger file.

```
L0: [f1][f2][f3][f4]  → compact → L1: [merged_f1234]
L1: [merged_f1234][g1][g2][g3] → compact → L2: [big_merged]
```

- **Write amplification**: each byte is written once per level crossing. WA ≈ L levels
- **Space amplification**: up to N copies of a key can coexist (before compaction triggers). Up to 2× space during compaction peak
- **Read amplification**: high — many L0 files to check, no key-range isolation
- **Best for**: write-heavy workloads (time-series, event logs)

**Leveled Compaction (RocksDB default, LevelDB)**

Each level has a fixed total size budget (e.g., L1=256MB, L2=2.56GB, L3=25.6GB). When a level exceeds budget, pick ONE SSTable and merge it with the overlapping SSTables in the next level.

At L1+, SSTables have non-overlapping key ranges. A point read at level *i* touches at most 1 file.

- **Write amplification**: O(size_ratio × L). RocksDB defaults: ratio=10, L=7 → WA ≈ 10-30×
- **Space amplification**: ~1.1× — at most one copy of each key per level
- **Read amplification**: ~L files (1 per level, due to non-overlapping ranges)
- **Best for**: read-heavy workloads, space-constrained environments

**Universal Compaction (RocksDB option)**

Tiered at lower levels (absorb write bursts), leveled at upper levels (maintain read performance). Used for workloads with mixed read/write patterns.

### Our Implementation: Simplified Tiered

We implement tiered compaction: when L0 accumulates ≥ `l0_compaction_trigger` SSTables, merge ALL of L0 and ALL of L1 into a new L1 SSTable.

```cpp
void do_compact_level(size_t level) {
    // Collect all entries from this level AND the target level
    std::vector<SSTableEntry> all_entries;
    for (auto& sst : levels_[level + 1])  // existing L_{i+1}
        for (auto& e : sst->entries()) all_entries.push_back(e);
    for (auto& sst : levels_[level])      // L_i being compacted
        for (auto& e : sst->entries()) all_entries.push_back(e);

    // Sort by key; within same key, L_i entries come last (they're newer)
    std::stable_sort(all_entries.begin(), all_entries.end(),
        [](const SSTableEntry& a, const SSTableEntry& b) {
            return a.key < b.key;
        });

    // Dedup: keep only the most recent version of each key
    std::vector<SSTableEntry> merged;
    bool is_last_level = (level + 1 == levels_.size() - 1);

    for (size_t i = 0; i < all_entries.size(); ) {
        size_t j = i;
        while (j < all_entries.size() && all_entries[j].key == all_entries[i].key)
            ++j;
        SSTableEntry& newest = all_entries[j - 1];

        // Garbage-collect tombstones at the last level
        if (newest.tombstone && is_last_level) { /* drop it */ }
        else merged.push_back(newest);

        i = j;
    }

    levels_[level].clear();
    levels_[level + 1].clear();
    levels_[level + 1].push_back(
        std::make_unique<SSTable>(std::move(merged), ++seq_counter_)
    );
}
```

The dedup step is the key: after stable-sorting by key, multiple versions of the same key appear consecutively, in order of age (oldest first, newest last). Keeping only `all_entries[j-1]` (the last one) always gives us the most recent version.

---

## The RUM Conjecture: No Free Lunch

The LSM tree design is governed by a fundamental constraint called the **RUM Conjecture** (Athanassoulis et al., 2016): you cannot simultaneously minimise all three of:

- **R**ead overhead (RA = bytes read / bytes useful)
- **U**pdate overhead (WA = bytes written / logical write size)  
- **M**emory overhead (SA = bytes stored / bytes of live data)

Every storage structure picks at most two:

| Structure | Read Cost | Write Cost | Space Cost |
|---|---|---|---|
| B+ Tree | **Low** O(log n) | Medium (random writes) | **Low** |
| Hash Index | **Low** O(1) | Medium | **Low** |
| LSM (tiered) | Medium (multiple levels) | **Low** (sequential) | Medium |
| LSM (leveled) | **Low** (bounded) | High (10-30× WA) | **Low** |
| Append-only log | High O(n) scan | **Low** O(1) append | Medium |

The beauty of the LSM architecture is that it makes the tradeoff *tunable*: by adjusting level sizes, compaction triggers, and compaction strategy, you can slide along the RUM tradeoff curve to match your workload.

### Write Amplification Derivation

For leveled compaction with *L* levels and size ratio *R* (each level is *R*× larger):

Each byte is written once to the WAL + once to L0 + potentially once per level during compaction. In the worst case, a byte participates in one compaction per level:

```
WA = 1 (WAL) + 1 (L0 flush) + R (L0→L1) + R (L1→L2) + ... + R (L_{L-1}→L_L)
   = 2 + R × L
```

For RocksDB defaults (R=10, L=6): WA ≈ 62×. This is measured in practice as **25–35×** because most keys don't cross all levels and tombstones reduce the effective data size.

For tiered compaction: compaction merges N files into 1, so each byte participates in O(log_N total_data / level_size) compactions. Practical WA ≈ 3–10×.

### Read Amplification Derivation

For a point query with Bloom filter FPR *p*:
- MemTable: 1 lookup (always)
- L0: check all |L0| files (they may overlap) — each filtered by Bloom → expected |L0| × p files read
- L1+: with leveled compaction, 1 file per level (no overlap) → each filtered → expected p files read per level

```
RA = 1 + |L0| × p + (L - 1) × p
   ≈ 1 + p × (|L0| + L)
```

With p=0.01, |L0|=4, L=6: RA ≈ 1 + 0.01 × 10 = 1.1. Excellent!

For a **range query** of width *w* keys: Bloom filters are useless (we need all files covering the range). RA = |L0| + L files per range query. This is why LSM trees are designed for point-lookup-heavy workloads.

---

## Tombstones: Lazy Deletion

A fundamental consequence of immutability: you cannot delete a key from an SSTable. The file is immutable.

Instead, LSM trees use **tombstones**: a delete operation inserts a special marker (`tombstone = true`) into the MemTable. This marker propagates through the system like any other write.

```
Timeline:
  t=1: put("user:alice", "active") → MemTable
  t=2: MemTable flushes → L0[0]: { "user:alice": "active" }
  t=3: remove("user:alice") → MemTable: { "user:alice": TOMBSTONE }
  t=4: MemTable flushes → L0[1]: { "user:alice": TOMBSTONE }

Read at t=4:
  Check MemTable → empty
  Check L0[1] (newest) → finds TOMBSTONE → return null  ✓
  (L0[0] with "active" is never reached)
```

The tombstone is the **write**, and it wins because the read path always checks newer files first.

**Garbage collection**: tombstones are eventually eliminated during compaction. When a tombstone reaches the deepest level (below which no older copy can exist), the compaction drops it — there's nothing left to shadow.

This is why **compaction must reach all levels** for tombstones to be reclaimed. A key deleted early can persist in disk storage until compaction reaches the bottom level. This is a real operational issue: Cassandra has famous production incidents where tombstone accumulation caused read performance to crater.

---

## Amplification Metrics in Our Implementation

```cpp
struct Metrics {
    uint64_t total_writes       = 0;
    uint64_t bytes_written_disk = 0;
    uint64_t compaction_count   = 0;
    uint64_t bloom_filter_hits  = 0;
    uint64_t sstable_reads      = 0;

    double write_amplification() const {
        return static_cast<double>(bytes_written_disk) / total_writes;
    }
};
```

Running the `LargeDataset` test (1,000 keys, memtable_size=100, l0_trigger=4):

```
total_writes:       1000
bytes_written_disk: ~8,000 (flush) + ~8,000 (compaction) = ~16,000
write_amplification: ~2.0×

compaction_count: 3
sstable_reads (missing key queries): much lower with Bloom filters
```

The measured WA of ~2× is low because our tiered compaction only crosses 2 levels and the data fits in a small number of SSTables. Real-world RocksDB WA is 25–30× under sustained write load.

---

## How Real Databases Use LSM Trees

### RocksDB (Meta/Facebook, 2012)

RocksDB is a fork of LevelDB optimised for SSD-heavy, multi-core server workloads.

**MemTable**: RocksDB supports pluggable MemTable implementations. The default is still a SkipList, but RocksDB 6.x added a `HashLinkedListMemTable` (hash-bucket per key prefix, for prefix-scan workloads) and a `VectorMemTable` (append-only, for bulk loads). The SkipList MemTable supports **concurrent writes** via a lock-free CAS-based implementation.

**Compaction**: RocksDB defaults to leveled compaction with *L* levels, level ratio *R*=10, and L0 size 256MB. L1 = 256MB, L2 = 2.56GB, L3 = 25.6GB, etc. An SST file at level *i* is compacted with the overlapping SST files at level *i+1* when the level size budget is exceeded.

**Bloom filters**: RocksDB uses a partitioned Bloom filter (the filter is split across 128 partitions, one per cache block) to improve cache efficiency for large filters. Prefix Bloom filters allow `prefix_seek` queries to use the filter even for range scans.

**Write throughput**: on a single NVMe SSD, RocksDB sustains **500 MB/s to 1 GB/s** of write throughput. This is why it's the storage engine for TiKV (TiDB), CockroachDB, YugabyteDB, and dozens of other systems.

### Apache Cassandra (2008)

Cassandra uses LSM as its primary storage engine (via its own "storage engine," not RocksDB).

**MemTable**: a ConcurrentSkipListMap (Java). Multiple tables share a MemTable allocation pool; when total usage exceeds a threshold, the largest MemTable is flushed.

**Compaction**: Cassandra 4.x supports three strategies:
- **STCS (Size-Tiered)**: default, write-optimised
- **LCS (Leveled)**: better read performance, higher WA, used for latency-sensitive workloads
- **TWCS (Time-Window)**: designed for time-series data — compacts SSTables within the same time window together, then makes them immutable. Tombstones from TTL-expired data are collected efficiently.

**Tombstone hell**: Cassandra is notorious for tombstone accumulation issues. A table receiving many deletes can accumulate millions of tombstones across SSTables; a wide-partition scan must traverse all of them, causing GC pressure and query timeouts. The fix is aggressive compaction and TTL tuning — but this requires explicit operator intervention.

### ClickHouse MergeTree (2016)

ClickHouse's MergeTree table engine is an LSM-inspired structure optimised for analytics (bulk ingestion + column scan).

**Key differences from RocksDB**:
- Parts (ClickHouse's SSTables) are columnar, not row-oriented — each column is stored in a separate file within a part
- No MemTable in the traditional sense — data is written in **batches** of at least thousands of rows
- Compaction ("merges") is background-triggered based on part count per partition
- The primary index is a **sparse index** (one entry per 8192 rows), stored in memory — exactly the structure we implemented
- **Bloom filter**: optional, per-column; not per-SSTable as in RocksDB

ClickHouse's `INSERT` performance is astounding — 500,000 to millions of rows per second — because it writes large sorted parts directly, avoiding the overhead of the SkipList MemTable path.

### LevelDB (Google, 2011)

The original open-source LSM implementation from Google (by Jeff Dean and Sanjay Ghemawat). Our implementation follows LevelDB's design closely:

- Single-threaded compaction
- SkipList MemTable with lock (no concurrent writes)
- 7 levels, size ratio 10
- Per-SSTable Bloom filter (2 bytes per key in original LevelDB, 10 bits in RocksDB)
- Prefix compression within data blocks (keys that share a prefix store only the suffix)

---

## Comparison Table

| Metric | Hash Index | B+ Tree | LSM (tiered) | LSM (leveled) | Append Log |
|---|---|---|---|---|---|
| Point read | O(1) | O(log n) | O(L) + Bloom | O(L) + Bloom | O(n) scan |
| Range scan | ✗ | O(log n + k) | O(L × file_scan) | O(L + k) | O(n) |
| Write | O(1) amort. | O(log n) random | **O(1) sequential** | **O(1) sequential** | O(1) append |
| Write amplification | ~1× | ~1-3× | ~3-10× | **~25-60×** | **1×** |
| Space amplification | ~1× | ~1× | ~2× peak | **~1.1×** | grows forever |
| Delete | O(1) | O(log n) | tombstone | tombstone | ✗ or rewrite |
| Ordering | ✗ | ✓ | ✓ (slow) | ✓ | ✓ (insertion) |
| Best for | key-value cache | OLTP, indexes | write-heavy | mixed | event logs |

---

## Unit Tests: Covering the Edge Cases

We wrote 28 unit tests covering all three components. The most important ones:

**BloomFilter: NoFalseNegatives** — for 500 inserted keys, every single one must return `possibly_contains = true`. This tests the fundamental guarantee that must never be violated.

**BloomFilter: FalsePositiveRateWithinBounds** — inserts 1,000 keys, queries 10,000 non-inserted keys, verifies that the actual FPR is within 2.5× of the theoretical value.

**SkipList: LargeInsertionMaintainsOrder** — inserts 1,000 keys in shuffled order, verifies `scan_all` returns them in sorted order.

**LSMTree: TombstoneInSSTableHidesOlderValue** — writes a key, flushes to L0, writes a tombstone, flushes again. Verifies the tombstone (newer SSTable, higher sequence number) hides the value (older SSTable).

**LSMTree: CompactionEliminatesDuplicates** — writes the same key 9 times across multiple flushes, forces compaction, verifies only version_8 (the latest) survives.

**LSMTree: BloomFiltersReduceSSTableReads** — runs identical queries on two LSM trees (one with, one without Bloom filters), verifies the Bloom-enabled version has fewer or equal SSTable reads.

---

## Design Decisions and Tradeoffs

### SkipList vs std::map as MemTable

We chose SkipList over `std::map` (red-black tree) for three reasons:
1. **Lock-free concurrent extension**: adding concurrent writes to a SkipList requires only CAS on forward pointers. Red-black tree rebalancing touches multiple nodes and is hard to make lock-free.
2. **Cache behavior during flush**: level-0 pointers form a linked list — O(n) sequential scan with good prefetch behavior. In-order BST traversal is less cache-friendly.
3. **Pedagogical value**: SkipList is a beautiful structure that combines randomness and linked lists to achieve O(log n) expected time.

### Fixed-Size Forward Array vs Dynamic Allocation

```cpp
std::array<SkipNode*, SKIPLIST_MAX_LEVEL> forward;  // we use this
// vs
std::vector<SkipNode*> forward;                      // alternative
```

The fixed array wastes memory for low-level nodes (a level-1 node allocates 15 unused pointers) but avoids a heap allocation per node. For a MemTable with 100k entries, this is ~12.8 MB of pointer storage either way, but the fixed array version has better allocation locality.

### Tiered vs Leveled Compaction

We chose tiered for simplicity: when a level is full, merge everything at once. Leveled compaction is more complex: it must select which SSTable at level *i* to compact and which overlapping SSTables at level *i+1* to include.

For a teaching implementation, tiered captures all the essential mechanics. For production use, switch to leveled for better read performance and space efficiency.

### Tombstone Garbage Collection

Our tombstone GC is conservative: we only collect tombstones at the *last* level (where we know no older copy exists). This is correct but can be improved:

- If we track the minimum sequence number of any active snapshot, we can collect tombstones once the tombstone's sequence number is below all active snapshots (even if not at the last level). This is how RocksDB handles snapshot-aware GC.

---

## What's Next

**Week 3: Columnar Storage** — B+ trees and LSM trees are row-oriented: all fields of a row are stored together. For analytics queries (`SELECT avg(revenue) FROM orders WHERE year = 2024`), reading the entire row to access one column is wasteful.

Columnar storage (Parquet, ClickHouse, DuckDB) stores each column separately. A scan of 10 columns across 1 billion rows reads only the 10 relevant column files — a 10× reduction in I/O for typical analytics. Combined with run-length encoding, delta encoding, and dictionary encoding, columnar storage achieves 10–20× compression ratios on real-world analytics data, turning what would be a 500 GB table into a 30 GB file.

We'll implement a columnar storage engine with multiple compression schemes and measure the I/O reduction empirically.

---

*Code: [github.com/anocaj/db-internals/pull/9](https://github.com/anocaj/db-internals/pull/9)*
