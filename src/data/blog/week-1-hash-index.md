---
title: "Hash Indexes from Scratch: Why Redis is Just a Big Hash Table"
author: Alban
pubDatetime: 2026-02-23T09:00:00Z
slug: week-1-hash-index
featured: true
draft: false
tags:
  - db-internals
  - c++
  - data-structures
description: "Building a hash index from first principles — FNV-1a hashing, separate chaining, open addressing with linear probing, and dynamic rehashing. Week 1 of the db-internals series."
---

## The Problem

A B+ tree finds a key in **O(log n)**. For a table with a billion rows, that's about 30 comparisons — traversing a tree from root to leaf.

But what if you don't need range queries? What if you just want: *"give me the value for this key, right now"*?

A hash index answers in **O(1)** expected time — not 30 comparisons, not even 2. One.

That gap isn't magic. It's a direct consequence of trading one capability (ordering) for another (direct addressing). Understanding *why* that tradeoff holds is what this post is about.

---

## The Core Idea: Direct Addressing

The simplest possible lookup structure is a direct-address table: allocate an array of size *U* (the universe of all possible keys), and store value *v* at index *k*. Lookup is `array[k]` — truly O(1), no hashing at all.

The problem: if keys are 64-bit integers, *U* = 2⁶⁴. That's 18 exabytes for a table of integers. Direct addressing is impractical for any realistic key space.

A **hash function** h: U → {0, …, m−1} compresses the universe down to *m* buckets, where *m* is proportional to the number of keys we actually store. Now we can allocate a reasonably-sized array and use `array[h(k)]` for lookup.

The cost: two distinct keys can map to the same bucket. This is a **collision**, and it's unavoidable.

---

## Collisions Are Inevitable: The Birthday Paradox

Why are collisions unavoidable? Consider inserting *n* keys into a table of *m* buckets, assuming the hash function distributes keys uniformly.

The probability that all *n* keys land in distinct buckets is:

```
P(no collision) = (m/m) × (m-1)/m × (m-2)/m × … × (m-n+1)/m
               = ∏_{i=0}^{n-1} (1 - i/m)
```

Using the approximation 1 - x ≈ e^(-x) for small x:

```
P(no collision) ≈ exp(-∑_{i=0}^{n-1} i/m)
               = exp(-n(n-1) / 2m)
```

For this to stay above 0.5, we need roughly **n < √(m)**. So with m = 1,000,000 buckets, collisions become likely after just ~1,000 insertions. This is the birthday paradox applied to hashing — and it tells us collision handling isn't an edge case; it's the central design problem.

---

## Collision Resolution Strategy 1: Separate Chaining

Each bucket holds a pointer to a linked list of (key, value) pairs. Collisions simply append to the chain.

```
bucket[7] → ["user_42": Alice] → ["user_99": Bob] → null
```

**Expected performance** under uniform hashing, with load factor α = n/m:
- Successful search: **1 + α/2** comparisons
- Unsuccessful search: **1 + α** comparisons
- Insert: O(1) (prepend to chain)
- Delete: O(1) (unlink node)

At α = 1.0 (as many elements as buckets), expected search cost is ~1.5 comparisons. This is remarkably cheap — the analysis comes from the linearity of expectation over the random choices of the hash function (see CLRS §11.2 for the formal proof).

**The cache problem.** Each list node is a separate heap allocation. A cache miss typically costs 100–300 ns; an L1 cache hit costs ~1 ns. On a modern CPU with 64-byte cache lines, a linked list traversal of length *k* causes *k* cache misses. This is the dominant cost in practice, dwarfing the O(1) theoretical bound.

**Used by:** Java's `HashMap`, PostgreSQL hash joins, most textbook implementations.

---

## Collision Resolution Strategy 2: Open Addressing

Instead of allocating external nodes, all (key, value) pairs live directly in the array. On collision, we **probe** — scan forward for an empty slot.

The simplest probe sequence is **linear probing**: on collision at slot *i*, try *i+1*, *i+2*, … (mod *m*).

```
h("user_42") = 7, bucket[7] occupied
h("user_42") = 7, try bucket[8] → empty → insert here
```

**Why it's cache-friendly.** Linear probing scans contiguous memory. A modern CPU prefetcher can detect forward sequential access and load cache lines ahead of your probe. In the common case (low load factor, short probe sequences), the entire probe fits in a single 64-byte cache line.

### The Clustering Problem

Linear probing suffers from **primary clustering**: once a run of occupied slots forms, any new key hashing into the run extends it, making future probes longer. Expected probe length grows as O(1/(1-α)²) near α = 1 — performance degrades badly above α ≈ 0.7.

Formally, under linear probing with load factor α:
- Successful search: ~½(1 + 1/(1-α)) comparisons
- Unsuccessful search: ~½(1 + 1/(1-α)²) comparisons

At α = 0.9: successful ≈ 5.5 probes, unsuccessful ≈ 50.5 probes. This is why we rehash before reaching high load.

### The Deletion Problem

You cannot simply clear a deleted slot. Consider:

```
insert "A" → slot 3
insert "B" → slot 3 occupied, probe to slot 4
delete "A" → clear slot 3
search "B" → hash to slot 3, slot 3 empty → wrongly conclude B is absent
```

The probe chain is broken. The standard fix is **tombstones**: mark deleted slots with a sentinel value. Probing skips tombstones (treats them as occupied); insertion reuses them.

Tombstones accumulate over time, degrading performance. A more aggressive fix is **Robin Hood hashing** (discussed below).

**Used by:** Python's `dict`, Go's `map`, most high-performance hash tables.

---

## Beyond Linear Probing: Robin Hood Hashing

Robin Hood hashing is a refinement of open addressing that reduces variance in probe length. The insight: when inserting a new key, if the new key has probed further from its home slot than the key currently occupying the slot, **swap them** and continue inserting the displaced key.

```
"Rich" key (close to home) gives its slot to a "poor" key (far from home)
→ Robin Hood: steal from the rich, give to the poor
```

This bounds the maximum probe length and enables a neat deletion strategy: when deleting a key, back-shift subsequent keys that would benefit (avoiding tombstones entirely).

Robin Hood hashing is used in **Rust's `HashMap`** (via Hashbrown) and **Abseil's `flat_hash_map`**.

---

## The Hash Function: What Makes It Good?

### Requirements

A hash function for a hash table needs:
1. **Uniform distribution** — keys should spread evenly. Formally, for a random key *k*, P(h(k) = i) = 1/m for all buckets *i*.
2. **Avalanche effect** — a 1-bit change in the input should flip ~half the output bits. This prevents clustering from structured key sets.
3. **Speed** — called on every operation; microseconds matter at scale.
4. **Determinism** — same input always produces same output (required for correctness; randomised hashing is a separate topic).

### Universal Hashing

A **universal hash family** H is a set of functions such that for any two distinct keys x ≠ y:

```
P_{h ∈ H}[h(x) = h(y)] ≤ 1/m
```

If you pick h uniformly at random from a universal family, the expected number of collisions for any fixed set of keys is at most n/m = α. This is the theoretical foundation for the O(1) expected-case bounds — it doesn't matter how adversarial your key distribution is, as long as the hash function is chosen randomly.

In practice we don't re-hash randomly, but well-designed hash functions approximate this behaviour.

### FNV-1a

We implemented **FNV-1a** (Fowler–Noll–Vo, 1a variant):

```cpp
uint64_t fnv1a(const void* data, size_t len) {
    uint64_t hash = 14695981039346656037ULL; // FNV offset basis
    for (size_t i = 0; i < len; ++i) {
        hash ^= ((const uint8_t*)data)[i]; // XOR before multiply
        hash *= 1099511628211ULL;           // FNV prime
    }
    return hash;
}
```

The **FNV-1a** variant (XOR then multiply) has better avalanche behaviour than FNV-1 (multiply then XOR) for inputs that differ only in low-order bits — common in sequential integer keys.

FNV-1a is not cryptographic (it's not collision-resistant against adversarial inputs), but it's fast, simple, and good enough for internal database use where key distributions are not adversarial.

### What Real Databases Use

- **PostgreSQL** uses a Jenkins-style hash internally
- **RocksDB / LevelDB** use a custom murmur-inspired hash for SST bloom filters
- **Redis** uses SipHash-1-2 (a fast *cryptographic* hash) to defend against hash-flooding attacks — an adversary sending crafted keys to force O(n) worst-case behaviour in a hash table exposed over a network

---

## Dynamic Resizing: Amortised Analysis

As the table fills, collision chains lengthen and probe sequences grow. The standard fix: when load factor α exceeds a threshold, **double the capacity** and rehash every key.

```
if (size_ / static_cast<double>(capacity_) >= max_load_factor_) {
    rehash(capacity_ * 2);
}
```

**Why doubling?** Consider a table that has rehashed *k* times, so capacity = m · 2^k. The total work across all rehashes is:

```
n + n/2 + n/4 + … + 1 = 2n
```

This geometric series converges: the total rehashing cost is O(n), giving **amortised O(1) per insert** — the same argument as `std::vector::push_back`.

The threshold matters:
- **Separate chaining: 0.75** — chains absorb overflow without catastrophic degradation
- **Open addressing: 0.50–0.70** — clustering accelerates rapidly past this; 0.70 is a common practical choice

### Incremental Rehashing

Doubling halts the world for O(n) time — unacceptable for latency-sensitive systems. **Redis** uses incremental rehashing: it maintains two hash tables simultaneously during resize, migrating a few buckets per request until migration is complete. Read operations check both tables; once migration finishes, the old table is freed.

---

## Extendible Hashing: Disk-Based Hash Indexes

Everything above assumes the hash table fits in memory. For disk-based storage (where the database index must live on pages), a different approach is needed: **extendible hashing**.

The idea: use only the first *d* bits of the hash value to find a directory entry (a pointer to a disk page/"bucket"). When a page overflows, split it and increment *d* for that bucket. The directory doubles only when necessary.

```
d=2: directory has 4 entries (00, 01, 10, 11)
Insert key with h(k) = 0b1101... → use first 2 bits → directory[11]
Bucket[11] overflows → split, d=3 for this bucket
```

This gives O(1) disk I/Os per lookup (1 directory lookup + 1 page read) and avoids the full table rebuild of static hashing. PostgreSQL's hash index uses a variant of this structure.

---

## What We Built

Two C++17 implementations, templated on key and value type:

### `ChainingHashIndex<K, V>`

```cpp
template<typename K, typename V>
class ChainingHashIndex {
    std::vector<std::list<std::pair<K, V>>> buckets_;
    size_t size_ = 0;
    static constexpr double MAX_LOAD = 0.75;
    // ...
};
```

- Separate chaining with `std::list`
- Rehashes at α = 0.75
- `insert`, `search`, `remove`, `load_factor`

### `OpenAddressingHashIndex<K, V>`

```cpp
template<typename K, typename V>
class OpenAddressingHashIndex {
    enum class State { EMPTY, OCCUPIED, TOMBSTONE };
    struct Slot { K key; V value; State state = State::EMPTY; };
    std::vector<Slot> slots_;
    size_t size_ = 0;
    static constexpr double MAX_LOAD = 0.70;
    // ...
};
```

- Linear probing
- Tombstone deletion
- Rehashes at α = 0.70 (skipping tombstones in the count)

**Test coverage:** 17 unit tests, correctness verified against `std::unordered_map` across:
- Sequential integer keys
- Random string keys
- High collision scenarios (forced via reduced capacity)
- Insert-delete-reinsert cycles (tombstone reuse)
- Rehash correctness (all keys accessible after resize)

---

## Performance Characteristics in Practice

| Metric | Separate Chaining | Open Addressing (Linear) |
|---|---|---|
| Memory per entry | Key + value + pointer (+ allocator overhead) | Key + value + state byte |
| Cache behaviour | Poor (pointer chasing) | Good (sequential scan) |
| High load (α > 0.8) | Graceful degradation | Rapid degradation |
| Deletion | O(1), no side effects | Tombstones accumulate |
| Implementation complexity | Low | Medium |

In benchmarks on modern hardware, open addressing typically wins at low-to-medium load factors (α < 0.7) because the cache advantage dominates. Chaining wins at very high load factors because clustering doesn't apply.

---

## In Real Systems

**Redis** is fundamentally a giant hash table — its global keyspace is a `dict` structure using separate chaining with incremental rehashing. Every `GET`/`SET` is O(1).

**PostgreSQL** hash indexes use extendible hashing on disk pages. They support only equality predicates but are faster than B-tree indexes for pure point lookups on high-cardinality columns.

**DuckDB / Spark hash joins**: when joining two relations, build a hash table on the smaller one (the "build side"), then probe it with every row from the larger one (the "probe side"). Cost: O(n + m) vs O(n × m) for nested-loop join. The build phase is exactly inserting *n* rows into a hash table; the probe phase is *m* lookups.

**Robin Hood / Swiss Table** (Abseil): Google's `absl::flat_hash_map` uses SIMD-accelerated probing — it stores the top 7 bits of each key's hash in a separate metadata array, and uses SSE2/NEON to check 16 slots simultaneously. A single cache line holds 16 metadata bytes, so one load + one SIMD compare resolves most lookups without touching the actual key/value data.

---

## Why Not Always Use Hash Indexes?

Hash indexes cannot answer range queries (`WHERE age BETWEEN 20 AND 30`). The hash function maps nearby keys to completely different buckets — there is no spatial locality. For ordered access, a B+ tree is required.

The fundamental tradeoff:

| Structure | Point lookup | Range scan | Ordering |
|---|---|---|---|
| Hash index | **O(1)** expected | ✗ impossible | ✗ |
| B+ tree | O(log n) | O(log n + k) | ✓ |
| Skip list | O(log n) expected | O(log n + k) | ✓ |

No single structure dominates. Query workload determines the right choice — which is why real databases support multiple index types on the same column.

---

## What's Next

**Week 2: LSM Trees** — the write-optimised storage structure behind RocksDB, Cassandra, and LevelDB.

The core problem: random writes to a B-tree on spinning disk cause random I/O, limited to ~100–200 IOPS. An LSM tree turns random writes into sequential writes by batching mutations in a memory buffer (MemTable) and flushing to disk as immutable sorted runs. Sequential I/O on the same hardware reaches 100–200 MB/s — three orders of magnitude faster.

The cost is read amplification: a lookup must check the MemTable plus potentially many on-disk levels. Bloom filters make this tractable.

---

*Code: [github.com/anocaj/db-internals/pull/8](https://github.com/anocaj/db-internals/pull/8)*
