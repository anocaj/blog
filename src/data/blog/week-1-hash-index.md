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

A B+ tree finds a key in **O(log n)**. For a table with a billion rows, that's about 30 comparisons. Pretty fast.

But what if you don't need range queries? What if you just want: *"give me the value for this key, right now"*?

A hash index does it in **O(1)**. Not 30 comparisons — one.

That's the deal.

## The Core Insight

A **hash function** maps any key (a string, an integer, anything) to a number. That number becomes an array index. Instead of *searching* for your key, you *calculate* exactly where it lives.

```
search("user_42") → hash → 7 → array[7] → found it
```

It's like knowing which shelf a book is on without checking the catalogue.

The catch: two different keys can map to the same index. This is called a **collision**, and handling it is the entire design challenge.

## Two Ways to Handle Collisions

### 1. Separate Chaining

Each bucket holds a linked list. Collisions just append to the list.

```
bucket[7] → ["user_42": Alice] → ["user_99": Bob] → null
```

**Pros:** Simple. Works well at high load factors. Deletion is easy.  
**Cons:** Pointer chasing kills cache performance. Each node is a heap allocation.

Used by Java's `HashMap` and PostgreSQL hash joins.

### 2. Open Addressing (Linear Probing)

All data lives directly in the array. On collision, probe the next slot forward.

```
bucket[7] occupied → try bucket[8] → try bucket[9] → found empty → insert
```

**Pros:** Cache-friendly — you're scanning sequential memory, which modern CPUs love.  
**Cons:** "Clustering" — long runs of occupied slots slow down probes. Deletion is tricky.

**The deletion problem:** You can't just clear a slot. If you do, the probe chain breaks for keys that were inserted past that slot. The fix: mark deleted slots as **tombstones** instead of clearing them. Probing skips tombstones; insertion can reuse them.

Used by Python's `dict` and Go's `map`.

## The Hash Function: FNV-1a

A good hash function needs:
- **Uniform distribution** — keys spread evenly across buckets
- **Avalanche effect** — small input changes → completely different output
- **Speed** — called on every operation

We used **FNV-1a** (Fowler–Noll–Vo):

```cpp
uint64_t fnv1a(const void* data, size_t len) {
    uint64_t hash = 14695981039346656037ULL; // offset basis
    for (size_t i = 0; i < len; ++i) {
        hash ^= ((const uint8_t*)data)[i]; // XOR byte in
        hash *= 1099511628211ULL;           // multiply by prime
    }
    return hash;
}
```

XOR-then-multiply (vs multiply-then-XOR in FNV-1) gives better avalanche for small differences. Deterministic across compilers and platforms — important for reproducible tests.

## Dynamic Rehashing

As the table fills up, collision chains get longer and performance degrades. The fix: when the **load factor** (elements / buckets) exceeds a threshold, double the bucket count and redistribute everything.

```
load factor = size / bucket_count

if load_factor >= 0.75:
    new_capacity = old_capacity * 2
    rehash everything
```

Rehashing is O(n) but happens infrequently enough that it's **amortized O(1)** per insert — the same argument as dynamic arrays.

We use 0.75 for chaining (links absorb overflow) and 0.70 for open addressing (clustering gets bad faster).

## What We Built

Two implementations in C++17:

- **`ChainingHashIndex<K, V>`** — separate chaining, max load 0.75
- **`OpenAddressingHashIndex<K, V>`** — linear probing + tombstone deletion, max load 0.70
- Both support `insert`, `search`, `remove`, `size`, `empty`, `load_factor`
- Both rehash automatically
- 17 unit tests, correctness verified against `std::unordered_map`

## In the Real World

**Redis** is fundamentally a giant hash table. Every key-value lookup is O(1). The whole product is built on this one idea.

**PostgreSQL** supports hash indexes for equality queries (`WHERE id = 42`). They're faster than B-tree for pure point lookups but can't do range scans.

**Hash joins** in query engines (PostgreSQL, DuckDB, Spark) build a hash table on the smaller relation, then probe it with every row from the larger one. O(n+m) vs O(n×m) for nested loops.

## Why Not Always Use Hash Indexes?

Hash indexes can't answer range queries (`WHERE age BETWEEN 20 AND 30`). The hash function destroys ordering — keys close in value end up in completely different buckets. For ordered access, you need a B+ tree.

This is the fundamental tradeoff: **hash = O(1) point lookups, no ordering**. **B+ tree = O(log n) everything, ordering preserved**.

## What's Next

**Week 2: LSM Trees** — the write-optimized storage structure behind RocksDB, Cassandra, and LevelDB. The core insight: turn random writes into sequential writes by batching them in memory first.

---

*Code: [github.com/anocaj/db-internals/pull/8](https://github.com/anocaj/db-internals/pull/8)*
