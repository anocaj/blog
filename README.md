# DB Internals Blog

A minimal technical blog documenting the implementation of core database algorithms from first principles — written in C++17, explained clearly.

**Live:** https://anocaj.github.io/blog/

## What's Here

Weekly deep-dives into the structures powering modern databases:

- Hash Indexes
- LSM Trees
- Columnar Storage + Compression
- Buffer Pool Manager
- Query Engine (SQL → AST → execution)
- Concurrency / MVCC
- WAL & Crash Recovery

Each post covers the problem, the insight, the implementation, and how real databases (PostgreSQL, RocksDB, ClickHouse) use it.

## Related

- [anocaj/db-internals](https://github.com/anocaj/db-internals) — the C++ implementations
- [anocaj/mini-data-cloud](https://github.com/anocaj/mini-data-cloud) — distributed query engine

## Stack

Built with [Astro](https://astro.build/) + [AstroPaper](https://github.com/satnaing/astro-paper). Deployed to GitHub Pages on every push to `main`.

## Local Dev

```bash
npm install
npm run dev
```
