---
layout: ../layouts/AboutLayout.astro
title: "About"
---

Hi, I'm **Alban** — a software engineer interested in how databases work under the hood, and increasingly in how those same ideas reappear inside modern AI systems.

This blog is a learning journal with two parallel series.

---

## Series 1: DB Internals from Scratch

Implementing core database algorithms in C++17, from first principles. Not just *how* they work — *why* they exist, what problem they solve, and what tradeoffs they make.

All code lives in [anocaj/db-internals](https://github.com/anocaj/db-internals).

| Week | Topic | Post | Code |
|------|-------|------|------|
| 1 | Hash Index | [Read →](/blog/posts/week-1-hash-index/) | ✅ Done |
| 2 | LSM Tree | Coming soon | — |
| 3 | Columnar Storage | Coming soon | — |
| 4 | Buffer Pool Manager | Coming soon | — |
| 5 | Query Engine | Coming soon | — |
| 6 | Concurrency / MVCC | Coming soon | — |
| 7 | WAL & Recovery | Coming soon | — |

---

## Series 2: DB Internals → AI Systems

Every week, a companion post mapping that week's data structure to where it shows up — often unacknowledged — in LLMs and AI agent systems. The field is at the stage databases were in the 1970s. The engineers who understand both layers will build better systems.

| Week | DB Concept | AI/LLM Equivalent | Post |
|------|------------|-------------------|------|
| 1 | Hash Index | KV Cache, MQA/GQA, PagedAttention | [Read →](/blog/posts/llm-1-kv-cache/) |
| 2 | LSM Tree | Vector store ingestion, HNSW write buffering | Coming soon |
| 3 | Columnar Storage | Tensor memory layouts, FlashAttention tiling | Coming soon |
| 4 | Buffer Pool Manager | PagedAttention, context window eviction | Coming soon |
| 5 | Query Engine | RAG pipelines, retrieval as hash join | Coming soon |
| 6 | MVCC | Multi-agent state consistency | Coming soon |
| 7 | WAL & Recovery | Agent action logs, rollback, LangSmith traces | Coming soon |

---

## Other Projects

- [mini-data-cloud](https://github.com/anocaj/mini-data-cloud) — distributed query engine with Java, Apache Arrow, and Iceberg

## Get in Touch

Find me on [GitHub](https://github.com/anocaj).
