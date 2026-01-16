# rag-mcp

A local-first, Rust-based Retrieval-Augmented Generation (RAG) system with MCP (Model Context Protocol) integration. Designed for ~100K documents with sub-second query latency and eventual consistency across multiple nodes.

## Status: Design Phase

This repository contains the complete design specification for a production-ready RAG system. Implementation has not yet begun.

## Quick Links

| Document | Description |
|----------|-------------|
| [DESIGN.md](./DESIGN.md) | Architecture deep-dive, data flow, component interactions |
| [API.md](./API.md) | MCP tool and resource specifications |
| [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) | Code patterns, gotchas, decision rationale |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common failure modes and mitigations |
| [TESTING.md](./TESTING.md) | Test strategy and verification plan |

## Crate Documentation

| Crate | Purpose | Doc |
|-------|---------|-----|
| rag-core | Domain types, traits, errors | [docs/rag-core.md](./docs/rag-core.md) |
| rag-store | SQLite + sqlite-vec storage | [docs/rag-store.md](./docs/rag-store.md) |
| rag-embed | Local embedding via ONNX | [docs/rag-embed.md](./docs/rag-embed.md) |
| rag-chunk | Content-aware chunking | [docs/rag-chunk.md](./docs/rag-chunk.md) |
| rag-query | Hybrid search + ranking | [docs/rag-query.md](./docs/rag-query.md) |
| rag-sync | Multi-node synchronization | [docs/rag-sync.md](./docs/rag-sync.md) |
| rag-mcp | MCP server implementation | [docs/rag-mcp.md](./docs/rag-mcp.md) |
| rag-cli | Command-line interface | [docs/rag-cli.md](./docs/rag-cli.md) |

## Design Goals

1. **Local-first**: No cloud dependencies. Runs entirely on local hardware.
2. **MCP-native**: First-class integration with Claude and other MCP clients.
3. **Hybrid search**: Combines semantic (vector) and lexical (BM25) retrieval.
4. **Multi-node sync**: Eventual consistency via HLC-based conflict resolution.
5. **Incremental updates**: Efficient handling of document changes.
6. **Content-aware**: Different chunking strategies for code, docs, and chat.

## Target Specifications

| Metric | Target | Confidence |
|--------|--------|------------|
| Document capacity | 100K documents | HIGH |
| Query latency (p50) | <200ms | HIGH |
| Query latency (p99) | <800ms | MEDIUM |
| Ingestion throughput | 100 docs/sec | MEDIUM |
| Embedding throughput | 50 chunks/sec | MEDIUM |
| Storage efficiency | ~2KB/chunk overhead | HIGH |
| Sync convergence | <5s on LAN | MEDIUM |

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Rust | Performance, safety, good SQLite/ONNX bindings |
| Vector store | sqlite-vec | Embedded, no external deps, good perf at scale |
| Keyword search | SQLite FTS5 | Integrated, mature, efficient |
| Embeddings | nomic-embed-text-v1.5 | Local, 768-dim, 8K context, good quality |
| Embedding runtime | ort (ONNX Runtime) | Cross-platform, hardware acceleration |
| MCP | rmcp or mcp-rust-sdk | Native Rust MCP implementation |
| Sync | Custom HTTP/REST | Simpler than custom protocols |

## Non-Goals

- **Not a general vector database**: Optimized specifically for RAG workloads
- **Not strongly consistent**: Eventual consistency is acceptable
- **Not a distributed system**: Multi-node is peer-to-peer, not clustered
- **Not cloud-native**: No Kubernetes, no cloud storage backends

## Prerequisites

For eventual implementation:

- Rust 1.75+ (for async traits)
- SQLite 3.45+ (for sqlite-vec compatibility)
- ONNX Runtime 1.16+
- ~2GB disk for embedding model
- ~4GB RAM recommended

## License

TBD - likely MIT or Apache-2.0
