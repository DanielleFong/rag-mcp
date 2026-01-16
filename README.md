# rag-mcp

**Version: 0.0.1 | Status: PRE-ALPHA**

A local-first, Rust-based Retrieval-Augmented Generation (RAG) system with MCP (Model Context Protocol) integration. Designed for ~100K documents with sub-second query latency.

## Current State

Core functionality is **implemented and tested** (47 tests passing):

- Hybrid search (FTS5 keyword + sqlite-vec vector)
- Recursive and adaptive chunking
- Reciprocal Rank Fusion for result ranking
- Full CLI for all operations
- MCP server logic (tools, not protocol transport)

**Not yet implemented:** Multi-node sync (rag-sync), TUI (rag-tui), actual ONNX model loading (uses MockEmbedder).

## Quick Start

```bash
# Build
cargo build --release

# Initialize database
rag init

# Create a collection
rag collection create mycode --description "My codebase"

# Ingest files
rag ingest ./src -c mycode -r

# Search
rag search "error handling" -c mycode

# Stats
rag stats
```

## Crates

| Crate | Status | Purpose |
|-------|--------|---------|
| rag-core | Done | Domain types, traits, HLC, errors |
| rag-store | Done | SQLite + FTS5 + sqlite-vec storage |
| rag-embed | Done | Embeddings (MockEmbedder for now) |
| rag-chunk | Done | Recursive and adaptive chunking |
| rag-query | Done | Hybrid search + RRF fusion |
| rag-mcp | Done | Server logic for MCP tools |
| rag-cli | Done | Full CLI binary |
| rag-sync | Stub | Multi-node sync (deferred) |
| rag-tui | Stub | Terminal UI (deferred) |

## Documentation

| Document | Description |
|----------|-------------|
| [DESIGN.md](./DESIGN.md) | Architecture, data flow, components |
| [API.md](./API.md) | MCP tool specifications |
| [IMPLEMENTATION_NOTES.md](./IMPLEMENTATION_NOTES.md) | Code patterns, decisions |
| [TESTING.md](./TESTING.md) | Test strategy |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common issues |

## Technology Stack

| Component | Choice |
|-----------|--------|
| Language | Rust 1.75+ |
| Vector store | sqlite-vec |
| Keyword search | SQLite FTS5 |
| Embeddings | ONNX Runtime (ort 2.0) |
| Fusion | Reciprocal Rank Fusion |

## Requirements

- Rust 1.75+
- SQLite 3.45+ (bundled)
- ~4GB RAM recommended

## Credits

Built by:
- **Claude** (Anthropic) - Claude Opus 4.5
- **Matt Parlmer**
- **Danielle Fong**

## License

MIT
