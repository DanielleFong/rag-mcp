# rag-tui - Terminal User Interface

The primary human interface for interacting with and visualizing the RAG system state. Built with Ratatui for cross-platform terminal rendering.

## Overview

| Attribute | Value |
|-----------|-------|
| Crate | `rag-tui` |
| Framework | Ratatui 0.26+ |
| Backend | Crossterm |
| Async Runtime | Tokio |
| Est. Lines | ~3,000 |
| Confidence | HIGH |

## Design Philosophy

1. **Information density**: Show maximum useful information without clutter
2. **Keyboard-first**: All operations accessible via keyboard shortcuts
3. **Real-time updates**: Live refresh of statistics and sync status
4. **Claude-friendly**: Built-in screenshot capability for AI assessment
5. **Discoverability**: Help overlays, breadcrumbs, status hints

---

## Screenshot Capability

### Purpose

Enable Claude Code to capture and assess the visual state of the application, facilitating AI-assisted debugging, documentation, and workflow automation.

### Implementation

```rust
/// Screenshot output format
enum ScreenshotFormat {
    Png,           // Rendered terminal screenshot (via terminal-screenshot crate)
    Ansi,          // Raw ANSI escape sequences (text file)
    Json,          // Structured state dump (machine-readable)
    Svg,           // Vector rendering (documentation)
}

/// Screenshot configuration
struct ScreenshotConfig {
    output_dir: PathBuf,           // Default: ~/.local/share/rag-mcp/screenshots
    format: ScreenshotFormat,      // Default: Png
    include_timestamp: bool,       // Default: true
    auto_capture_on_error: bool,   // Default: true
}

/// Screenshot command (available from any view)
/// Hotkey: F12 or Ctrl+Shift+S
fn capture_screenshot(app: &App, config: &ScreenshotConfig) -> Result<PathBuf> {
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!("rag_tui_{}_{}.{}",
        app.current_view.name(),
        timestamp,
        config.format.extension()
    );

    let path = config.output_dir.join(&filename);

    match config.format {
        ScreenshotFormat::Png => {
            // Use terminal-screenshot or capture via platform APIs
            capture_terminal_png(&path)?;
        }
        ScreenshotFormat::Ansi => {
            // Render current frame to ANSI and save
            let ansi = app.render_to_ansi()?;
            std::fs::write(&path, ansi)?;
        }
        ScreenshotFormat::Json => {
            // Dump structured state
            let state = app.to_state_snapshot()?;
            std::fs::write(&path, serde_json::to_string_pretty(&state)?)?;
        }
        ScreenshotFormat::Svg => {
            // Use vt100 + resvg for vector output
            render_to_svg(app, &path)?;
        }
    }

    // Copy path to clipboard for easy pasting
    copy_to_clipboard(&path.display().to_string())?;

    Ok(path)
}

/// State snapshot for JSON export
#[derive(Serialize)]
struct StateSnapshot {
    timestamp: String,
    view: String,
    collections: Vec<CollectionStats>,
    selected_item: Option<SelectedItem>,
    search_query: Option<String>,
    search_results: Option<Vec<SearchResultSummary>>,
    sync_status: SyncStatus,
    errors: Vec<ErrorEntry>,
    performance: PerformanceMetrics,
}
```

### Claude Code Integration

```bash
# Capture screenshot and return path
rag-tui --screenshot --format png --output /tmp/screenshot.png

# Capture state dump
rag-tui --screenshot --format json --output /tmp/state.json

# Capture and wait for specific view
rag-tui --screenshot --view search --wait-for-render
```

**Confidence: HIGH** - Screenshot is achievable; multiple format options cover different use cases.

---

## Screen Layout

### Main Dashboard (Default View)

```
┌─rag-mcp ─────────────────────────────────────────────────────────────────────┐
│ [D]ashboard  [C]ollections  [S]earch  [I]ngest  [Y]nc  [L]ogs  [H]elp  [Q]uit │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Collections ──────────────────┐ ┌─ System Status ────────────────────────┐│
│ │ NAME          DOCS    CHUNKS   │ │ Store:    /home/user/.local/rag.db    ││
│ │ ─────────────────────────────  │ │ Size:     1.2 GB                       ││
│ │ > code        1,234   12,450   │ │ Vectors:  125,000                      ││
│ │   docs          456    4,230   │ │ Model:    nomic-embed-text-v1.5        ││
│ │   chat          789    8,910   │ │ Uptime:   2h 34m                       ││
│ │   papers        123    2,100   │ │                                        ││
│ │                                │ │ Memory:   ████████░░ 1.8/4.0 GB        ││
│ │                                │ │ CPU:      ██░░░░░░░░ 12%               ││
│ └────────────────────────────────┘ └────────────────────────────────────────┘│
│ ┌─ Recent Activity ──────────────────────────────────────────────────────────┐│
│ │ 14:32:01  INGEST   +3 docs to 'code' (src/lib.rs, src/main.rs, Cargo.toml)││
│ │ 14:31:45  SEARCH   "error handling" → 8 results (45ms)                    ││
│ │ 14:30:12  SYNC     peer-alpha: pulled 12 changes, pushed 5                ││
│ │ 14:28:33  INGEST   +1 doc to 'docs' (README.md)                           ││
│ │ 14:25:00  DELETE   removed 'old-notes.txt' from 'docs'                    ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Sync Peers ─────────────────────────────────────────────────────────────┐ │
│ │ PEER          ENDPOINT                    LAST SYNC    STATUS            │ │
│ │ peer-alpha    http://192.168.1.10:8765    2m ago       ● Connected       │ │
│ │ peer-beta     http://192.168.1.11:8765    15m ago      ○ Idle            │ │
│ │ peer-gamma    http://192.168.1.12:8765    1h ago       ✗ Unreachable     │ │
│ └───────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ F12:Screenshot  ?:Help  /:Search  Tab:Focus  q:Quit                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Collections Browser

```
┌─rag-mcp ─ Collections ───────────────────────────────────────────────────────┐
│ [D]ashboard  [C]ollections  [S]earch  [I]ngest  [Y]nc  [L]ogs  [H]elp  [Q]uit │
├─────────────────────────────────┬────────────────────────────────────────────┤
│ ┌─ Collections ───────────────┐ │ ┌─ Collection: code ─────────────────────┐ │
│ │ > code        [1,234 docs]  │ │ │ Created:    2024-01-15 10:30:00        │ │
│ │   docs        [  456 docs]  │ │ │ Documents:  1,234                      │ │
│ │   chat        [  789 docs]  │ │ │ Chunks:     12,450                     │ │
│ │   papers      [  123 docs]  │ │ │ Vectors:    12,450                     │ │
│ │                             │ │ │ Size:       450 MB                     │ │
│ │ [n]ew  [d]elete  [r]ename   │ │ │                                        │ │
│ └─────────────────────────────┘ │ │ Content Types:                         │ │
│                                 │ │   Rust:       45%  ████████░░░         │ │
│ ┌─ Documents ─────────────────┐ │ │   Python:     30%  ██████░░░░░         │ │
│ │ SOURCE                  SZ  │ │ │   TypeScript: 20%  ████░░░░░░░         │ │
│ │ ─────────────────────────── │ │ │   Other:       5%  █░░░░░░░░░░         │ │
│ │ > src/lib.rs          4.2KB │ │ │                                        │ │
│ │   src/main.rs         1.8KB │ │ │ Avg Chunk Size: 312 tokens             │ │
│ │   src/query.rs        3.1KB │ │ │ Avg Doc Size:   10.1 chunks            │ │
│ │   src/store.rs        5.4KB │ │ └────────────────────────────────────────┘ │
│ │   Cargo.toml          0.8KB │ │                                            │
│ │   README.md           2.1KB │ │ ┌─ Document Preview ────────────────────┐  │
│ │                             │ │ │ // src/lib.rs                         │  │
│ │ Page 1/124  ↑↓:Nav  Enter  │ │ │ //! RAG system core library            │  │
│ └─────────────────────────────┘ │ │ //!                                    │  │
│                                 │ │ //! This crate provides the main...   │  │
│                                 │ │                                        │  │
│                                 │ │ pub mod store;                         │  │
│                                 │ │ pub mod query;                         │  │
│                                 │ │ pub mod embed;                         │  │
│                                 │ │                                        │  │
│                                 │ │ [v]iew full  [e]dit meta  [x]delete   │  │
│                                 │ └────────────────────────────────────────┘  │
├─────────────────────────────────┴────────────────────────────────────────────┤
│ ←→:Panes  ↑↓:Select  Enter:Open  /:Filter  Esc:Back                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Search Interface

```
┌─rag-mcp ─ Search ────────────────────────────────────────────────────────────┐
│ [D]ashboard  [C]ollections  [S]earch  [I]ngest  [Y]nc  [L]ogs  [H]elp  [Q]uit │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Query ────────────────────────────────────────────────────────────────────┐│
│ │ > error handling in async functions_                                      ││
│ │                                                                            ││
│ │ Collection: [All ▼]  Mode: [Hybrid ▼]  Top-K: [10 ▼]  Expand: [✓]         ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Results (8 matches in 47ms) ──────────────────────────────────────────────┐│
│ │                                                                            ││
│ │ 1. src/query.rs:142-168  [code]                          Score: 0.847     ││
│ │ ┌──────────────────────────────────────────────────────────────────────┐  ││
│ │ │ async fn handle_query_error(&self, err: QueryError) -> Result<()> {  │  ││
│ │ │     match err {                                                       │  ││
│ │ │         QueryError::Timeout => {                                      │  ││
│ │ │             // Retry with exponential backoff                         │  ││
│ │ │             self.retry_with_backoff().await?;                         │  ││
│ │ │         }                                                             │  ││
│ │ │         QueryError::NotFound => return Ok(()),                        │  ││
│ │ │         _ => return Err(err.into()),                                  │  ││
│ │ │     }                                                                 │  ││
│ │ │ }                                                                     │  ││
│ │ └──────────────────────────────────────────────────────────────────────┘  ││
│ │                                                                            ││
│ │ 2. docs/error-handling.md:45-78  [docs]                   Score: 0.812    ││
│ │ ┌──────────────────────────────────────────────────────────────────────┐  ││
│ │ │ ## Error Handling in Async Code                                       │  ││
│ │ │                                                                       │  ││
│ │ │ When working with async functions, error handling requires special    │  ││
│ │ │ consideration. The `?` operator works seamlessly, but you need to...  │  ││
│ │ └──────────────────────────────────────────────────────────────────────┘  ││
│ │                                                                            ││
│ │ [more results below - scroll with j/k]                                    ││
│ └────────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ Enter:Search  Tab:Options  j/k:Navigate  o:Open  y:Copy  Esc:Clear           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Ingest Interface

```
┌─rag-mcp ─ Ingest ────────────────────────────────────────────────────────────┐
│ [D]ashboard  [C]ollections  [S]earch  [I]ngest  [Y]nc  [L]ogs  [H]elp  [Q]uit │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Source ───────────────────────────────────────────────────────────────────┐│
│ │ URI: file:///home/user/projects/myapp/src/_                               ││
│ │                                                                            ││
│ │ [Tab to autocomplete]                    Recent: ~/projects  ~/documents  ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Options ──────────────────────────────────────────────────────────────────┐│
│ │ Collection:    [code ▼]                                                   ││
│ │ Content Type:  [Auto-detect ▼]                                            ││
│ │ Recursive:     [✓] Include subdirectories                                 ││
│ │ Patterns:      *.rs, *.py, *.ts                                           ││
│ │ Ignore:        target/, node_modules/, .git/                              ││
│ │ Watch:         [ ] Continue watching for changes                          ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Preview ──────────────────────────────────────────────────────────────────┐│
│ │ Files to ingest: 47                                                        ││
│ │                                                                            ││
│ │   src/lib.rs              4.2 KB   Rust                                   ││
│ │   src/main.rs             1.8 KB   Rust                                   ││
│ │   src/query/mod.rs        3.1 KB   Rust                                   ││
│ │   src/query/hybrid.rs     2.4 KB   Rust                                   ││
│ │   src/store/sqlite.rs     5.4 KB   Rust                                   ││
│ │   ...and 42 more files                                                    ││
│ │                                                                            ││
│ │ Estimated: ~470 chunks, ~1.2 MB embeddings                                ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Progress ─────────────────────────────────────────────────────────────────┐│
│ │ Ready to ingest. Press Enter to start.                                    ││
│ └────────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ Enter:Start  Tab:Fields  Ctrl+C:Cancel                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Progress View (During Ingest)

```
┌─rag-mcp ─ Ingest ────────────────────────────────────────────────────────────┐
│ [D]ashboard  [C]ollections  [S]earch  [I]ngest  [Y]nc  [L]ogs  [H]elp  [Q]uit │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Ingesting: /home/user/projects/myapp/src ─────────────────────────────────┐│
│ │                                                                            ││
│ │ Overall Progress                                                           ││
│ │ ████████████████████░░░░░░░░░░░░░░░░░░░░  47%  (22/47 files)              ││
│ │                                                                            ││
│ │ Current File: src/query/hybrid.rs                                         ││
│ │ ████████████████████████████░░░░░░░░░░░░  68%  Embedding...               ││
│ │                                                                            ││
│ │ ┌─ Stage Breakdown ──────────────────────────────────────────────────────┐││
│ │ │ Stage        Done     Total    Rate         ETA                        │││
│ │ │ ──────────────────────────────────────────────────────────────────────│││
│ │ │ Loading      22       47       12.3/s       2s                         │││
│ │ │ Chunking     21       47       8.5/s        3s                         │││
│ │ │ Embedding    18       210      45.2/s       4s                         │││
│ │ │ Storing      18       210      102.4/s      2s                         │││
│ │ └────────────────────────────────────────────────────────────────────────┘││
│ │                                                                            ││
│ │ Statistics:                                                                ││
│ │   Chunks created:  210                                                    ││
│ │   Tokens processed: 67,340                                                ││
│ │   Embeddings:       18 batches                                            ││
│ │   Elapsed:          8.4s                                                  ││
│ │   Errors:           0                                                     ││
│ │                                                                            ││
│ └────────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ Ctrl+C:Cancel  (ingest will complete current batch before stopping)          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Sync Manager

```
┌─rag-mcp ─ Sync ──────────────────────────────────────────────────────────────┐
│ [D]ashboard  [C]ollections  [S]earch  [I]ngest  [Y]nc  [L]ogs  [H]elp  [Q]uit │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Node Identity ────────────────────────────────────────────────────────────┐│
│ │ Node ID:      node-7b3f                                                   ││
│ │ Endpoint:     http://0.0.0.0:8765                                         ││
│ │ Current HLC:  00018d9e8f3c00000001000a                                    ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Peers ────────────────────────────────────────────────────────────────────┐│
│ │ PEER         ENDPOINT                 LAST SYNC   STATUS      ACTIONS     ││
│ │ ────────────────────────────────────────────────────────────────────────  ││
│ │ > peer-alpha http://192.168.1.10:8765  2m ago     ● Connected  [s]ync     ││
│ │   peer-beta  http://192.168.1.11:8765  15m ago    ○ Idle       [s]ync     ││
│ │   peer-gamma http://192.168.1.12:8765  1h ago     ✗ Error      [r]etry    ││
│ │                                                                            ││
│ │ [a]dd peer  [e]dit  [d]elete                                              ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Peer Details: peer-alpha ─────────────────────────────────────────────────┐│
│ │ Added:          2024-01-10 14:30:00                                       ││
│ │ Total Syncs:    342                                                       ││
│ │ Last Pulled:    12 changes                                                ││
│ │ Last Pushed:    5 changes                                                 ││
│ │ Conflicts:      2 (resolved by LWW)                                       ││
│ │                                                                            ││
│ │ ┌─ Sync History ────────────────────────────────────────────────────────┐ ││
│ │ │ TIME       DIR    CHANGES  STATUS                                     │ ││
│ │ │ 14:30:12   pull   12       ✓ success                                  │ ││
│ │ │ 14:30:13   push   5        ✓ success                                  │ ││
│ │ │ 14:15:00   pull   8        ✓ success                                  │ ││
│ │ │ 14:00:45   pull   3        ⚠ partial (1 conflict)                     │ ││
│ │ └───────────────────────────────────────────────────────────────────────┘ ││
│ └────────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ Enter:Sync Now  Tab:Panes  a:Add Peer  Esc:Back                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Log Viewer

```
┌─rag-mcp ─ Logs ──────────────────────────────────────────────────────────────┐
│ [D]ashboard  [C]ollections  [S]earch  [I]ngest  [Y]nc  [L]ogs  [H]elp  [Q]uit │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Filters ──────────────────────────────────────────────────────────────────┐│
│ │ Level: [All ▼]  Component: [All ▼]  Search: _______________  [x] Autoscroll│
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Log Stream ───────────────────────────────────────────────────────────────┐│
│ │ 14:32:01.234 INFO  [ingest] Starting batch ingest: 3 files                ││
│ │ 14:32:01.245 DEBUG [chunk]  Chunking src/lib.rs (Rust, AST-aware)         ││
│ │ 14:32:01.267 DEBUG [chunk]  Generated 12 chunks (avg 284 tokens)          ││
│ │ 14:32:01.268 DEBUG [embed]  Embedding batch 1/1 (12 texts)                ││
│ │ 14:32:01.489 DEBUG [embed]  Batch complete: 221ms                         ││
│ │ 14:32:01.490 DEBUG [store]  Inserting document: src/lib.rs                ││
│ │ 14:32:01.512 DEBUG [store]  Inserted 12 chunks, 12 embeddings             ││
│ │ 14:32:01.513 INFO  [ingest] Completed: src/lib.rs (12 chunks, 512ms)      ││
│ │ 14:32:01.514 DEBUG [chunk]  Chunking src/main.rs (Rust, AST-aware)        ││
│ │ 14:32:01.523 DEBUG [chunk]  Generated 5 chunks (avg 198 tokens)           ││
│ │ 14:32:01.524 DEBUG [embed]  Embedding batch 1/1 (5 texts)                 ││
│ │ 14:32:01.612 DEBUG [embed]  Batch complete: 88ms                          ││
│ │ 14:32:01.613 INFO  [ingest] Completed: src/main.rs (5 chunks, 99ms)       ││
│ │ 14:32:01.614 WARN  [sync]   Peer peer-gamma unreachable: connection refused│
│ │ 14:32:01.615 DEBUG [chunk]  Chunking Cargo.toml (TOML, record-based)      ││
│ │ 14:32:01.618 DEBUG [chunk]  Generated 2 chunks (avg 156 tokens)           ││
│ │ 14:32:01.690 INFO  [ingest] Batch complete: 3 files, 19 chunks, 456ms     ││
│ │                                                                            ││
│ │ ─── END OF LOG ─── (watching for new entries)                             ││
│ └────────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ ↑↓:Scroll  g/G:Top/Bottom  /:Search  f:Filter  c:Clear  Esc:Back             │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Chunk Inspector (Detail View)

```
┌─rag-mcp ─ Chunk Inspector ───────────────────────────────────────────────────┐
│ ← Back to Search Results                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Chunk: 01HRE4KXQN... ─────────────────────────────────────────────────────┐│
│ │ Document:   src/query/hybrid.rs                                           ││
│ │ Collection: code                                                          ││
│ │ Index:      3 of 8 in document                                            ││
│ │ Tokens:     312                                                           ││
│ │ Characters: 1,247                                                         ││
│ │ Lines:      142-168                                                       ││
│ │ Strategy:   AST-aware (function_item)                                     ││
│ │ Node Type:  function_item                                                 ││
│ │ Node Name:  handle_query_error                                            ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Content ──────────────────────────────────────────────────────────────────┐│
│ │ 142 │ async fn handle_query_error(&self, err: QueryError) -> Result<()> { ││
│ │ 143 │     match err {                                                      ││
│ │ 144 │         QueryError::Timeout => {                                     ││
│ │ 145 │             // Retry with exponential backoff                        ││
│ │ 146 │             self.retry_with_backoff().await?;                        ││
│ │ 147 │         }                                                            ││
│ │ 148 │         QueryError::NotFound => return Ok(()),                       ││
│ │ 149 │         _ => return Err(err.into()),                                 ││
│ │ 150 │     }                                                                ││
│ │ 151 │ }                                                                    ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Embedding (first 16 dims) ────────────────────────────────────────────────┐│
│ │ [0.023, -0.156, 0.089, 0.234, -0.012, 0.178, -0.045, 0.301, ...]          ││
│ │ Norm: 1.0000  Model: nomic-embed-text-v1.5                                ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ Adjacent Chunks ──────────────────────────────────────────────────────────┐│
│ │ ← Prev: validate_query (lines 128-141)                                    ││
│ │ → Next: execute_search (lines 170-195)                                    ││
│ └────────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ [p]rev chunk  [n]ext chunk  [o]pen in editor  [y]copy  Esc:Back              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Embedding Visualizer

```
┌─rag-mcp ─ Embedding Visualizer ──────────────────────────────────────────────┐
│ ← Back                                                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ ┌─ Query Embedding vs Top Results ───────────────────────────────────────────┐│
│ │                                                                            ││
│ │ Similarity Heatmap (cosine):                                               ││
│ │                                                                            ││
│ │ Query    ████████████████████████████████████████  1.000 (self)           ││
│ │ Result 1 ██████████████████████████████████░░░░░░  0.847                  ││
│ │ Result 2 █████████████████████████████████░░░░░░░  0.812                  ││
│ │ Result 3 ████████████████████████████░░░░░░░░░░░░  0.734                  ││
│ │ Result 4 ███████████████████████████░░░░░░░░░░░░░  0.721                  ││
│ │ Result 5 ██████████████████████░░░░░░░░░░░░░░░░░░  0.654                  ││
│ │                                                                            ││
│ └────────────────────────────────────────────────────────────────────────────┘│
│ ┌─ t-SNE Projection (2D) ────────────────────────────────────────────────────┐│
│ │                                                                            ││
│ │                    ·2                                                      ││
│ │              ·3    ·1                                                      ││
│ │                  ★Q                                                        ││
│ │            ·4                                                              ││
│ │                      ·5                                                    ││
│ │                                                                            ││
│ │                                       ·8                                   ││
│ │                           ·7    ·6                                         ││
│ │                                                                            ││
│ │ ★ = Query   · = Result   [Number = rank]                                   ││
│ └────────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ ↑↓:Select Result  Enter:Inspect  Esc:Back                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Keyboard Shortcuts

### Global Shortcuts

| Key | Action |
|-----|--------|
| `D` | Go to Dashboard |
| `C` | Go to Collections |
| `S` | Go to Search |
| `I` | Go to Ingest |
| `Y` | Go to Sync |
| `L` | Go to Logs |
| `H` or `?` | Show Help |
| `Q` or `Ctrl+C` | Quit |
| `F12` or `Ctrl+Shift+S` | Take Screenshot |
| `/` | Focus search (context-dependent) |
| `Tab` | Next pane/field |
| `Shift+Tab` | Previous pane/field |
| `Esc` | Back/Cancel |

### Navigation

| Key | Action |
|-----|--------|
| `j` or `↓` | Move down |
| `k` or `↑` | Move up |
| `h` or `←` | Move left / Previous pane |
| `l` or `→` | Move right / Next pane |
| `g` | Go to top |
| `G` | Go to bottom |
| `Ctrl+D` | Page down |
| `Ctrl+U` | Page up |
| `Enter` | Select/Confirm |

### Search View

| Key | Action |
|-----|--------|
| `Enter` | Execute search |
| `Tab` | Cycle through options |
| `o` | Open selected result in editor |
| `y` | Copy selected result to clipboard |
| `v` | View full document |
| `e` | Inspect embedding |

### Collections View

| Key | Action |
|-----|--------|
| `n` | New collection |
| `d` | Delete selected |
| `r` | Rename selected |
| `Enter` | Expand/View details |

### Ingest View

| Key | Action |
|-----|--------|
| `Enter` | Start ingest |
| `Tab` | Next field |
| `Ctrl+C` | Cancel ingest |

### Sync View

| Key | Action |
|-----|--------|
| `s` | Sync with selected peer |
| `a` | Add new peer |
| `e` | Edit peer |
| `d` | Delete peer |
| `r` | Retry failed sync |

---

## Architecture

### Component Structure

```rust
// Main application state
struct App {
    // Current view
    current_view: View,
    view_stack: Vec<View>,  // For back navigation

    // Shared state
    store: Arc<Store>,
    query_engine: Arc<QueryEngine>,
    embedder: Arc<Embedder>,
    sync_manager: Arc<SyncManager>,

    // UI state
    collections_state: CollectionsState,
    search_state: SearchState,
    ingest_state: IngestState,
    sync_state: SyncState,
    logs_state: LogsState,

    // Config
    config: TuiConfig,
    screenshot_config: ScreenshotConfig,

    // Event handling
    event_rx: mpsc::Receiver<AppEvent>,
}

enum View {
    Dashboard,
    Collections,
    Search,
    Ingest,
    Sync,
    Logs,
    Help,
    ChunkInspector(Ulid),
    EmbeddingVisualizer,
}

// Event types
enum AppEvent {
    Key(KeyEvent),
    Tick,
    IngestProgress(IngestProgress),
    SyncUpdate(SyncUpdate),
    LogEntry(LogEntry),
    SearchComplete(SearchResults),
    Error(RagError),
}
```

### Rendering Pipeline

```rust
fn render(app: &App, frame: &mut Frame) {
    // 1. Render chrome (tabs, status bar)
    render_chrome(frame, app);

    // 2. Render current view
    match &app.current_view {
        View::Dashboard => render_dashboard(frame, app),
        View::Collections => render_collections(frame, app),
        View::Search => render_search(frame, app),
        View::Ingest => render_ingest(frame, app),
        View::Sync => render_sync(frame, app),
        View::Logs => render_logs(frame, app),
        View::Help => render_help(frame, app),
        View::ChunkInspector(id) => render_chunk_inspector(frame, app, id),
        View::EmbeddingVisualizer => render_embedding_viz(frame, app),
    }

    // 3. Render overlays (help, errors, confirmations)
    if app.show_help_overlay {
        render_help_overlay(frame, app);
    }

    if let Some(error) = &app.current_error {
        render_error_toast(frame, error);
    }
}
```

### Event Loop

```rust
async fn run(mut app: App) -> Result<()> {
    let mut terminal = setup_terminal()?;

    loop {
        // Render
        terminal.draw(|f| render(&app, f))?;

        // Handle events
        tokio::select! {
            // Keyboard input
            Some(event) = app.event_rx.recv() => {
                match event {
                    AppEvent::Key(key) => {
                        if handle_key(&mut app, key)? == Action::Quit {
                            break;
                        }
                    }
                    AppEvent::Tick => {
                        // Update animations, refresh stats
                        app.tick().await?;
                    }
                    AppEvent::IngestProgress(p) => {
                        app.ingest_state.update_progress(p);
                    }
                    AppEvent::SyncUpdate(u) => {
                        app.sync_state.update(u);
                    }
                    AppEvent::LogEntry(e) => {
                        app.logs_state.push(e);
                    }
                    AppEvent::SearchComplete(r) => {
                        app.search_state.set_results(r);
                    }
                    AppEvent::Error(e) => {
                        app.show_error(e);
                    }
                }
            }
        }
    }

    restore_terminal()?;
    Ok(())
}
```

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ratatui` | 0.26+ | TUI framework |
| `crossterm` | 0.27+ | Terminal backend |
| `tokio` | 1.0+ | Async runtime |
| `tracing` | 0.1+ | Logging |
| `tracing-subscriber` | 0.3+ | Log formatting |
| `arboard` | 3.0+ | Clipboard |
| `terminal-screenshot` | 0.1+ | PNG capture |
| `vt100` | 0.15+ | Terminal state for SVG |
| `clap` | 4.0+ | CLI arguments |
| `dirs` | 5.0+ | Platform directories |

---

## Configuration

```toml
# ~/.config/rag-mcp/tui.toml

[general]
# Refresh rate in milliseconds
tick_rate = 250

# Default view on startup
default_view = "dashboard"

# Color theme
theme = "dark"  # "dark", "light", "nord", "dracula"

[screenshot]
# Output directory for screenshots
output_dir = "~/.local/share/rag-mcp/screenshots"

# Default format
format = "png"  # "png", "ansi", "json", "svg"

# Auto-capture on errors
auto_capture_on_error = true

[keybindings]
# Override default keybindings
# quit = "ctrl+q"
# screenshot = "f12"

[display]
# Show line numbers in code views
line_numbers = true

# Syntax highlighting
syntax_highlighting = true

# Max log entries to keep in memory
max_log_entries = 10000

# Truncate long lines at this width (0 = no truncation)
max_line_width = 0
```

---

## Confidence Assessment

| Feature | Confidence | Notes |
|---------|------------|-------|
| Core TUI framework | HIGH | Ratatui is mature and well-documented |
| Screenshot (PNG) | MEDIUM | Platform-dependent; may need fallbacks |
| Screenshot (JSON) | HIGH | Just serialization |
| Screenshot (SVG) | MEDIUM | Requires vt100 parsing |
| Real-time updates | HIGH | Standard tokio channel pattern |
| Embedding visualizer | MEDIUM | t-SNE requires extra deps, may simplify |
| Keyboard handling | HIGH | Well-understood patterns |
| Syntax highlighting | MEDIUM | May use syntect; adds complexity |

---

## Future Enhancements

1. **Mouse support**: Click to select, scroll
2. **Themes**: User-defined color schemes
3. **Plugins**: Lua scripting for custom views
4. **Remote mode**: Connect to remote RAG instance
5. **Export**: Export search results to various formats
6. **Diff view**: Compare document versions
7. **Batch operations**: Multi-select for bulk actions
