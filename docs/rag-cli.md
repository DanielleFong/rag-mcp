# rag-cli

Command-line interface for manual RAG operations and debugging.

## Overview

| Attribute | Value |
|-----------|-------|
| Purpose | CLI for manual ops, debugging, administration |
| Dependencies | clap, tokio, rag-core, rag-query, rag-store |
| Est. Lines | ~500 |
| Confidence | HIGH |

This crate provides a comprehensive CLI for interacting with the RAG system outside of MCP.

---

## Module Structure

```
rag-cli/
├── src/
│   ├── main.rs          # Entry point and command dispatch
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── search.rs
│   │   ├── ingest.rs
│   │   ├── collections.rs
│   │   ├── documents.rs
│   │   ├── sync.rs
│   │   ├── stats.rs
│   │   └── repl.rs
│   └── output.rs        # Output formatting
└── Cargo.toml
```

---

## Command Structure

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rag")]
#[command(about = "Local RAG system CLI")]
#[command(version)]
struct Cli {
    /// Path to configuration file
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Database path (overrides config)
    #[arg(short, long, global = true)]
    database: Option<PathBuf>,

    /// Output format
    #[arg(short, long, global = true, default_value = "pretty")]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search the knowledge base
    Search(SearchArgs),

    /// Ingest documents
    Ingest(IngestArgs),

    /// Manage collections
    Collection(CollectionArgs),

    /// Manage documents
    Document(DocumentArgs),

    /// Sync with peers
    Sync(SyncArgs),

    /// Show statistics
    Stats(StatsArgs),

    /// Interactive REPL mode
    Repl,

    /// Start MCP server
    Serve(ServeArgs),

    /// Optimize database
    Optimize,

    /// Export/import data
    Export(ExportArgs),
    Import(ImportArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Pretty,
    Json,
    Csv,
    Quiet,
}
```

---

## Search Command

```rust
#[derive(Args)]
struct SearchArgs {
    /// Search query
    query: String,

    /// Collection to search
    #[arg(short, long)]
    collection: Option<String>,

    /// Number of results
    #[arg(short = 'k', long, default_value = "10")]
    top_k: usize,

    /// Search mode
    #[arg(short, long, default_value = "hybrid")]
    mode: SearchMode,

    /// Show detailed output
    #[arg(short, long)]
    verbose: bool,

    /// Show execution trace
    #[arg(long)]
    trace: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum SearchMode {
    Hybrid,
    Vector,
    Keyword,
}

async fn run_search(args: SearchArgs, ctx: &Context) -> Result<()> {
    let config = match args.mode {
        SearchMode::Hybrid => QueryConfig {
            final_k: args.top_k,
            hybrid_alpha: 0.5,
            enable_tracing: args.trace,
            ..Default::default()
        },
        SearchMode::Vector => QueryConfig {
            final_k: args.top_k,
            hybrid_alpha: 1.0,
            keyword_k: 0,
            enable_tracing: args.trace,
            ..Default::default()
        },
        SearchMode::Keyword => QueryConfig {
            final_k: args.top_k,
            hybrid_alpha: 0.0,
            vector_k: 0,
            enable_tracing: args.trace,
            ..Default::default()
        },
    };

    let results = ctx.query_engine
        .search_with_config(&args.query, args.collection.as_deref(), &config)
        .await?;

    // Output results
    match ctx.format {
        OutputFormat::Pretty => {
            println!("Query: {}", args.query);
            println!("Results: {}\n", results.results.len());

            for (i, r) in results.results.iter().enumerate() {
                println!("[{}] Score: {:.4}", i + 1, r.score);
                println!("    Source: {}", r.chunk.doc_id);
                println!("    Lines: {}-{}", r.chunk.metadata.line_start, r.chunk.metadata.line_end);

                if args.verbose {
                    println!("    Content:");
                    for line in r.chunk.content.lines().take(10) {
                        println!("      {}", line);
                    }
                    if r.chunk.content.lines().count() > 10 {
                        println!("      ...");
                    }
                } else {
                    let preview: String = r.chunk.content.chars().take(100).collect();
                    println!("    Preview: {}...", preview);
                }
                println!();
            }

            if args.trace {
                if let Some(trace) = &results.trace {
                    println!("\nExecution Trace:");
                    println!("{}", trace.summary());
                }
            }
        }

        OutputFormat::Json => {
            let output = serde_json::to_string_pretty(&serde_json::json!({
                "query": args.query,
                "results": results.results.iter().map(|r| {
                    serde_json::json!({
                        "score": r.score,
                        "doc_id": r.chunk.doc_id.to_string(),
                        "content": r.chunk.content,
                        "metadata": r.chunk.metadata,
                    })
                }).collect::<Vec<_>>(),
                "trace": results.trace,
            }))?;
            println!("{}", output);
        }

        OutputFormat::Csv => {
            println!("rank,score,doc_id,lines,preview");
            for (i, r) in results.results.iter().enumerate() {
                let preview: String = r.chunk.content
                    .chars()
                    .take(50)
                    .filter(|c| *c != '\n' && *c != ',')
                    .collect();
                println!(
                    "{},{:.4},{},{}-{},\"{}\"",
                    i + 1,
                    r.score,
                    r.chunk.doc_id,
                    r.chunk.metadata.line_start,
                    r.chunk.metadata.line_end,
                    preview
                );
            }
        }

        OutputFormat::Quiet => {
            for r in &results.results {
                println!("{}", r.chunk.doc_id);
            }
        }
    }

    Ok(())
}
```

---

## Ingest Command

```rust
#[derive(Args)]
struct IngestArgs {
    /// Files or directories to ingest
    #[arg(required = true)]
    paths: Vec<PathBuf>,

    /// Target collection
    #[arg(short, long, required = true)]
    collection: String,

    /// File patterns to include (e.g., "*.rs")
    #[arg(short, long)]
    include: Vec<String>,

    /// File patterns to exclude
    #[arg(short = 'x', long)]
    exclude: Vec<String>,

    /// Recurse into directories
    #[arg(short, long, default_value = "true")]
    recursive: bool,

    /// Watch for changes
    #[arg(short, long)]
    watch: bool,

    /// Dry run (don't actually ingest)
    #[arg(long)]
    dry_run: bool,
}

async fn run_ingest(args: IngestArgs, ctx: &Context) -> Result<()> {
    // Ensure collection exists
    if ctx.store.get_collection(&args.collection).await?.is_none() {
        println!("Creating collection: {}", args.collection);
        ctx.store.create_collection(Collection {
            name: args.collection.clone(),
            description: None,
            settings: CollectionSettings::default(),
            created_at: now_millis(),
            hlc: HybridLogicalClock::new(ctx.node_id),
        }).await?;
    }

    // Collect files
    let files = collect_files(&args.paths, &args.include, &args.exclude, args.recursive)?;

    println!("Found {} files to ingest", files.len());

    if args.dry_run {
        for file in &files {
            println!("  {}", file.display());
        }
        return Ok(());
    }

    // Progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap());

    let mut success = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for file in files {
        pb.set_message(file.file_name().unwrap_or_default().to_string_lossy().to_string());

        match ingest_file(&file, &args.collection, ctx).await {
            Ok(IngestResult::Created(chunks)) => {
                success += 1;
                pb.println(format!("  + {} ({} chunks)", file.display(), chunks));
            }
            Ok(IngestResult::Skipped) => {
                skipped += 1;
            }
            Ok(IngestResult::Updated(chunks)) => {
                success += 1;
                pb.println(format!("  ~ {} ({} chunks)", file.display(), chunks));
            }
            Err(e) => {
                failed += 1;
                pb.println(format!("  ! {} - {}", file.display(), e));
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message("Done");

    println!("\nSummary:");
    println!("  Created/Updated: {}", success);
    println!("  Skipped (unchanged): {}", skipped);
    println!("  Failed: {}", failed);

    if args.watch {
        println!("\nWatching for changes... (Ctrl+C to stop)");
        watch_and_ingest(&args.paths, &args.collection, ctx).await?;
    }

    Ok(())
}

enum IngestResult {
    Created(usize),
    Updated(usize),
    Skipped,
}
```

---

## Collection Commands

```rust
#[derive(Args)]
struct CollectionArgs {
    #[command(subcommand)]
    command: CollectionCommands,
}

#[derive(Subcommand)]
enum CollectionCommands {
    /// List collections
    List,

    /// Create a collection
    Create {
        name: String,
        #[arg(short, long)]
        description: Option<String>,
    },

    /// Delete a collection
    Delete {
        name: String,
        #[arg(long)]
        force: bool,
    },

    /// Show collection details
    Show { name: String },
}

async fn run_collection(args: CollectionArgs, ctx: &Context) -> Result<()> {
    match args.command {
        CollectionCommands::List => {
            let collections = ctx.store.list_collections().await?;

            println!("{:<20} {:<10} {:<10} {}", "NAME", "DOCS", "CHUNKS", "DESCRIPTION");
            println!("{}", "-".repeat(60));

            for c in collections {
                let docs = ctx.store.count_documents(Some(&c.name)).await?;
                let chunks = ctx.store.count_chunks(Some(&c.name)).await?;
                println!(
                    "{:<20} {:<10} {:<10} {}",
                    c.name,
                    docs,
                    chunks,
                    c.description.unwrap_or_default()
                );
            }
        }

        CollectionCommands::Create { name, description } => {
            ctx.store.create_collection(Collection {
                name: name.clone(),
                description,
                settings: CollectionSettings::default(),
                created_at: now_millis(),
                hlc: HybridLogicalClock::new(ctx.node_id),
            }).await?;

            println!("Created collection: {}", name);
        }

        CollectionCommands::Delete { name, force } => {
            let docs = ctx.store.count_documents(Some(&name)).await?;

            if docs > 0 && !force {
                println!("Collection '{}' contains {} documents.", name, docs);
                println!("Use --force to delete anyway.");
                return Ok(());
            }

            ctx.store.delete_collection(&name).await?;
            println!("Deleted collection: {}", name);
        }

        CollectionCommands::Show { name } => {
            let collection = ctx.store.get_collection(&name).await?
                .ok_or_else(|| RagError::CollectionNotFound(name.clone()))?;

            let docs = ctx.store.count_documents(Some(&name)).await?;
            let chunks = ctx.store.count_chunks(Some(&name)).await?;

            println!("Collection: {}", collection.name);
            println!("Description: {}", collection.description.unwrap_or_default());
            println!("Documents: {}", docs);
            println!("Chunks: {}", chunks);
            println!("Created: {}", format_timestamp(collection.created_at));
            println!("\nSettings:");
            println!("  Max tokens: {}", collection.settings.chunking.max_tokens);
            println!("  Min tokens: {}", collection.settings.chunking.min_tokens);
            println!("  Overlap: {}", collection.settings.chunking.overlap_tokens);
        }
    }

    Ok(())
}
```

---

## REPL Mode

```rust
async fn run_repl(ctx: &Context) -> Result<()> {
    println!("RAG REPL - Type .help for commands, .quit to exit\n");

    let mut rl = rustyline::DefaultEditor::new()?;
    let history_path = dirs::data_dir()
        .map(|p| p.join("rag-mcp/history.txt"));

    if let Some(ref path) = history_path {
        let _ = rl.load_history(path);
    }

    loop {
        let prompt = "rag> ";
        let readline = rl.readline(prompt);

        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                rl.add_history_entry(line)?;

                if line.starts_with('.') {
                    // REPL command
                    match handle_repl_command(line, ctx).await {
                        Ok(should_quit) if should_quit => break,
                        Ok(_) => {}
                        Err(e) => println!("Error: {}", e),
                    }
                } else {
                    // Treat as search query
                    match ctx.query_engine.search(line, None).await {
                        Ok(results) => {
                            for (i, r) in results.results.iter().take(5).enumerate() {
                                println!("[{}] {:.3} - {}", i + 1, r.score, r.chunk.doc_id);
                                let preview: String = r.chunk.content.chars().take(80).collect();
                                println!("    {}", preview.replace('\n', " "));
                            }
                            println!();
                        }
                        Err(e) => println!("Search error: {}", e),
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }

    if let Some(ref path) = history_path {
        let _ = rl.save_history(path);
    }

    Ok(())
}

async fn handle_repl_command(line: &str, ctx: &Context) -> Result<bool> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let cmd = parts.get(0).unwrap_or(&"");

    match *cmd {
        ".quit" | ".exit" | ".q" => {
            println!("Goodbye!");
            return Ok(true);
        }

        ".help" | ".h" => {
            println!("REPL Commands:");
            println!("  .search <query>     Search (or just type query directly)");
            println!("  .collections        List collections");
            println!("  .stats              Show statistics");
            println!("  .inspect <id>       Inspect a chunk");
            println!("  .similar <id>       Find similar chunks");
            println!("  .explain <query>    Show query trace");
            println!("  .quit               Exit REPL");
        }

        ".collections" | ".c" => {
            let collections = ctx.store.list_collections().await?;
            for c in collections {
                let count = ctx.store.count_documents(Some(&c.name)).await?;
                println!("  {} ({} docs)", c.name, count);
            }
        }

        ".stats" => {
            let stats = ctx.store.stats().await?;
            println!("Collections: {}", stats.collections);
            println!("Documents: {}", stats.documents);
            println!("Chunks: {}", stats.chunks);
            println!("Storage: {} MB", stats.storage_bytes / 1_000_000);
        }

        ".search" | ".s" => {
            let query = parts[1..].join(" ");
            let results = ctx.query_engine.search(&query, None).await?;
            for (i, r) in results.results.iter().enumerate() {
                println!("[{}] {:.3} - {} ({})", i + 1, r.score, r.chunk.doc_id, r.chunk.id);
            }
        }

        ".inspect" | ".i" => {
            if parts.len() < 2 {
                println!("Usage: .inspect <chunk_id>");
                return Ok(false);
            }
            let id = Ulid::from_string(parts[1])?;
            if let Some(chunk) = ctx.store.get_chunk(id).await? {
                println!("Chunk: {}", chunk.id);
                println!("Document: {}", chunk.doc_id);
                println!("Index: {}", chunk.chunk_index);
                println!("Tokens: {}", chunk.token_count);
                println!("Lines: {}-{}", chunk.metadata.line_start, chunk.metadata.line_end);
                println!("\nContent:\n{}", chunk.content);
            } else {
                println!("Chunk not found");
            }
        }

        ".similar" => {
            if parts.len() < 2 {
                println!("Usage: .similar <chunk_id>");
                return Ok(false);
            }
            let id = Ulid::from_string(parts[1])?;
            let results = ctx.query_engine.find_similar(id, 5).await?;
            for (i, r) in results.results.iter().enumerate() {
                println!("[{}] {:.3} - {}", i + 1, r.score, r.chunk.id);
            }
        }

        ".explain" | ".e" => {
            let query = parts[1..].join(" ");
            let config = QueryConfig {
                enable_tracing: true,
                ..Default::default()
            };
            let results = ctx.query_engine.search_with_config(&query, None, &config).await?;
            if let Some(trace) = results.trace {
                println!("{}", trace.summary());
            }
        }

        _ => {
            println!("Unknown command: {}", cmd);
            println!("Type .help for available commands");
        }
    }

    Ok(false)
}
```

---

## Usage Examples

```bash
# Search
rag search "error handling in async functions"
rag search "config parsing" -c code -k 5 --trace
rag search "README" --mode keyword

# Ingest
rag ingest ./src -c code --include "*.rs"
rag ingest ./docs -c documentation --recursive
rag ingest ./src -c code --watch

# Collections
rag collection list
rag collection create my-project -d "Project documentation"
rag collection delete old-project --force

# Documents
rag document list -c code
rag document show 01HRE4KXQN...
rag document delete 01HRE4KXQN...

# Sync
rag sync status
rag sync add peer-alpha http://192.168.1.10:8765
rag sync now peer-alpha

# Stats
rag stats
rag stats -c code

# REPL
rag repl

# Server
rag serve
rag serve --port 9000

# Maintenance
rag optimize
rag export backup.json
rag import backup.json
```

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|------------|-------|
| clap CLI structure | HIGH | Mature library |
| Search command | HIGH | Direct query engine use |
| Ingest command | HIGH | File handling is standard |
| REPL mode | HIGH | rustyline is solid |
| Watch mode | MEDIUM | notify crate integration |
| Output formatting | HIGH | Standard patterns |

---

## Cargo.toml

```toml
[package]
name = "rag-cli"
version = "0.1.0"
edition = "2021"
description = "CLI for the RAG system"
license = "MIT"

[[bin]]
name = "rag"
path = "src/main.rs"

[dependencies]
rag-core = { path = "../rag-core" }
rag-store = { path = "../rag-store" }
rag-query = { path = "../rag-query" }
rag-embed = { path = "../rag-embed" }
rag-chunk = { path = "../rag-chunk" }
rag-sync = { path = "../rag-sync" }
rag-mcp = { path = "../rag-mcp" }
clap = { version = "4.4", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rustyline = "13.0"
indicatif = "0.17"
glob = "0.3"
notify = "6.1"
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
tempfile = "3.10"
```
