//! RAG CLI - Command-line interface for the RAG knowledge base.

use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use rag_mcp::{CollectionParams, IngestParams, RagMcpServer, SearchParams};

/// RAG - Local Retrieval-Augmented Generation knowledge base
#[derive(Parser)]
#[command(name = "rag")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Database path (default: ~/.rag/db.sqlite)
    #[arg(short, long, global = true)]
    database: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search the knowledge base
    Search {
        /// Search query
        query: String,

        /// Maximum number of results
        #[arg(short = 'k', long, default_value = "10")]
        top_k: u32,

        /// Collection to search (searches all if not specified)
        #[arg(short, long)]
        collection: Option<String>,
    },

    /// Ingest a file or directory into the knowledge base
    Ingest {
        /// Path to file or directory to ingest
        path: PathBuf,

        /// Collection to ingest into
        #[arg(short, long)]
        collection: String,

        /// Recursively process directories
        #[arg(short, long)]
        recursive: bool,
    },

    /// Manage collections
    Collection {
        #[command(subcommand)]
        action: CollectionAction,
    },

    /// Show statistics
    Stats {
        /// Collection to get stats for (all if not specified)
        #[arg(long)]
        collection: Option<String>,
    },

    /// Initialize the database
    Init,
}

#[derive(Subcommand)]
enum CollectionAction {
    /// List all collections
    List,

    /// Create a new collection
    Create {
        /// Collection name
        name: String,

        /// Description
        #[arg(long)]
        description: Option<String>,
    },

    /// Delete a collection
    Delete {
        /// Collection name
        name: String,
    },
}

fn get_db_path(db: Option<PathBuf>) -> PathBuf {
    if let Some(path) = db {
        return path;
    }

    // Default to ~/.rag/db.sqlite
    let home = dirs::home_dir().expect("Could not determine home directory");
    home.join(".rag").join("db.sqlite")
}

fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::WARN };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber).ok();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    setup_logging(cli.verbose);

    let db_path = get_db_path(cli.database);

    match cli.command {
        Commands::Init => {
            init_database(&db_path)?;
        }
        Commands::Search {
            query,
            top_k,
            collection,
        } => {
            let server = get_server(&db_path)?;
            search(&server, &query, top_k, collection).await;
        }
        Commands::Ingest {
            path,
            collection,
            recursive,
        } => {
            let server = get_server(&db_path)?;
            ingest(&server, &path, &collection, recursive).await?;
        }
        Commands::Collection { action } => {
            let server = get_server(&db_path)?;
            match action {
                CollectionAction::List => {
                    list_collections(&server).await;
                }
                CollectionAction::Create { name, description } => {
                    create_collection(&server, &name, description).await;
                }
                CollectionAction::Delete { name } => {
                    delete_collection(&server, &name).await;
                }
            }
        }
        Commands::Stats { collection } => {
            let server = get_server(&db_path)?;
            stats(&server, collection.as_deref()).await;
        }
    }

    Ok(())
}

fn init_database(db_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Create parent directory if needed
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Create the database by opening the server
    let _server = RagMcpServer::new(db_path)?;
    println!("Initialized database at: {}", db_path.display());
    Ok(())
}

fn get_server(db_path: &PathBuf) -> Result<RagMcpServer, Box<dyn std::error::Error>> {
    // Check if database directory exists
    if let Some(parent) = db_path.parent() {
        if !parent.exists() {
            eprintln!(
                "Database directory does not exist. Run 'rag init' first, or specify a path with -d."
            );
            std::process::exit(1);
        }
    }

    Ok(RagMcpServer::new(db_path)?)
}

async fn search(server: &RagMcpServer, query: &str, top_k: u32, collection: Option<String>) {
    let params = SearchParams {
        query: query.to_string(),
        top_k,
        collection,
    };

    let result = server.search(params).await;
    if result.success {
        println!("{}", result.message);
    } else {
        eprintln!("Error: {}", result.message);
        std::process::exit(1);
    }
}

async fn ingest(
    server: &RagMcpServer,
    path: &PathBuf,
    collection: &str,
    recursive: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let files = collect_files(path, recursive)?;

    if files.is_empty() {
        println!("No supported files found at: {}", path.display());
        return Ok(());
    }

    println!(
        "Ingesting {} file(s) into collection '{}'...",
        files.len(),
        collection
    );

    let mut success_count = 0;
    let mut error_count = 0;

    for file_path in files {
        let content = match fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  Error reading {}: {}", file_path.display(), e);
                error_count += 1;
                continue;
            }
        };

        let source_uri = format!("file://{}", file_path.canonicalize()?.display());

        let params = IngestParams {
            collection: collection.to_string(),
            source_uri,
            content,
            content_type: detect_content_type(&file_path),
        };

        let result = server.ingest(params).await;
        if result.success {
            println!("  {} - OK", file_path.display());
            success_count += 1;
        } else {
            eprintln!("  {} - Error: {}", file_path.display(), result.message);
            error_count += 1;
        }
    }

    println!(
        "\nComplete: {} succeeded, {} failed",
        success_count, error_count
    );

    Ok(())
}

fn collect_files(path: &PathBuf, recursive: bool) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut files = Vec::new();

    if path.is_file() {
        if is_supported_file(path) {
            files.push(path.clone());
        }
    } else if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();

            if entry_path.is_file() && is_supported_file(&entry_path) {
                files.push(entry_path);
            } else if entry_path.is_dir() && recursive {
                files.extend(collect_files(&entry_path, recursive)?);
            }
        }
    }

    Ok(files)
}

fn is_supported_file(path: &PathBuf) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(
        ext,
        "rs" | "py"
            | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "go"
            | "md"
            | "txt"
            | "json"
            | "yaml"
            | "yml"
            | "toml"
            | "html"
            | "css"
            | "c"
            | "cpp"
            | "h"
            | "hpp"
            | "java"
            | "rb"
            | "sh"
    )
}

fn detect_content_type(path: &PathBuf) -> Option<String> {
    let ext = path.extension().and_then(|e| e.to_str())?;
    let content_type = match ext {
        "rs" => "rust",
        "py" => "python",
        "js" => "javascript",
        "ts" | "tsx" => "typescript",
        "go" => "go",
        "md" => "markdown",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        _ => return None,
    };
    Some(content_type.to_string())
}

async fn list_collections(server: &RagMcpServer) {
    let result = server.list_collections().await;
    if result.success {
        println!("{}", result.message);
    } else {
        eprintln!("Error: {}", result.message);
        std::process::exit(1);
    }
}

async fn create_collection(server: &RagMcpServer, name: &str, description: Option<String>) {
    let params = CollectionParams {
        name: name.to_string(),
        description,
    };

    let result = server.create_collection(params).await;
    if result.success {
        println!("{}", result.message);
    } else {
        eprintln!("Error: {}", result.message);
        std::process::exit(1);
    }
}

async fn delete_collection(server: &RagMcpServer, name: &str) {
    let result = server.delete_collection(name).await;
    if result.success {
        println!("{}", result.message);
    } else {
        eprintln!("Error: {}", result.message);
        std::process::exit(1);
    }
}

async fn stats(server: &RagMcpServer, collection: Option<&str>) {
    let result = server.stats(collection).await;
    if result.success {
        println!("{}", result.message);
    } else {
        eprintln!("Error: {}", result.message);
        std::process::exit(1);
    }
}
