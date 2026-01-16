# TESTING.md - Test Strategy

Comprehensive testing strategy for the RAG system.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Categories](#test-categories)
3. [Test Fixtures](#test-fixtures)
4. [Integration Testing](#integration-testing)
5. [Performance Testing](#performance-testing)
6. [Quality Assurance](#quality-assurance)
7. [CI/CD Pipeline](#cicd-pipeline)

---

## Testing Philosophy

### Principles

1. **Test behavior, not implementation** - Tests should verify what the system does, not how it does it
2. **Prefer integration over unit** - Many bugs occur at component boundaries
3. **Automate everything** - Manual testing doesn't scale
4. **Fast feedback** - Tests should run quickly during development
5. **Meaningful coverage** - 80% coverage with good tests > 100% coverage with weak tests

### Test Pyramid

```
        /\
       /  \       E2E Tests (few, slow, high confidence)
      /----\
     /      \     Integration Tests (moderate, medium speed)
    /--------\
   /          \   Unit Tests (many, fast, low confidence per test)
  /------------\
```

---

## Test Categories

### Unit Tests

**Location:** `crates/*/src/**/*.rs` (inline `#[cfg(test)]` modules)

**Scope:**
- Pure functions
- Data transformations
- Algorithms (RRF, HLC, etc.)

**Example:**

```rust
// rag-core/src/hlc.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlc_tick_increments_logical() {
        let mut hlc = HybridLogicalClock::new(1);
        let t1 = hlc.tick();
        let t2 = hlc.tick();

        assert!(t2 > t1);
        assert_eq!(t2.node_id, 1);
    }

    #[test]
    fn test_hlc_update_advances_on_remote() {
        let mut local = HybridLogicalClock::new(1);
        let remote = HybridLogicalClock {
            wall_time: local.wall_time + 1000,
            logical: 5,
            node_id: 2,
        };

        local.update(&remote);

        assert!(local.wall_time >= remote.wall_time);
        assert!(local.logical > remote.logical);
    }

    #[test]
    fn test_hlc_byte_ordering() {
        let a = HybridLogicalClock { wall_time: 100, logical: 0, node_id: 1 };
        let b = HybridLogicalClock { wall_time: 100, logical: 1, node_id: 1 };
        let c = HybridLogicalClock { wall_time: 101, logical: 0, node_id: 1 };

        assert!(a.to_bytes() < b.to_bytes());
        assert!(b.to_bytes() < c.to_bytes());
    }
}
```

### Integration Tests

**Location:** `crates/*/tests/*.rs`

**Scope:**
- Component interactions
- Database operations
- Full pipelines

**Example:**

```rust
// rag-store/tests/store_integration.rs
use rag_core::*;
use rag_store::SqliteStore;
use tempfile::tempdir;

#[tokio::test]
async fn test_document_lifecycle() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    let store = SqliteStore::open(&db_path, 1).await.unwrap();

    // Create collection
    store.create_collection(Collection {
        name: "test".to_string(),
        description: None,
        settings: CollectionSettings::default(),
        created_at: 0,
        hlc: HybridLogicalClock::new(1),
    }).await.unwrap();

    // Insert document
    let doc = Document::new("test", "file://test.rs", "content", ContentType::Rust);
    let doc_id = store.insert_document(doc.clone()).await.unwrap();

    // Retrieve
    let retrieved = store.get_document(doc_id).await.unwrap().unwrap();
    assert_eq!(retrieved.source_uri, "file://test.rs");

    // Delete
    store.delete_document(doc_id).await.unwrap();
    assert!(store.get_document(doc_id).await.unwrap().is_none());
}

#[tokio::test]
async fn test_search_returns_results() {
    let (store, embedder) = setup_test_store().await;

    // Ingest test document
    ingest_test_document(&store, &embedder, "fn main() { println!(\"hello\"); }").await;

    // Search
    let engine = QueryEngine::new(
        Arc::new(store),
        Arc::new(embedder),
        QueryConfig::default(),
    );

    let results = engine.search("main function", None).await.unwrap();

    assert!(!results.results.is_empty());
    assert!(results.results[0].content.contains("main"));
}
```

### End-to-End Tests

**Location:** `tests/e2e/*.rs`

**Scope:**
- Full system workflows
- MCP protocol
- CLI commands

**Example:**

```rust
// tests/e2e/mcp_test.rs
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader, Write};

#[test]
fn test_mcp_search_tool() {
    // Start server
    let mut server = Command::new(env!("CARGO_BIN_EXE_rag-mcp"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdin = server.stdin.take().unwrap();
    let stdout = BufReader::new(server.stdout.take().unwrap());

    // Initialize
    let init_request = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
    writeln!(stdin, "{}", init_request).unwrap();

    // Read response
    let mut lines = stdout.lines();
    let response = lines.next().unwrap().unwrap();
    assert!(response.contains("capabilities"));

    // Search
    let search_request = r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"rag_search","arguments":{"query":"test"}}}"#;
    writeln!(stdin, "{}", search_request).unwrap();

    let response = lines.next().unwrap().unwrap();
    assert!(response.contains("results"));

    server.kill().unwrap();
}
```

---

## Test Fixtures

### Standard Test Corpus

```rust
// tests/fixtures/mod.rs

pub struct TestCorpus;

impl TestCorpus {
    /// Small set of Rust files for testing.
    pub fn rust_files() -> Vec<(&'static str, &'static str)> {
        vec![
            ("lib.rs", include_str!("corpus/rust/lib.rs")),
            ("main.rs", include_str!("corpus/rust/main.rs")),
            ("utils.rs", include_str!("corpus/rust/utils.rs")),
        ]
    }

    /// Markdown documentation.
    pub fn docs() -> Vec<(&'static str, &'static str)> {
        vec![
            ("README.md", include_str!("corpus/docs/README.md")),
            ("GUIDE.md", include_str!("corpus/docs/GUIDE.md")),
        ]
    }

    /// Search queries with expected results.
    pub fn queries() -> Vec<QueryFixture> {
        vec![
            QueryFixture {
                query: "main function",
                expected_file: "main.rs",
                min_score: 0.7,
            },
            QueryFixture {
                query: "error handling",
                expected_file: "utils.rs",
                min_score: 0.6,
            },
        ]
    }
}

pub struct QueryFixture {
    pub query: &'static str,
    pub expected_file: &'static str,
    pub min_score: f32,
}
```

### Mock Components

```rust
// tests/mocks/mod.rs

use async_trait::async_trait;
use mockall::mock;
use rag_core::*;

mock! {
    pub Store {}

    #[async_trait]
    impl Store for Store {
        async fn get_document(&self, id: Ulid) -> Result<Option<Document>>;
        async fn insert_document(&self, doc: Document) -> Result<Ulid>;
        async fn delete_document(&self, id: Ulid) -> Result<()>;
        async fn vector_search(&self, query: &[f32], k: usize, collection: Option<&str>) -> Result<Vec<(Ulid, f32)>>;
        async fn keyword_search(&self, query: &str, k: usize, collection: Option<&str>) -> Result<Vec<(Ulid, f32)>>;
        // ... other methods
    }
}

mock! {
    pub Embedder {}

    impl Embedder for Embedder {
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
        fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
        fn dimension(&self) -> usize;
        fn max_tokens(&self) -> usize;
        fn model_id(&self) -> &str;
    }
}
```

---

## Integration Testing

### Database Tests

```rust
#[tokio::test]
async fn test_concurrent_reads() {
    let store = setup_store().await;

    // Spawn multiple readers
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let store = store.clone();
            tokio::spawn(async move {
                for _ in 0..100 {
                    store.list_collections().await.unwrap();
                }
            })
        })
        .collect();

    // All should complete without deadlock
    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_transaction_rollback() {
    let store = setup_store().await;

    // Start transaction, insert, then simulate failure
    let result: Result<(), RagError> = async {
        store.insert_document(doc1).await?;
        // Simulate failure
        return Err(RagError::Internal("simulated".into()));
    }.await;

    assert!(result.is_err());

    // Document should not exist (rolled back)
    assert!(store.get_document(doc1.id).await.unwrap().is_none());
}
```

### Embedding Tests

```rust
#[test]
fn test_embedding_dimension() {
    let embedder = OnnxEmbedder::new(MODEL_PATH).unwrap();
    let embedding = embedder.embed(&["test"]).unwrap();

    assert_eq!(embedding[0].len(), 768);
}

#[test]
fn test_embedding_normalization() {
    let embedder = OnnxEmbedder::new(MODEL_PATH).unwrap();
    let embedding = embedder.embed(&["test"]).unwrap();

    let norm: f32 = embedding[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.001);
}

#[test]
fn test_similar_texts_have_high_similarity() {
    let embedder = OnnxEmbedder::new(MODEL_PATH).unwrap();

    let emb1 = embedder.embed(&["The cat sat on the mat"]).unwrap();
    let emb2 = embedder.embed(&["A cat was sitting on the rug"]).unwrap();
    let emb3 = embedder.embed(&["Stock prices fell yesterday"]).unwrap();

    let sim_12 = cosine_similarity(&emb1[0], &emb2[0]);
    let sim_13 = cosine_similarity(&emb1[0], &emb3[0]);

    assert!(sim_12 > sim_13, "Similar texts should have higher similarity");
    assert!(sim_12 > 0.7, "Similar texts should have >0.7 similarity");
}
```

### Search Quality Tests

```rust
#[tokio::test]
async fn test_search_recall() {
    let (store, embedder) = setup_with_corpus().await;
    let engine = QueryEngine::new(store, embedder, QueryConfig::default());

    let fixtures = TestCorpus::queries();
    let mut passed = 0;

    for fixture in &fixtures {
        let results = engine.search(fixture.query, None).await.unwrap();

        let found = results.results.iter().any(|r| {
            r.chunk.doc_id.to_string().contains(fixture.expected_file)
                && r.score >= fixture.min_score
        });

        if found {
            passed += 1;
        }
    }

    let recall = passed as f32 / fixtures.len() as f32;
    assert!(recall >= 0.8, "Search recall should be >= 80%, got {:.1}%", recall * 100.0);
}
```

---

## Performance Testing

### Benchmarks

```rust
// benches/search_benchmark.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn search_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (store, embedder) = rt.block_on(setup_benchmark_store());
    let engine = QueryEngine::new(store, embedder, QueryConfig::default());

    let mut group = c.benchmark_group("search");

    for k in [5, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::new("top_k", k), k, |b, &k| {
            b.to_async(&rt).iter(|| async {
                let config = QueryConfig { final_k: k, ..Default::default() };
                engine.search_with_config("error handling", None, &config).await
            });
        });
    }

    group.finish();
}

fn ingest_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("ingest_100_files", |b| {
        b.iter(|| {
            rt.block_on(async {
                let store = setup_temp_store().await;
                let corpus = TestCorpus::rust_files();

                for (name, content) in &corpus {
                    ingest_document(&store, name, content).await.unwrap();
                }
            })
        })
    });
}

criterion_group!(benches, search_benchmark, ingest_benchmark);
criterion_main!(benches);
```

### Load Tests

```rust
#[tokio::test]
async fn test_concurrent_search_load() {
    let engine = setup_engine().await;

    let queries = vec![
        "error handling",
        "main function",
        "configuration",
        "database connection",
        "async await",
    ];

    let start = std::time::Instant::now();
    let mut handles = Vec::new();

    // 100 concurrent searches
    for i in 0..100 {
        let engine = engine.clone();
        let query = queries[i % queries.len()].to_string();

        handles.push(tokio::spawn(async move {
            engine.search(&query, None).await
        }));
    }

    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    let duration = start.elapsed();
    let qps = 100.0 / duration.as_secs_f64();

    println!("QPS: {:.1}", qps);
    assert!(qps >= 10.0, "Should handle >= 10 QPS, got {:.1}", qps);
}
```

### Memory Tests

```rust
#[test]
fn test_embedding_memory_usage() {
    let embedder = OnnxEmbedder::new(MODEL_PATH).unwrap();

    let before = get_memory_usage();

    // Embed 1000 texts
    let texts: Vec<String> = (0..1000).map(|i| format!("Sample text {}", i)).collect();
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let _embeddings = embedder.embed(&refs).unwrap();

    let after = get_memory_usage();
    let delta_mb = (after - before) as f64 / 1_000_000.0;

    println!("Memory delta: {:.1} MB", delta_mb);
    assert!(delta_mb < 500.0, "Embedding 1000 texts should use <500MB, used {:.1}", delta_mb);
}
```

---

## Quality Assurance

### Retrieval Quality Metrics

```rust
struct RetrievalMetrics {
    precision_at_k: HashMap<usize, f32>,
    recall_at_k: HashMap<usize, f32>,
    ndcg: f32,
    mrr: f32,
}

impl RetrievalMetrics {
    fn compute(results: &[SearchResult], relevance: &[Ulid]) -> Self {
        let mut metrics = Self {
            precision_at_k: HashMap::new(),
            recall_at_k: HashMap::new(),
            ndcg: 0.0,
            mrr: 0.0,
        };

        let relevant_set: HashSet<_> = relevance.iter().collect();
        let mut first_relevant_rank = None;

        for k in [1, 3, 5, 10, 20] {
            let top_k: HashSet<_> = results.iter()
                .take(k)
                .map(|r| &r.chunk.id)
                .collect();

            let relevant_in_top_k = top_k.intersection(&relevant_set).count();

            metrics.precision_at_k.insert(k, relevant_in_top_k as f32 / k as f32);
            metrics.recall_at_k.insert(k, relevant_in_top_k as f32 / relevance.len() as f32);
        }

        // MRR
        for (i, r) in results.iter().enumerate() {
            if relevant_set.contains(&r.chunk.id) {
                metrics.mrr = 1.0 / (i + 1) as f32;
                break;
            }
        }

        // NDCG (simplified)
        let dcg: f32 = results.iter()
            .enumerate()
            .filter(|(_, r)| relevant_set.contains(&r.chunk.id))
            .map(|(i, _)| 1.0 / (i + 2) as f32.log2())
            .sum();

        let ideal_dcg: f32 = (0..relevance.len())
            .map(|i| 1.0 / (i + 2) as f32.log2())
            .sum();

        metrics.ndcg = if ideal_dcg > 0.0 { dcg / ideal_dcg } else { 0.0 };

        metrics
    }
}
```

### Regression Testing

```rust
#[test]
fn test_no_retrieval_quality_regression() {
    let baseline_mrr = 0.75;
    let baseline_recall_10 = 0.85;

    let engine = setup_engine();
    let test_queries = load_test_queries();

    let mut total_mrr = 0.0;
    let mut total_recall_10 = 0.0;

    for (query, relevant) in &test_queries {
        let results = engine.search(query, None).await.unwrap();
        let metrics = RetrievalMetrics::compute(&results.results, relevant);

        total_mrr += metrics.mrr;
        total_recall_10 += metrics.recall_at_k[&10];
    }

    let avg_mrr = total_mrr / test_queries.len() as f32;
    let avg_recall = total_recall_10 / test_queries.len() as f32;

    assert!(avg_mrr >= baseline_mrr * 0.95, "MRR regressed: {:.3} < {:.3}", avg_mrr, baseline_mrr);
    assert!(avg_recall >= baseline_recall_10 * 0.95, "Recall@10 regressed: {:.3} < {:.3}", avg_recall, baseline_recall_10);
}
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always
  RAG_MODEL_PATH: ./models/nomic-embed-text-v1.5

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - name: Check formatting
        run: cargo fmt --all -- --check
      - name: Clippy
        run: cargo clippy --workspace -- -D warnings

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libsqlite3-dev

      - name: Cache model
        uses: actions/cache@v4
        with:
          path: ./models
          key: models-nomic-v1.5

      - name: Download model
        run: ./scripts/download-model.sh

      - name: Run tests
        run: cargo test --workspace

  integration:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Build
        run: cargo build --release

      - name: Run integration tests
        run: cargo test --release --test '*'

  benchmark:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run benchmarks
        run: cargo bench --bench search_benchmark -- --noplot

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmarks
          path: target/criterion
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: fmt
        name: cargo fmt
        entry: cargo fmt --all --
        language: system
        types: [rust]
        pass_filenames: false

      - id: clippy
        name: cargo clippy
        entry: cargo clippy --workspace -- -D warnings
        language: system
        types: [rust]
        pass_filenames: false

      - id: test
        name: cargo test
        entry: cargo test --workspace --lib
        language: system
        types: [rust]
        pass_filenames: false
        stages: [push]
```

---

## Test Commands Reference

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p rag-core
cargo test -p rag-store

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_hlc_ordering

# Run integration tests only
cargo test --test '*'

# Run benchmarks
cargo bench

# Run with coverage
cargo tarpaulin --workspace --out html

# Run mutation testing
cargo mutants --workspace
```
