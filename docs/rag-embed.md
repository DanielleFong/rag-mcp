# rag-embed

Local embedding generation using ONNX Runtime with the nomic-embed-text-v1.5 model.

## Overview

| Attribute | Value |
|-----------|-------|
| Purpose | Generate embeddings for documents and queries |
| Dependencies | ort (ONNX Runtime), tokenizers, rag-core |
| Est. Lines | ~600 |
| Confidence | HIGH |

This crate implements the `Embedder` trait from rag-core using local ONNX models.

---

## Module Structure

```
rag-embed/
├── src/
│   ├── lib.rs           # Public exports
│   ├── embedder.rs      # OnnxEmbedder implementation
│   ├── tokenizer.rs     # Tokenizer wrapper
│   ├── pooling.rs       # Mean pooling implementation
│   └── normalize.rs     # L2 normalization
├── models/              # Model download scripts
│   └── download.sh
└── Cargo.toml
```

---

## OnnxEmbedder Implementation

### Construction

```rust
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder};
use tokenizers::Tokenizer;
use std::path::Path;
use std::sync::Arc;

/// ONNX-based embedder using nomic-embed-text-v1.5.
pub struct OnnxEmbedder {
    /// ONNX session for inference.
    session: Session,

    /// Tokenizer for text preprocessing.
    tokenizer: Tokenizer,

    /// Embedding dimension.
    dimension: usize,

    /// Maximum tokens per input.
    max_tokens: usize,

    /// Model identifier.
    model_id: String,

    /// Batch size for embedding.
    batch_size: usize,
}

impl OnnxEmbedder {
    /// Create a new embedder from a model directory.
    ///
    /// The directory should contain:
    /// - model.onnx (the ONNX model)
    /// - tokenizer.json (the tokenizer)
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Initialize ONNX Runtime environment
        let environment = Environment::builder()
            .with_name("rag-embed")
            .with_execution_providers([
                // Try GPU first, fall back to CPU
                ExecutionProvider::CUDA(Default::default()),
                ExecutionProvider::CoreML(Default::default()),
                ExecutionProvider::CPU(Default::default()),
            ])
            .build()?;

        // Load ONNX model
        let model_path = model_dir.join("model.onnx");
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(&model_path)?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RagError::EmbeddingModel(e.to_string()))?;

        Ok(Self {
            session,
            tokenizer,
            dimension: 768,       // nomic-embed-text-v1.5 dimension
            max_tokens: 8192,     // nomic-embed-text-v1.5 context length
            model_id: "nomic-embed-text-v1.5".to_string(),
            batch_size: 32,
        })
    }

    /// Create with custom configuration.
    pub fn with_config(model_dir: impl AsRef<Path>, config: EmbedderConfig) -> Result<Self> {
        let mut embedder = Self::new(model_dir)?;
        embedder.batch_size = config.batch_size;
        Ok(embedder)
    }
}

/// Configuration for the embedder.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Batch size for embedding multiple texts.
    pub batch_size: usize,

    /// Number of threads for inference.
    pub num_threads: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_threads: 4,
        }
    }
}
```

### Core Embedding Logic

```rust
impl OnnxEmbedder {
    /// Embed a batch of texts (internal method).
    fn embed_batch_internal(&self, texts: &[&str], prefix: &str) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Add prefix to each text (nomic uses different prefixes for doc vs query)
        let prefixed: Vec<String> = texts.iter()
            .map(|t| format!("{}{}", prefix, t))
            .collect();

        let prefixed_refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();

        // Tokenize
        let encodings = self.tokenizer
            .encode_batch(prefixed_refs.clone(), true)
            .map_err(|e| RagError::EmbeddingModel(e.to_string()))?;

        // Validate lengths
        for (i, encoding) in encodings.iter().enumerate() {
            let len = encoding.get_ids().len();
            if len > self.max_tokens {
                return Err(RagError::TextTooLong {
                    tokens: len,
                    max: self.max_tokens,
                });
            }
        }

        // Prepare input tensors
        let batch_size = encodings.len();
        let max_length = encodings.iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Pad sequences
        let mut input_ids = vec![0i64; batch_size * max_length];
        let mut attention_mask = vec![0i64; batch_size * max_length];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            for (j, &id) in ids.iter().enumerate() {
                input_ids[i * max_length + j] = id as i64;
            }
            for (j, &m) in mask.iter().enumerate() {
                attention_mask[i * max_length + j] = m as i64;
            }
        }

        // Create ONNX tensors
        let input_ids_tensor = ort::inputs![
            "input_ids" => (vec![batch_size, max_length], input_ids),
        ]?;

        let attention_mask_tensor = ort::inputs![
            "attention_mask" => (vec![batch_size, max_length], attention_mask),
        ]?;

        // Run inference
        let outputs = self.session.run(vec![
            input_ids_tensor,
            attention_mask_tensor,
        ])?;

        // Extract embeddings (last_hidden_state)
        let hidden_states = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()?;

        // Mean pooling over non-padding tokens
        let embeddings = self.mean_pool(
            hidden_states.view(),
            &attention_mask,
            batch_size,
            max_length,
        );

        // L2 normalize
        let normalized = self.l2_normalize(embeddings);

        Ok(normalized)
    }

    /// Mean pooling over sequence dimension, weighted by attention mask.
    fn mean_pool(
        &self,
        hidden_states: ndarray::ArrayView3<f32>,  // [batch, seq, hidden]
        attention_mask: &[i64],                    // [batch * seq]
        batch_size: usize,
        seq_length: usize,
    ) -> Vec<Vec<f32>> {
        let hidden_dim = self.dimension;
        let mut embeddings = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut sum = vec![0.0f32; hidden_dim];
            let mut count = 0.0f32;

            for s in 0..seq_length {
                let mask = attention_mask[b * seq_length + s] as f32;
                if mask > 0.0 {
                    for d in 0..hidden_dim {
                        sum[d] += hidden_states[[b, s, d]] * mask;
                    }
                    count += mask;
                }
            }

            // Normalize by count
            if count > 0.0 {
                for d in 0..hidden_dim {
                    sum[d] /= count;
                }
            }

            embeddings.push(sum);
        }

        embeddings
    }

    /// L2 normalize embeddings.
    fn l2_normalize(&self, embeddings: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        embeddings.into_iter()
            .map(|mut emb| {
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut emb {
                        *x /= norm;
                    }
                }
                emb
            })
            .collect()
    }
}
```

### Embedder Trait Implementation

```rust
impl Embedder for OnnxEmbedder {
    /// Embed multiple texts (for documents).
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Check for empty texts
        for (i, text) in texts.iter().enumerate() {
            if text.trim().is_empty() {
                return Err(RagError::EmptyText);
            }
        }

        // Process in batches
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(self.batch_size) {
            // Document prefix for nomic model
            let embeddings = self.embed_batch_internal(batch, "search_document: ")?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Embed a single query.
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        if query.trim().is_empty() {
            return Err(RagError::EmptyText);
        }

        // Query prefix for nomic model
        let embeddings = self.embed_batch_internal(&[query], "search_query: ")?;

        Ok(embeddings.into_iter().next().unwrap())
    }

    /// Get embedding dimension.
    fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get maximum input tokens.
    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Get model identifier.
    fn model_id(&self) -> &str {
        &self.model_id
    }
}
```

---

## Model Download

```bash
#!/bin/bash
# models/download.sh

MODEL_NAME="nomic-embed-text-v1.5"
MODEL_DIR="${1:-./models/${MODEL_NAME}}"

mkdir -p "$MODEL_DIR"

echo "Downloading ${MODEL_NAME}..."

# Download from Hugging Face (ONNX version)
curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx" \
    -o "${MODEL_DIR}/model.onnx"

curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json" \
    -o "${MODEL_DIR}/tokenizer.json"

curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer_config.json" \
    -o "${MODEL_DIR}/tokenizer_config.json"

echo "Downloaded to ${MODEL_DIR}"
```

---

## Token Counting Utility

```rust
// tokenizer.rs

impl OnnxEmbedder {
    /// Count tokens in a text without embedding.
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| RagError::EmbeddingModel(e.to_string()))?;

        Ok(encoding.get_ids().len())
    }

    /// Check if text fits within context.
    pub fn fits_in_context(&self, text: &str) -> Result<bool> {
        Ok(self.count_tokens(text)? <= self.max_tokens)
    }

    /// Truncate text to fit within context.
    pub fn truncate_to_context(&self, text: &str) -> Result<String> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| RagError::EmbeddingModel(e.to_string()))?;

        let ids = encoding.get_ids();
        if ids.len() <= self.max_tokens {
            return Ok(text.to_string());
        }

        // Decode truncated tokens
        let truncated_ids: Vec<u32> = ids[..self.max_tokens].to_vec();
        let truncated = self.tokenizer
            .decode(&truncated_ids, true)
            .map_err(|e| RagError::EmbeddingModel(e.to_string()))?;

        Ok(truncated)
    }
}
```

---

## Standalone Token Counter

For use without loading the full embedding model:

```rust
/// Lightweight tokenizer wrapper for token counting.
pub struct TokenCounter {
    tokenizer: Tokenizer,
}

impl TokenCounter {
    /// Load tokenizer from path.
    pub fn new(tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| RagError::EmbeddingModel(e.to_string()))?;

        Ok(Self { tokenizer })
    }

    /// Count tokens in text.
    pub fn count(&self, text: &str) -> Result<usize> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| RagError::EmbeddingModel(e.to_string()))?;

        Ok(encoding.get_ids().len())
    }

    /// Count tokens in multiple texts.
    pub fn count_batch(&self, texts: &[&str]) -> Result<Vec<usize>> {
        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), false)
            .map_err(|e| RagError::EmbeddingModel(e.to_string()))?;

        Ok(encodings.iter().map(|e| e.get_ids().len()).collect())
    }
}
```

---

## Performance Considerations

### Batching

```rust
impl OnnxEmbedder {
    /// Optimal batch size depends on available memory.
    ///
    /// Rough estimates for nomic-embed (768 dim, 8K context):
    /// - Batch 1:  ~200 MB peak
    /// - Batch 8:  ~500 MB peak
    /// - Batch 32: ~1.5 GB peak
    /// - Batch 64: ~3 GB peak
    pub fn recommended_batch_size(available_memory_gb: f32) -> usize {
        if available_memory_gb >= 4.0 {
            64
        } else if available_memory_gb >= 2.0 {
            32
        } else if available_memory_gb >= 1.0 {
            8
        } else {
            1
        }
    }
}
```

### Hardware Acceleration

```rust
impl OnnxEmbedder {
    /// Check available execution providers.
    pub fn available_providers() -> Vec<String> {
        ort::get_available_providers()
    }

    /// Check if GPU acceleration is available.
    pub fn has_gpu_support() -> bool {
        let providers = Self::available_providers();
        providers.iter().any(|p| {
            p.contains("CUDA") || p.contains("CoreML") || p.contains("DirectML")
        })
    }
}
```

### Throughput Estimates

| Batch Size | Hardware | Throughput (chunks/sec) | Memory |
|------------|----------|------------------------|--------|
| 1 | CPU (4 core) | ~5 | 200 MB |
| 8 | CPU (4 core) | ~25 | 500 MB |
| 32 | CPU (4 core) | ~50 | 1.5 GB |
| 32 | GPU (RTX 3080) | ~200 | 2 GB |
| 64 | GPU (RTX 3080) | ~300 | 3 GB |

**Confidence: MEDIUM** - Numbers are estimates; actual varies with text length and hardware.

---

## Error Handling

```rust
impl OnnxEmbedder {
    /// Embed with detailed error information.
    pub fn embed_with_errors(&self, texts: &[&str]) -> Vec<Result<Vec<f32>>> {
        texts.iter()
            .map(|text| {
                if text.trim().is_empty() {
                    return Err(RagError::EmptyText);
                }

                let token_count = self.count_tokens(text)?;
                if token_count > self.max_tokens {
                    return Err(RagError::TextTooLong {
                        tokens: token_count,
                        max: self.max_tokens,
                    });
                }

                let embeddings = self.embed_batch_internal(&[text], "search_document: ")?;
                Ok(embeddings.into_iter().next().unwrap())
            })
            .collect()
    }
}
```

---

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimension() {
        let embedder = OnnxEmbedder::new("./models/nomic-embed-text-v1.5").unwrap();
        assert_eq!(embedder.dimension(), 768);
    }

    #[test]
    fn test_single_embedding() {
        let embedder = OnnxEmbedder::new("./models/nomic-embed-text-v1.5").unwrap();
        let embedding = embedder.embed(&["Hello, world!"]).unwrap();

        assert_eq!(embedding.len(), 1);
        assert_eq!(embedding[0].len(), 768);

        // Check L2 normalization
        let norm: f32 = embedding[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_embedding() {
        let embedder = OnnxEmbedder::new("./models/nomic-embed-text-v1.5").unwrap();
        let texts = vec!["First text", "Second text", "Third text"];
        let embeddings = embedder.embed(&texts.iter().map(|s| *s).collect::<Vec<_>>()).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 768);
        }
    }

    #[test]
    fn test_query_vs_document() {
        let embedder = OnnxEmbedder::new("./models/nomic-embed-text-v1.5").unwrap();

        let text = "What is machine learning?";
        let doc_emb = embedder.embed(&[text]).unwrap();
        let query_emb = embedder.embed_query(text).unwrap();

        // Should be different due to different prefixes
        assert_ne!(doc_emb[0], query_emb);
    }

    #[test]
    fn test_similarity() {
        let embedder = OnnxEmbedder::new("./models/nomic-embed-text-v1.5").unwrap();

        let similar1 = "The cat sat on the mat";
        let similar2 = "A cat was sitting on the rug";
        let different = "The stock market crashed yesterday";

        let emb1 = embedder.embed(&[similar1]).unwrap();
        let emb2 = embedder.embed(&[similar2]).unwrap();
        let emb3 = embedder.embed(&[different]).unwrap();

        let sim_12 = cosine_similarity(&emb1[0], &emb2[0]);
        let sim_13 = cosine_similarity(&emb1[0], &emb3[0]);

        // Similar sentences should have higher similarity
        assert!(sim_12 > sim_13);
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn test_empty_text_error() {
        let embedder = OnnxEmbedder::new("./models/nomic-embed-text-v1.5").unwrap();
        let result = embedder.embed(&[""]);
        assert!(matches!(result, Err(RagError::EmptyText)));
    }

    #[test]
    fn test_token_counting() {
        let embedder = OnnxEmbedder::new("./models/nomic-embed-text-v1.5").unwrap();
        let count = embedder.count_tokens("Hello, world!").unwrap();
        assert!(count > 0 && count < 10);
    }
}
```

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|------------|-------|
| ONNX Runtime integration | HIGH | ort crate is mature |
| Tokenizer integration | HIGH | tokenizers crate from HF |
| nomic model compatibility | HIGH | Standard ONNX model |
| Mean pooling | HIGH | Standard algorithm |
| L2 normalization | HIGH | Simple math |
| Batch processing | HIGH | Standard pattern |
| Performance estimates | MEDIUM | Hardware-dependent |
| GPU acceleration | MEDIUM | Requires testing on target hardware |

---

## Cargo.toml

```toml
[package]
name = "rag-embed"
version = "0.1.0"
edition = "2021"
description = "ONNX-based embedding for the RAG system"
license = "MIT"

[dependencies]
rag-core = { path = "../rag-core" }
ort = { version = "2.0", features = ["load-dynamic"] }
tokenizers = { version = "0.15", features = [] }
ndarray = "0.15"
tracing = "0.1"

[dev-dependencies]
tokio = { version = "1.0", features = ["macros", "rt-multi-thread"] }

[features]
default = []
cuda = ["ort/cuda"]
coreml = ["ort/coreml"]
```

---

## Model Information

### nomic-embed-text-v1.5

| Property | Value |
|----------|-------|
| Dimensions | 768 |
| Max Tokens | 8192 |
| License | Apache 2.0 |
| Size | ~280 MB (ONNX) |
| Quality | Good (MTEB benchmark) |

**Key Features:**
- Long context (8K tokens)
- Separate prefixes for queries vs documents
- Good balance of quality and speed
- Fully local, no API needed

**Prefixes:**
- Document: `search_document: `
- Query: `search_query: `

This asymmetric encoding helps the model distinguish between queries (questions) and documents (answers).
