//! ONNX-based embedding model implementation.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ndarray::ArrayViewD;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use rag_core::{Embedder, RagError, Result};

/// nomic-embed-text-v1.5 configuration.
const EMBEDDING_DIM: usize = 768;
const MAX_TOKENS: usize = 8192;

/// Document prefix for asymmetric retrieval.
const DOCUMENT_PREFIX: &str = "search_document: ";

/// Query prefix for asymmetric retrieval.
const QUERY_PREFIX: &str = "search_query: ";

/// ONNX-based embedder using nomic-embed-text-v1.5 or compatible model.
pub struct OnnxEmbedder {
    /// ONNX inference session (wrapped in Mutex for interior mutability).
    session: Mutex<Session>,

    /// Tokenizer for the model.
    tokenizer: Arc<Tokenizer>,

    /// Embedding dimension.
    dimension: usize,

    /// Maximum token count.
    max_tokens: usize,
}

impl OnnxEmbedder {
    /// Create a new embedder from model and tokenizer paths.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `tokenizer_path` - Path to the tokenizer.json file
    pub fn new(model_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref();
        let tokenizer_path = tokenizer_path.as_ref();

        info!("Loading ONNX model from {:?}", model_path);

        // Initialize ONNX Runtime session
        let session = Session::builder()
            .map_err(|e| RagError::embedding(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RagError::embedding(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(4)
            .map_err(|e| RagError::embedding(format!("Failed to set thread count: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| RagError::embedding(format!("Failed to load model: {}", e)))?;

        info!("Loading tokenizer from {:?}", tokenizer_path);

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| RagError::embedding(format!("Failed to load tokenizer: {}", e)))?;

        info!(
            "Embedder initialized: dim={}, max_tokens={}",
            EMBEDDING_DIM, MAX_TOKENS
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: Arc::new(tokenizer),
            dimension: EMBEDDING_DIM,
            max_tokens: MAX_TOKENS,
        })
    }

    /// Create an embedder with custom dimensions (for testing/other models).
    pub fn with_config(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        dimension: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        let mut embedder = Self::new(model_path, tokenizer_path)?;
        embedder.dimension = dimension;
        embedder.max_tokens = max_tokens;
        Ok(embedder)
    }

    /// Embed a batch of texts with a given prefix.
    fn embed_batch(&self, texts: &[&str], prefix: &str) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Add prefix to each text
        let prefixed: Vec<String> = texts.iter().map(|t| format!("{}{}", prefix, t)).collect();
        let prefixed_refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();

        // Tokenize batch
        let encodings = self
            .tokenizer
            .encode_batch(prefixed_refs, true)
            .map_err(|e| RagError::embedding(format!("Tokenization failed: {}", e)))?;

        // Get max length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.max_tokens);

        let batch_size = encodings.len();

        debug!(
            "Embedding batch: size={}, max_len={}",
            batch_size, max_len
        );

        // Prepare input tensors
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = ids.len().min(max_len);

            for j in 0..len {
                input_ids[i * max_len + j] = ids[j] as i64;
                attention_mask[i * max_len + j] = mask[j] as i64;
            }
        }

        // Create tensors using ort's Tensor type
        let input_ids_tensor = Tensor::from_array((
            vec![batch_size, max_len],
            input_ids,
        ))
        .map_err(|e| RagError::embedding(format!("Failed to create input tensor: {}", e)))?;

        let attention_mask_tensor = Tensor::from_array((
            vec![batch_size, max_len],
            attention_mask,
        ))
        .map_err(|e| RagError::embedding(format!("Failed to create mask tensor: {}", e)))?;

        // Run inference using the inputs! macro
        let mut session = self
            .session
            .lock()
            .map_err(|e| RagError::embedding(format!("Failed to lock session: {}", e)))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor
            ])
            .map_err(|e| RagError::embedding(format!("Inference failed: {}", e)))?;

        // Extract embeddings from output
        // Models typically output (batch_size, seq_len, hidden_dim)
        // We need to do mean pooling over the sequence dimension

        // Get the first output (different models name outputs differently)
        let output_names: Vec<_> = outputs.iter().map(|(k, _)| k.to_string()).collect();
        debug!("Output names: {:?}", output_names);

        let (_, output) = outputs
            .iter()
            .next()
            .ok_or_else(|| RagError::embedding("No output tensor found"))?;

        // Use try_extract_array to get an ndarray view
        let view = output
            .try_extract_array::<f32>()
            .map_err(|e| RagError::embedding(format!("Failed to extract tensor: {}", e)))?;

        let shape_dims: Vec<usize> = view.shape().to_vec();
        debug!("Output shape: {:?}", shape_dims);

        // Handle different output shapes
        let embeddings = if shape_dims.len() == 3 {
            // (batch_size, seq_len, hidden_dim) - need mean pooling
            self.mean_pool_3d_ndarray(&view, &encodings, max_len)?
        } else if shape_dims.len() == 2 {
            // (batch_size, hidden_dim) - already pooled
            let hidden_dim = shape_dims[1];
            (0..batch_size)
                .map(|i| {
                    let embedding: Vec<f32> = (0..hidden_dim)
                        .map(|j| view[[i, j]])
                        .collect();
                    self.l2_normalize(embedding)
                })
                .collect()
        } else {
            return Err(RagError::embedding(format!(
                "Unexpected output shape: {:?}",
                shape_dims
            )));
        };

        Ok(embeddings)
    }

    /// Mean pooling over sequence dimension with attention mask.
    ///
    /// Works with ndarray view of shape [batch, seq, hidden]
    fn mean_pool_3d_ndarray(
        &self,
        tensor: &ArrayViewD<'_, f32>,
        encodings: &[tokenizers::Encoding],
        max_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let shape = tensor.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden_dim = shape[2];

        let mut embeddings = Vec::with_capacity(batch_size);

        for (i, encoding) in encodings.iter().enumerate() {
            let attention_mask = encoding.get_attention_mask();
            let valid_len = attention_mask.iter().take(max_len).filter(|&&m| m == 1).count();

            if valid_len == 0 {
                embeddings.push(vec![0.0; hidden_dim]);
                continue;
            }

            // Sum embeddings for valid tokens
            let mut sum = vec![0.0f32; hidden_dim];
            for j in 0..valid_len.min(max_len).min(seq_len) {
                if j < attention_mask.len() && attention_mask[j] == 1 {
                    for k in 0..hidden_dim {
                        sum[k] += tensor[[i, j, k]];
                    }
                }
            }

            // Compute mean
            let embedding: Vec<f32> = sum.iter().map(|s| s / valid_len as f32).collect();

            // L2 normalize
            embeddings.push(self.l2_normalize(embedding));
        }

        Ok(embeddings)
    }

    /// L2 normalize a vector.
    fn l2_normalize(&self, mut v: Vec<f32>) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

#[async_trait]
impl Embedder for OnnxEmbedder {
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Run embedding synchronously (Session is not Send)
        // In production, consider a dedicated embedder thread
        self.embed_batch(texts, DOCUMENT_PREFIX)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let texts = [text];
        let results = self.embed_batch(&texts, QUERY_PREFIX)?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| RagError::embedding("No embedding returned"))
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| RagError::embedding(format!("Tokenization failed: {}", e)))?;
        Ok(encoding.get_ids().len())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

/// A mock embedder for testing that doesn't require actual models.
pub struct MockEmbedder {
    dimension: usize,
    max_tokens: usize,
}

impl MockEmbedder {
    /// Create a new mock embedder with default settings.
    pub fn new() -> Self {
        Self {
            dimension: 768,
            max_tokens: 8192,
        }
    }

    /// Create a mock embedder with custom settings.
    pub fn with_config(dimension: usize, max_tokens: usize) -> Self {
        Self { dimension, max_tokens }
    }
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // Return deterministic mock embeddings based on text hash
        Ok(texts
            .iter()
            .map(|text| {
                let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
                let mut embedding = vec![0.0f32; self.dimension];
                for (i, v) in embedding.iter_mut().enumerate() {
                    *v = ((hash.wrapping_mul(i as u64 + 1)) as f32 % 1000.0) / 1000.0 - 0.5;
                }
                // L2 normalize
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut embedding {
                        *x /= norm;
                    }
                }
                embedding
            })
            .collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_documents(&[text]).await?;
        Ok(results.into_iter().next().unwrap())
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        // Rough approximation: ~4 chars per token
        Ok(text.len() / 4 + 1)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedder() {
        let embedder = MockEmbedder::new();

        assert_eq!(embedder.dimension(), 768);
        assert_eq!(embedder.max_tokens(), 8192);

        let texts = ["Hello world", "Rust is great"];
        let embeddings = embedder.embed_documents(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 768);
        assert_eq!(embeddings[1].len(), 768);

        // Check L2 normalization
        let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_mock_embedder_query() {
        let embedder = MockEmbedder::new();

        let embedding = embedder.embed_query("test query").await.unwrap();
        assert_eq!(embedding.len(), 768);

        // Check L2 normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_count_tokens() {
        let embedder = MockEmbedder::new();
        let count = embedder.count_tokens("Hello world, this is a test").unwrap();
        assert!(count > 0);
    }

    #[tokio::test]
    async fn test_deterministic_embeddings() {
        let embedder = MockEmbedder::new();

        let text = "consistent input";
        let e1 = embedder.embed_query(text).await.unwrap();
        let e2 = embedder.embed_query(text).await.unwrap();

        // Same input should produce same output
        assert_eq!(e1, e2);
    }

    #[tokio::test]
    async fn test_different_texts_different_embeddings() {
        let embedder = MockEmbedder::new();

        let e1 = embedder.embed_query("hello").await.unwrap();
        let e2 = embedder.embed_query("world").await.unwrap();

        // Different inputs should produce different outputs
        assert_ne!(e1, e2);
    }

    #[tokio::test]
    async fn test_mock_embedder_custom_config() {
        let embedder = MockEmbedder::with_config(384, 512);

        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.max_tokens(), 512);

        let embedding = embedder.embed_query("test").await.unwrap();
        assert_eq!(embedding.len(), 384);
    }
}
