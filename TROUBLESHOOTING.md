# TROUBLESHOOTING.md

Common issues, failure modes, and their solutions.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Database Issues](#database-issues)
3. [Embedding Issues](#embedding-issues)
4. [Search Quality Issues](#search-quality-issues)
5. [Performance Issues](#performance-issues)
6. [Sync Issues](#sync-issues)
7. [MCP Issues](#mcp-issues)
8. [Memory Issues](#memory-issues)

---

## Installation Issues

### sqlite-vec Extension Not Found

**Symptoms:**
```
Error: unable to open database file
Error: no such module: vec0
```

**Causes:**
- sqlite-vec extension not installed
- Extension path not in library path
- SQLite version too old

**Solutions:**

1. **Install sqlite-vec:**
   ```bash
   # macOS
   brew install sqlite-vec

   # Linux (build from source)
   git clone https://github.com/asg017/sqlite-vec
   cd sqlite-vec
   make
   sudo make install
   ```

2. **Set library path:**
   ```bash
   # macOS
   export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

   # Linux
   export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
   ```

3. **Check SQLite version:**
   ```bash
   sqlite3 --version
   # Need 3.45+ for sqlite-vec
   ```

**Confidence: HIGH** - Well-documented installation process.

---

### ONNX Runtime Errors

**Symptoms:**
```
Error: ORT initialization failed
Error: CUDA not available
Error: Model file not found
```

**Causes:**
- ONNX Runtime not installed correctly
- Missing CUDA libraries (for GPU)
- Model file path incorrect

**Solutions:**

1. **Verify ONNX Runtime:**
   ```bash
   # Check if library is accessible
   ldconfig -p | grep onnxruntime
   ```

2. **CPU fallback:**
   ```rust
   // Force CPU execution provider
   let session = SessionBuilder::new(&environment)?
       .with_execution_providers([ExecutionProvider::CPU(Default::default())])?
       .commit_from_file(&model_path)?;
   ```

3. **Download model:**
   ```bash
   ./scripts/download-model.sh
   # Or manually:
   curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx" \
       -o models/nomic-embed-text-v1.5/model.onnx
   ```

**Confidence: HIGH** - Standard ONNX deployment issues.

---

### Tree-sitter Grammar Errors

**Symptoms:**
```
Error: Language not found
Error: Failed to parse file
Warning: Falling back to recursive chunking
```

**Causes:**
- Tree-sitter grammar not included
- Unsupported language version
- File encoding issues

**Solutions:**

1. **Check supported languages:**
   ```rust
   let supported = chunker.supported_types();
   println!("{:?}", supported);
   ```

2. **Add missing grammar:**
   ```toml
   # Cargo.toml
   [dependencies]
   tree-sitter-ruby = "0.21"  # Add missing language
   ```

3. **Force content type:**
   ```bash
   rag ingest file.xyz --content-type rust
   ```

**Confidence: HIGH** - Tree-sitter is well-documented.

---

## Database Issues

### Database Locked

**Symptoms:**
```
Error: database is locked
Error: SQLITE_BUSY
```

**Causes:**
- Multiple processes writing simultaneously
- Long-running transaction
- Connection not closed properly

**Solutions:**

1. **Check for other processes:**
   ```bash
   lsof /path/to/rag.db
   fuser /path/to/rag.db
   ```

2. **Increase busy timeout:**
   ```sql
   PRAGMA busy_timeout = 30000;  -- 30 seconds
   ```

3. **Use WAL mode (default):**
   ```sql
   PRAGMA journal_mode = WAL;
   ```

4. **Kill stuck processes:**
   ```bash
   # Find and kill
   pkill -f "rag-mcp"
   ```

**Confidence: HIGH** - Standard SQLite concurrency issue.

---

### Database Corruption

**Symptoms:**
```
Error: database disk image is malformed
Error: SQLITE_CORRUPT
```

**Causes:**
- Power loss during write
- Disk failure
- Bug in application

**Solutions:**

1. **Run integrity check:**
   ```sql
   PRAGMA integrity_check;
   ```

2. **Attempt recovery:**
   ```bash
   sqlite3 rag.db ".recover" | sqlite3 rag-recovered.db
   ```

3. **Restore from backup:**
   ```bash
   cp ~/.local/share/rag-mcp/backups/rag.db.bak rag.db
   ```

4. **Rebuild from scratch:**
   ```bash
   # If all else fails
   rm rag.db
   rag ingest ./original-files -c collection
   ```

**Confidence: MEDIUM** - Recovery may not be possible.

---

### Vector Index Issues

**Symptoms:**
```
Error: vec_chunks: invalid dimension
Error: Vector search returned no results
Warning: Slow vector search
```

**Causes:**
- Dimension mismatch (model changed)
- Index not built
- Corrupted index

**Solutions:**

1. **Check dimensions:**
   ```sql
   -- Should match embedding model dimension (768 for nomic)
   SELECT typeof(embedding), length(embedding)/4 as dims
   FROM vec_chunks LIMIT 1;
   ```

2. **Rebuild vector index:**
   ```sql
   -- Drop and recreate
   DROP TABLE IF EXISTS vec_chunks;
   CREATE VIRTUAL TABLE vec_chunks USING vec0(
       chunk_id TEXT PRIMARY KEY,
       embedding float[768] distance_metric=cosine
   );
   -- Re-insert embeddings from chunks
   ```

3. **Check for empty index:**
   ```sql
   SELECT COUNT(*) FROM vec_chunks;
   -- Should match chunk count
   ```

**Confidence: MEDIUM** - sqlite-vec recovery less documented.

---

## Embedding Issues

### Out of Memory During Embedding

**Symptoms:**
```
Error: OOM when allocating tensor
Error: CUDA out of memory
Process killed
```

**Causes:**
- Batch size too large
- Text too long
- Multiple models loaded

**Solutions:**

1. **Reduce batch size:**
   ```toml
   # config.toml
   [embedding]
   batch_size = 8  # Reduce from default 32
   ```

2. **Truncate long texts:**
   ```rust
   // Truncate to model max tokens
   let truncated = embedder.truncate_to_context(text)?;
   ```

3. **Use CPU for large batches:**
   ```rust
   // CPU has more memory than GPU
   .with_execution_providers([ExecutionProvider::CPU(Default::default())])?
   ```

**Confidence: HIGH** - Standard ML memory issues.

---

### Embedding Quality Issues

**Symptoms:**
- Search returns irrelevant results
- Similar documents have low similarity
- Different topics have high similarity

**Causes:**
- Wrong model for domain
- Text too short for meaningful embedding
- Missing document/query prefixes

**Solutions:**

1. **Verify prefix usage:**
   ```rust
   // Documents: "search_document: <text>"
   // Queries: "search_query: <text>"
   // These are REQUIRED for nomic-embed
   ```

2. **Check chunk sizes:**
   ```sql
   SELECT AVG(token_count), MIN(token_count), MAX(token_count)
   FROM chunks;
   -- Very short chunks (<20 tokens) may have poor embeddings
   ```

3. **Try different model:**
   - Consider BGE, E5, or other models for specific domains

**Confidence: MEDIUM** - Embedding quality is domain-dependent.

---

## Search Quality Issues

### No Results Returned

**Symptoms:**
```
Results: 0
Query: "my search term"
```

**Causes:**
- Empty index
- Collection filter excludes all
- Query too specific

**Solutions:**

1. **Check index contents:**
   ```bash
   rag stats
   # Verify documents and chunks > 0
   ```

2. **Try without collection filter:**
   ```bash
   rag search "query" --collection ""
   ```

3. **Use keyword search only:**
   ```bash
   rag search "query" --mode keyword
   ```

4. **Check FTS5 index:**
   ```sql
   SELECT * FROM chunks_fts WHERE chunks_fts MATCH 'your query';
   ```

**Confidence: HIGH** - Easy to diagnose.

---

### Irrelevant Results

**Symptoms:**
- Results don't match query intent
- High scores but wrong content
- Semantic search misses obvious matches

**Causes:**
- Poor chunking boundaries
- Hybrid alpha misconfigured
- Index stale

**Solutions:**

1. **Adjust hybrid alpha:**
   ```toml
   # config.toml
   [search]
   hybrid_alpha = 0.3  # More weight to keyword search
   ```

2. **Use query trace:**
   ```bash
   rag search "query" --trace
   # Examine which results come from vector vs keyword
   ```

3. **Review chunk quality:**
   ```bash
   rag document show <doc_id>
   # Check if chunks have good boundaries
   ```

4. **Re-ingest with different settings:**
   ```bash
   rag document delete <doc_id>
   rag ingest file --max-tokens 256  # Smaller chunks
   ```

**Confidence: MEDIUM** - Relevance tuning is iterative.

---

### Slow Search

**Symptoms:**
- Search takes >1 second
- Timeout errors
- UI unresponsive

**Causes:**
- Index too large
- No connection pooling
- Disk I/O bottleneck

**Solutions:**

1. **Check index size:**
   ```bash
   rag stats
   # >1M chunks may need optimization
   ```

2. **Run optimization:**
   ```bash
   rag optimize
   ```

3. **Use SSD storage:**
   ```bash
   # Move database to SSD
   mv ~/.local/share/rag-mcp/rag.db /ssd/rag.db
   # Update config
   ```

4. **Reduce result count:**
   ```bash
   rag search "query" -k 5  # Instead of default 10
   ```

**Confidence: HIGH** - Standard optimization techniques.

---

## Performance Issues

### High Memory Usage

**Symptoms:**
- Process uses >4GB RAM
- System swap active
- OOM killer invoked

**Causes:**
- Large embedding model in memory
- SQLite cache too large
- Memory leak

**Solutions:**

1. **Check memory breakdown:**
   ```bash
   # On Linux
   pmap -x $(pgrep rag-mcp)
   ```

2. **Reduce SQLite cache:**
   ```sql
   PRAGMA cache_size = -16000;  -- 16MB instead of 64MB
   ```

3. **Use smaller batch size:**
   ```toml
   [embedding]
   batch_size = 8
   ```

4. **Monitor for leaks:**
   ```bash
   while true; do
       ps -o rss= -p $(pgrep rag-mcp)
       sleep 60
   done
   ```

**Confidence: MEDIUM** - Memory tuning is system-specific.

---

### Slow Ingestion

**Symptoms:**
- <10 docs/second
- Embedding is bottleneck
- High CPU during ingest

**Causes:**
- No GPU acceleration
- Small batch size
- Disk I/O

**Solutions:**

1. **Enable GPU (if available):**
   ```bash
   # Check if CUDA available
   nvidia-smi
   ```

2. **Increase batch size:**
   ```toml
   [embedding]
   batch_size = 64
   ```

3. **Use async pipeline:**
   ```bash
   # Ingest in background
   rag ingest ./large-dir -c collection &
   ```

4. **Profile bottleneck:**
   ```bash
   RUST_LOG=debug rag ingest ./test
   # Look for slow stages
   ```

**Confidence: HIGH** - Ingestion is CPU/GPU bound.

---

## Sync Issues

### Peer Unreachable

**Symptoms:**
```
Error: connection refused
Error: timeout
Sync failed with peer
```

**Causes:**
- Peer not running
- Firewall blocking
- Wrong endpoint

**Solutions:**

1. **Check peer status:**
   ```bash
   curl http://peer:8765/sync/watermark
   ```

2. **Check firewall:**
   ```bash
   # Allow port
   sudo ufw allow 8765
   ```

3. **Verify endpoint:**
   ```bash
   rag sync status
   # Check configured endpoints
   ```

**Confidence: HIGH** - Standard networking issues.

---

### Sync Conflicts

**Symptoms:**
```
Warning: Conflict on document X
Sync completed with 5 conflicts
```

**Causes:**
- Concurrent edits on both nodes
- Clock drift
- Network partition recovery

**Solutions:**

1. **Review conflicts:**
   ```bash
   rag sync conflicts
   ```

2. **Manual resolution:**
   ```bash
   # Keep local version
   rag sync resolve <conflict_id> --keep-local
   ```

3. **Re-sync from scratch:**
   ```bash
   rag sync reset peer-alpha
   rag sync now peer-alpha
   ```

**Confidence: MEDIUM** - LWW may lose data.

---

## MCP Issues

### Claude Can't Find Server

**Symptoms:**
- Tools not appearing in Claude
- "Server not found" error
- Timeout on tool calls

**Causes:**
- MCP config incorrect
- Server not running
- Path issues

**Solutions:**

1. **Check MCP config:**
   ```json
   // ~/.config/claude-desktop/claude_desktop_config.json
   {
     "mcpServers": {
       "rag": {
         "command": "/full/path/to/rag-mcp",
         "args": []
       }
     }
   }
   ```

2. **Test server manually:**
   ```bash
   # Should respond on stdin
   echo '{"jsonrpc":"2.0","method":"initialize","id":1}' | rag-mcp
   ```

3. **Check logs:**
   ```bash
   # Claude Desktop logs
   tail -f ~/Library/Logs/Claude/claude_desktop.log
   ```

**Confidence: HIGH** - MCP protocol is well-defined.

---

### Tool Execution Fails

**Symptoms:**
```
Tool rag_search failed: ...
Internal error
```

**Causes:**
- Database path wrong
- Missing model
- Permission issues

**Solutions:**

1. **Check server config:**
   ```bash
   cat ~/.config/rag-mcp/config.toml
   ```

2. **Run server standalone:**
   ```bash
   RUST_LOG=debug rag serve
   # Watch for errors
   ```

3. **Check permissions:**
   ```bash
   ls -la ~/.local/share/rag-mcp/
   ```

**Confidence: HIGH** - Errors are logged.

---

## Memory Issues

### Memory Growth Over Time

**Symptoms:**
- Memory usage increases continuously
- Eventually runs out of memory
- Only happens after many operations

**Causes:**
- Connection pool growth
- Cache not evicting
- Actual memory leak

**Solutions:**

1. **Restart periodically:**
   ```bash
   # In cron or systemd
   0 * * * * systemctl restart rag-mcp
   ```

2. **Set cache limits:**
   ```toml
   [cache]
   max_entries = 1000
   ttl_seconds = 3600
   ```

3. **Report bug:**
   - Collect memory profile
   - Submit issue with reproduction steps

**Confidence: LOW** - Memory leaks require deep investigation.

---

## Diagnostic Commands

### Quick Health Check

```bash
# Check all components
rag stats
rag search "test" -k 1
rag collection list
```

### Database Diagnostics

```sql
-- Run in sqlite3 shell
PRAGMA integrity_check;
PRAGMA quick_check;
SELECT COUNT(*) FROM documents;
SELECT COUNT(*) FROM chunks;
SELECT COUNT(*) FROM vec_chunks;
SELECT COUNT(*) FROM chunks_fts;
```

### Performance Diagnostics

```bash
# Time a search
time rag search "test query" -k 10

# Check disk I/O
iostat -x 1 5

# Check memory
free -h
```

### Network Diagnostics

```bash
# Check sync endpoints
rag sync status

# Test connectivity
curl -v http://peer:8765/sync/watermark

# Check firewall
sudo iptables -L -n
```

---

## Getting Help

If these solutions don't resolve your issue:

1. **Search existing issues:**
   https://github.com/example/rag-mcp/issues

2. **Create new issue with:**
   - Error message (full text)
   - Steps to reproduce
   - System information (`rag --version`, OS, etc.)
   - Relevant logs

3. **Include diagnostics:**
   ```bash
   rag stats --json > diagnostics.json
   ```
