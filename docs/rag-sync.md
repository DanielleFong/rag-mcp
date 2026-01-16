# rag-sync

Multi-node synchronization with eventual consistency via HTTP/REST and Hybrid Logical Clocks.

## Overview

| Attribute | Value |
|-----------|-------|
| Purpose | Sync data between multiple RAG nodes |
| Dependencies | reqwest, tokio, rag-core, rag-store |
| Est. Lines | ~1,200 |
| Confidence | MEDIUM |

This crate implements peer-to-peer synchronization with last-writer-wins conflict resolution.

---

## Module Structure

```
rag-sync/
├── src/
│   ├── lib.rs           # Public exports
│   ├── manager.rs       # SyncManager orchestration
│   ├── peer.rs          # HttpSyncPeer implementation
│   ├── server.rs        # Sync HTTP server endpoints
│   ├── protocol.rs      # Sync protocol types
│   ├── conflict.rs      # Conflict resolution
│   └── scheduler.rs     # Background sync scheduling
└── Cargo.toml
```

---

## SyncManager

The main orchestrator for synchronization.

```rust
use rag_core::{Store, HybridLogicalClock, Change, ChangeSet, Result, RagError};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Manages synchronization with multiple peers.
pub struct SyncManager {
    /// Local store.
    store: Arc<dyn Store>,

    /// Configured peers.
    peers: RwLock<Vec<Arc<dyn SyncPeer>>>,

    /// Local node ID.
    node_id: u16,

    /// Sync configuration.
    config: SyncConfig,

    /// Background scheduler handle.
    scheduler: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Interval between sync attempts (seconds).
    pub interval_seconds: u64,

    /// Maximum changes per sync batch.
    pub batch_size: usize,

    /// Timeout for peer requests.
    pub request_timeout: Duration,

    /// Number of retries for failed syncs.
    pub max_retries: u32,

    /// Enable automatic background sync.
    pub auto_sync: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            interval_seconds: 60,
            batch_size: 1000,
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
            auto_sync: true,
        }
    }
}

impl SyncManager {
    pub fn new(
        store: Arc<dyn Store>,
        node_id: u16,
        config: SyncConfig,
    ) -> Self {
        Self {
            store,
            peers: RwLock::new(Vec::new()),
            node_id,
            config,
            scheduler: None,
        }
    }

    /// Add a sync peer.
    pub async fn add_peer(&self, peer: Arc<dyn SyncPeer>) {
        let mut peers = self.peers.write().await;
        peers.push(peer);
    }

    /// Remove a sync peer.
    pub async fn remove_peer(&self, peer_id: &str) {
        let mut peers = self.peers.write().await;
        peers.retain(|p| p.peer_id() != peer_id);
    }

    /// Start background sync scheduler.
    pub fn start_background_sync(&mut self) {
        if !self.config.auto_sync || self.scheduler.is_some() {
            return;
        }

        let store = self.store.clone();
        let peers = self.peers.clone();
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                Duration::from_secs(config.interval_seconds)
            );

            loop {
                interval.tick().await;

                let peers_guard = peers.read().await;
                for peer in peers_guard.iter() {
                    if let Err(e) = Self::sync_with_peer(&store, peer.as_ref(), &config).await {
                        tracing::warn!("Sync with {} failed: {}", peer.peer_id(), e);
                    }
                }
            }
        });

        self.scheduler = Some(handle);
    }

    /// Stop background sync.
    pub fn stop_background_sync(&mut self) {
        if let Some(handle) = self.scheduler.take() {
            handle.abort();
        }
    }

    /// Manually trigger sync with all peers.
    pub async fn sync_all(&self) -> Vec<SyncResult> {
        let peers = self.peers.read().await;
        let mut results = Vec::new();

        for peer in peers.iter() {
            let result = Self::sync_with_peer(&self.store, peer.as_ref(), &self.config).await;
            results.push(SyncResult {
                peer_id: peer.peer_id().to_string(),
                result,
            });
        }

        results
    }

    /// Sync with a specific peer.
    pub async fn sync_with(&self, peer_id: &str) -> Result<SyncStats> {
        let peers = self.peers.read().await;
        let peer = peers.iter()
            .find(|p| p.peer_id() == peer_id)
            .ok_or_else(|| RagError::PeerUnreachable(peer_id.to_string()))?;

        Self::sync_with_peer(&self.store, peer.as_ref(), &self.config).await
    }

    /// Internal sync implementation.
    async fn sync_with_peer(
        store: &Arc<dyn Store>,
        peer: &dyn SyncPeer,
        config: &SyncConfig,
    ) -> Result<SyncStats> {
        let mut stats = SyncStats::default();
        let start = std::time::Instant::now();

        // 1. Get peer's watermark
        let peer_watermark = peer.get_watermark().await?;
        let local_watermark = store.get_watermark().await?;

        // 2. Pull changes from peer
        let mut has_more = true;
        let mut pull_since = local_watermark;

        while has_more {
            let changes = peer.get_changes_since(pull_since, config.batch_size).await?;
            stats.pulled += changes.changes.len();

            if !changes.changes.is_empty() {
                let conflicts = store.apply_changes(changes.changes).await?;
                stats.conflicts += conflicts.len();
            }

            has_more = changes.has_more;
            pull_since = changes.watermark;
        }

        // 3. Push local changes to peer
        let mut push_since = peer_watermark;
        has_more = true;

        while has_more {
            let local_changes = store.get_changes_since(push_since).await?;

            if local_changes.is_empty() {
                break;
            }

            let batch: Vec<_> = local_changes.into_iter()
                .take(config.batch_size)
                .collect();

            let result = peer.push_changes(ChangeSet {
                changes: batch.clone(),
                watermark: store.get_watermark().await?,
                has_more: false,
            }).await?;

            stats.pushed += batch.len();
            stats.conflicts += result.conflicts.len();

            if batch.len() < config.batch_size {
                has_more = false;
            } else {
                // Update watermark for next batch
                push_since = result.new_watermark;
            }
        }

        stats.duration = start.elapsed();
        Ok(stats)
    }
}

/// Result of a sync operation.
pub struct SyncResult {
    pub peer_id: String,
    pub result: Result<SyncStats>,
}

/// Statistics from a sync operation.
#[derive(Debug, Default)]
pub struct SyncStats {
    pub pulled: usize,
    pub pushed: usize,
    pub conflicts: usize,
    pub duration: Duration,
}
```

---

## HTTP Sync Peer

Client implementation for communicating with remote peers.

```rust
use reqwest::Client;

/// HTTP-based sync peer implementation.
pub struct HttpSyncPeer {
    peer_id: String,
    endpoint: String,
    client: Client,
    timeout: Duration,
}

impl HttpSyncPeer {
    pub fn new(peer_id: String, endpoint: String, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            peer_id,
            endpoint,
            client,
            timeout,
        }
    }
}

#[async_trait]
impl SyncPeer for HttpSyncPeer {
    fn peer_id(&self) -> &str {
        &self.peer_id
    }

    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    async fn get_watermark(&self) -> Result<HybridLogicalClock> {
        let url = format!("{}/sync/watermark", self.endpoint);

        let response = self.client.get(&url)
            .send()
            .await
            .map_err(|e| RagError::SyncFailed {
                peer: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(RagError::SyncFailed {
                peer: self.peer_id.clone(),
                reason: format!("HTTP {}", response.status()),
            });
        }

        let body: WatermarkResponse = response.json().await
            .map_err(|e| RagError::SyncFailed {
                peer: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        HybridLogicalClock::from_hex(&body.hlc)
            .map_err(|_| RagError::InvalidHlc(body.hlc))
    }

    async fn get_changes_since(
        &self,
        hlc: HybridLogicalClock,
        limit: usize,
    ) -> Result<ChangeSet> {
        let url = format!(
            "{}/sync/changes?since={}&limit={}",
            self.endpoint,
            hlc.to_hex(),
            limit
        );

        let response = self.client.get(&url)
            .send()
            .await
            .map_err(|e| RagError::SyncFailed {
                peer: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(RagError::SyncFailed {
                peer: self.peer_id.clone(),
                reason: format!("HTTP {}", response.status()),
            });
        }

        let body: ChangesResponse = response.json().await
            .map_err(|e| RagError::SyncFailed {
                peer: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        Ok(ChangeSet {
            changes: body.changes,
            watermark: HybridLogicalClock::from_hex(&body.watermark)
                .map_err(|_| RagError::InvalidHlc(body.watermark.clone()))?,
            has_more: body.has_more,
        })
    }

    async fn push_changes(&self, changes: ChangeSet) -> Result<PushResult> {
        let url = format!("{}/sync/changes", self.endpoint);

        let request_body = PushChangesRequest {
            changes: changes.changes,
            source_watermark: changes.watermark.to_hex(),
        };

        let response = self.client.post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| RagError::SyncFailed {
                peer: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(RagError::SyncFailed {
                peer: self.peer_id.clone(),
                reason: format!("HTTP {}", response.status()),
            });
        }

        let body: PushChangesResponse = response.json().await
            .map_err(|e| RagError::SyncFailed {
                peer: self.peer_id.clone(),
                reason: e.to_string(),
            })?;

        Ok(PushResult {
            accepted: body.accepted,
            conflicts: body.conflicts,
            new_watermark: HybridLogicalClock::from_hex(&body.new_watermark)
                .map_err(|_| RagError::InvalidHlc(body.new_watermark))?,
        })
    }
}

// Protocol types for HTTP
#[derive(Serialize, Deserialize)]
struct WatermarkResponse {
    hlc: String,
    node_id: String,
}

#[derive(Serialize, Deserialize)]
struct ChangesResponse {
    changes: Vec<Change>,
    watermark: String,
    has_more: bool,
}

#[derive(Serialize, Deserialize)]
struct PushChangesRequest {
    changes: Vec<Change>,
    source_watermark: String,
}

#[derive(Serialize, Deserialize)]
struct PushChangesResponse {
    accepted: bool,
    conflicts: Vec<ConflictReport>,
    new_watermark: String,
}
```

---

## Sync Server

HTTP endpoints for receiving sync requests.

```rust
use axum::{Router, routing::{get, post}, extract::{State, Query}, Json};

/// Sync server state.
pub struct SyncServerState {
    store: Arc<dyn Store>,
    node_id: u16,
}

/// Create sync router.
pub fn sync_router(state: Arc<SyncServerState>) -> Router {
    Router::new()
        .route("/sync/watermark", get(get_watermark))
        .route("/sync/changes", get(get_changes).post(push_changes))
        .with_state(state)
}

/// GET /sync/watermark
async fn get_watermark(
    State(state): State<Arc<SyncServerState>>,
) -> Result<Json<WatermarkResponse>, SyncError> {
    let hlc = state.store.get_watermark().await?;

    Ok(Json(WatermarkResponse {
        hlc: hlc.to_hex(),
        node_id: format!("{:04x}", state.node_id),
    }))
}

/// Query parameters for GET /sync/changes
#[derive(Deserialize)]
struct GetChangesQuery {
    since: String,
    limit: Option<usize>,
}

/// GET /sync/changes?since={hlc}&limit={n}
async fn get_changes(
    State(state): State<Arc<SyncServerState>>,
    Query(query): Query<GetChangesQuery>,
) -> Result<Json<ChangesResponse>, SyncError> {
    let since = HybridLogicalClock::from_hex(&query.since)
        .map_err(|_| SyncError::InvalidHlc)?;

    let limit = query.limit.unwrap_or(1000).min(10000);

    let changes = state.store.get_changes_since(since).await?;
    let has_more = changes.len() > limit;

    let batch: Vec<_> = changes.into_iter().take(limit).collect();
    let watermark = state.store.get_watermark().await?;

    Ok(Json(ChangesResponse {
        changes: batch,
        watermark: watermark.to_hex(),
        has_more,
    }))
}

/// POST /sync/changes
async fn push_changes(
    State(state): State<Arc<SyncServerState>>,
    Json(request): Json<PushChangesRequest>,
) -> Result<Json<PushChangesResponse>, SyncError> {
    let conflicts = state.store.apply_changes(request.changes).await?;
    let new_watermark = state.store.get_watermark().await?;

    Ok(Json(PushChangesResponse {
        accepted: true,
        conflicts,
        new_watermark: new_watermark.to_hex(),
    }))
}

/// Sync-specific error type.
#[derive(Debug)]
pub enum SyncError {
    Store(RagError),
    InvalidHlc,
}

impl From<RagError> for SyncError {
    fn from(e: RagError) -> Self {
        SyncError::Store(e)
    }
}

impl axum::response::IntoResponse for SyncError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            SyncError::Store(e) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                e.to_string(),
            ),
            SyncError::InvalidHlc => (
                axum::http::StatusCode::BAD_REQUEST,
                "Invalid HLC format".to_string(),
            ),
        };

        (status, message).into_response()
    }
}
```

---

## Conflict Resolution

```rust
/// Conflict resolution strategies.
#[derive(Debug, Clone, Copy)]
pub enum ConflictStrategy {
    /// Last write wins based on HLC.
    LastWriterWins,

    /// Keep local version.
    PreferLocal,

    /// Take remote version.
    PreferRemote,

    /// Merge changes (for collections only).
    Merge,
}

impl Default for ConflictStrategy {
    fn default() -> Self {
        Self::LastWriterWins
    }
}

/// Resolve a conflict between local and remote changes.
pub fn resolve_conflict(
    local: &Change,
    remote: &Change,
    strategy: ConflictStrategy,
) -> ConflictResolution {
    match strategy {
        ConflictStrategy::LastWriterWins => {
            let local_hlc = get_change_hlc(local);
            let remote_hlc = get_change_hlc(remote);

            if local_hlc >= remote_hlc {
                ConflictResolution::KeepLocal
            } else {
                ConflictResolution::TakeRemote
            }
        }

        ConflictStrategy::PreferLocal => ConflictResolution::KeepLocal,
        ConflictStrategy::PreferRemote => ConflictResolution::TakeRemote,

        ConflictStrategy::Merge => {
            // Only applicable for collection settings
            match (local, remote) {
                (
                    Change::CollectionCreate { collection: local_coll },
                    Change::CollectionCreate { collection: remote_coll },
                ) => {
                    // Merge settings
                    ConflictResolution::Merged(Change::CollectionCreate {
                        collection: Collection {
                            settings: merge_settings(
                                &local_coll.settings,
                                &remote_coll.settings,
                            ),
                            ..local_coll.clone()
                        },
                    })
                }
                _ => {
                    // Fall back to LWW for non-mergeable changes
                    let local_hlc = get_change_hlc(local);
                    let remote_hlc = get_change_hlc(remote);

                    if local_hlc >= remote_hlc {
                        ConflictResolution::KeepLocal
                    } else {
                        ConflictResolution::TakeRemote
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum ConflictResolution {
    KeepLocal,
    TakeRemote,
    Merged(Change),
}

fn get_change_hlc(change: &Change) -> HybridLogicalClock {
    match change {
        Change::CollectionCreate { collection } => collection.hlc,
        Change::CollectionDelete { hlc, .. } => *hlc,
        Change::DocumentInsert { document, .. } => document.hlc,
        Change::DocumentUpdate { document, .. } => document.hlc,
        Change::DocumentDelete { hlc, .. } => *hlc,
    }
}

fn merge_settings(
    local: &CollectionSettings,
    remote: &CollectionSettings,
) -> CollectionSettings {
    // Simple merge: prefer larger values
    CollectionSettings {
        chunking: ChunkingSettings {
            max_tokens: local.chunking.max_tokens.max(remote.chunking.max_tokens),
            min_tokens: local.chunking.min_tokens.min(remote.chunking.min_tokens),
            overlap_tokens: local.chunking.overlap_tokens.max(remote.chunking.overlap_tokens),
        },
        search: SearchSettings {
            top_k: local.search.top_k.max(remote.search.top_k),
            hybrid: local.search.hybrid || remote.search.hybrid,
            hybrid_alpha: (local.search.hybrid_alpha + remote.search.hybrid_alpha) / 2.0,
        },
    }
}
```

---

## Edge Cases and Failure Modes

### Network Partition

```rust
impl SyncManager {
    /// Handle network partition recovery.
    pub async fn recover_from_partition(&self, peer_id: &str) -> Result<SyncStats> {
        // When a peer comes back online after a partition:
        // 1. Full sync from the peer's last known watermark
        // 2. May result in many conflicts if both sides were active

        tracing::info!("Recovering from partition with {}", peer_id);

        // Reset peer's last sync watermark to force full sync
        self.store.reset_peer_watermark(peer_id).await?;

        // Perform full sync
        self.sync_with(peer_id).await
    }
}
```

### Concurrent Modifications

```rust
impl SyncManager {
    /// Handle concurrent modifications during sync.
    ///
    /// If local changes occur during a sync operation, they may be
    /// missed in the current sync cycle. The next sync will pick them up.
    pub async fn sync_atomic(&self, peer_id: &str) -> Result<SyncStats> {
        // Acquire write lock to prevent concurrent modifications
        // This is optional and trades availability for consistency
        let _lock = self.store.acquire_write_lock().await?;

        self.sync_with(peer_id).await
    }
}
```

### Large Change Sets

```rust
impl SyncManager {
    /// Stream large change sets to avoid memory issues.
    pub async fn sync_streaming(&self, peer_id: &str) -> Result<SyncStats> {
        let peers = self.peers.read().await;
        let peer = peers.iter()
            .find(|p| p.peer_id() == peer_id)
            .ok_or_else(|| RagError::PeerUnreachable(peer_id.to_string()))?;

        let mut stats = SyncStats::default();
        let mut batch_count = 0;

        // Pull in batches
        let mut pull_since = self.store.get_peer_watermark(peer_id).await?;

        loop {
            let changes = peer.get_changes_since(pull_since, self.config.batch_size).await?;

            if changes.changes.is_empty() {
                break;
            }

            stats.pulled += changes.changes.len();
            let conflicts = self.store.apply_changes(changes.changes).await?;
            stats.conflicts += conflicts.len();

            pull_since = changes.watermark;
            batch_count += 1;

            if !changes.has_more {
                break;
            }

            // Yield to allow other tasks
            tokio::task::yield_now().await;
        }

        tracing::info!("Pulled {} batches from {}", batch_count, peer_id);

        // Push similarly...

        Ok(stats)
    }
}
```

---

## Confidence Assessment

| Component | Confidence | Notes |
|-----------|------------|-------|
| HLC algorithm | HIGH | Well-documented |
| HTTP protocol | HIGH | Standard REST |
| LWW resolution | HIGH | Simple and predictable |
| Batch sync | HIGH | Standard pattern |
| Partition recovery | MEDIUM | May need tuning |
| Streaming sync | MEDIUM | Edge cases possible |
| Merge strategy | LOW | Limited applicability |

### Known Limitations

1. **No tombstone garbage collection**: Deleted items stay in sync log forever
2. **No vector clock**: Cannot detect concurrent edits (only ordering)
3. **LWW data loss**: Concurrent edits on different nodes may lose data
4. **No partial sync**: Must sync entire collections, not individual documents

### Future Improvements

1. Implement tombstone expiration
2. Add operation-based CRDTs for settings
3. Implement delta compression for large documents
4. Add sync priority for important collections

---

## Cargo.toml

```toml
[package]
name = "rag-sync"
version = "0.1.0"
edition = "2021"
description = "Multi-node synchronization for the RAG system"
license = "MIT"

[dependencies]
rag-core = { path = "../rag-core" }
reqwest = { version = "0.11", features = ["json"] }
axum = "0.7"
tokio = { version = "1.0", features = ["sync", "time", "macros"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
async-trait = "0.1"
tracing = "0.1"

[dev-dependencies]
tokio = { version = "1.0", features = ["macros", "rt-multi-thread"] }
wiremock = "0.5"
```
