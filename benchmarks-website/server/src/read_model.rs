// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Materialized read model for the hot benchmark website paths.
//!
//! DuckDB remains the source of truth, but the normal latest-100 website
//! payloads are deterministic between ingests. This module builds those
//! payloads once per database snapshot, stores identity/gzip/brotli bytes in
//! memory, and lets handlers serve bytes directly on the hot path.

use std::future::Future;
use std::hash::Hasher as _;
use std::io::Read as _;
use std::io::Write as _;
use std::num::NonZeroU32;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Context as _;
use anyhow::Result;
use axum::body::Body;
use axum::http::HeaderMap;
use axum::http::HeaderValue;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::Response;
use bytes::Bytes;
use duckdb::Connection;
use flate2::Compression;
use flate2::write::GzEncoder;
use parking_lot::RwLock;
use serde::Serialize;
use tokio::sync::Mutex as AsyncMutex;
use twox_hash::XxHash64;
use vortex_utils::aliases::hash_map::HashMap;

use crate::api;
use crate::api::ChartResponse;
use crate::api::CommitWindow;
use crate::api::FilterUniverse;
use crate::api::Group;
use crate::api::GroupChartsResponse;
use crate::api::GroupsResponse;
use crate::api::NamedChartResponse;
use crate::api::Summary;
use crate::db;
use crate::db::DbHandle;
use crate::slug::ChartKey;

/// Number of charts included in one materialized group shard response.
pub const GROUP_SHARD_CHARTS: usize = 8;

/// Cache policy for a materialized artifact route.
#[derive(Debug, Clone, Copy)]
pub enum ArtifactCachePolicy {
    /// Stable URLs such as `/api/groups`; browsers should revalidate.
    Revalidate,
    /// Versioned URLs under `/api/artifacts/{generation}/...`.
    Immutable,
}

impl ArtifactCachePolicy {
    fn header_value(self) -> &'static str {
        match self {
            Self::Revalidate => "no-cache, max-age=0, must-revalidate",
            Self::Immutable => "public, max-age=31536000, immutable",
        }
    }
}

/// A JSON artifact encoded in every representation the server wants to serve.
#[derive(Debug, Clone)]
pub struct EncodedArtifact {
    identity: Bytes,
    gzip: Bytes,
    br: Bytes,
    etag: HeaderValue,
}

impl EncodedArtifact {
    fn new(generation_id: &str, identity: Vec<u8>) -> Result<Self> {
        let gzip = gzip_bytes(&identity).context("gzip artifact")?;
        let br = brotli_bytes(&identity).context("brotli artifact")?;
        let etag = HeaderValue::from_str(&format!("\"{generation_id}\""))
            .context("building artifact ETag")?;
        Ok(Self {
            identity: Bytes::from(identity),
            gzip: Bytes::from(gzip),
            br: Bytes::from(br),
            etag,
        })
    }

    /// Uncompressed bytes, used when an HTML page embeds a single chart.
    pub fn identity(&self) -> &Bytes {
        &self.identity
    }

    /// Build an Axum response using the client's `Accept-Encoding` and
    /// `If-None-Match` headers.
    pub fn response(&self, request_headers: &HeaderMap, policy: ArtifactCachePolicy) -> Response {
        if if_none_match_matches(request_headers, &self.etag) {
            return artifact_response_builder(StatusCode::NOT_MODIFIED, policy, &self.etag)
                .body(Body::empty())
                .expect("artifact 304 response");
        }

        let (encoding, bytes) = match preferred_encoding(request_headers) {
            ArtifactEncoding::Brotli => (Some("br"), self.br.clone()),
            ArtifactEncoding::Gzip => (Some("gzip"), self.gzip.clone()),
            ArtifactEncoding::Identity => (None, self.identity.clone()),
        };

        let mut builder = artifact_response_builder(StatusCode::OK, policy, &self.etag)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, bytes.len().to_string());
        if let Some(encoding) = encoding {
            builder = builder.header(header::CONTENT_ENCODING, encoding);
        }
        builder.body(Body::from(bytes)).expect("artifact response")
    }
}

fn artifact_response_builder(
    status: StatusCode,
    policy: ArtifactCachePolicy,
    etag: &HeaderValue,
) -> axum::http::response::Builder {
    Response::builder()
        .status(status)
        .header(header::CACHE_CONTROL, policy.header_value())
        .header(header::VARY, "Accept-Encoding")
        .header(header::ETAG, etag.clone())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArtifactEncoding {
    Brotli,
    Gzip,
    Identity,
}

fn preferred_encoding(headers: &HeaderMap) -> ArtifactEncoding {
    let Some(raw) = headers
        .get(header::ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
    else {
        return ArtifactEncoding::Identity;
    };
    if accepts_encoding(raw, "br") {
        ArtifactEncoding::Brotli
    } else if accepts_encoding(raw, "gzip") {
        ArtifactEncoding::Gzip
    } else {
        ArtifactEncoding::Identity
    }
}

fn accepts_encoding(raw: &str, expected: &str) -> bool {
    raw.split(',').any(|part| {
        let mut pieces = part.trim().split(';');
        let name = pieces.next().unwrap_or_default().trim();
        if !name.eq_ignore_ascii_case(expected) {
            return false;
        }
        !pieces.any(|piece| {
            let piece = piece.trim();
            piece
                .strip_prefix("q=")
                .is_some_and(|q| q.trim().starts_with('0'))
        })
    })
}

fn if_none_match_matches(headers: &HeaderMap, etag: &HeaderValue) -> bool {
    let Some(raw) = headers
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    let etag = etag.to_str().unwrap_or_default();
    raw.split(',').any(|candidate| {
        let candidate = candidate.trim();
        candidate == "*" || candidate == etag
    })
}

fn gzip_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes)?;
    Ok(encoder.finish()?)
}

fn brotli_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut reader = brotli::CompressorReader::new(bytes, 4096, 5, 22);
    let mut out = Vec::new();
    reader.read_to_end(&mut out)?;
    Ok(out)
}

type RebuildFuture = Pin<Box<dyn Future<Output = Result<ReadGeneration>> + Send>>;
type RebuildTask = Arc<dyn Fn() -> RebuildFuture + Send + Sync>;

/// Shared in-memory store for the active and previous read generations.
#[derive(Clone)]
pub struct ReadStore {
    inner: Arc<RwLock<ReadStoreInner>>,
    rebuild: Arc<AsyncMutex<RebuildState>>,
}

struct ReadStoreInner {
    active: Arc<ReadGeneration>,
    previous: Option<Arc<ReadGeneration>>,
}

#[derive(Default)]
struct RebuildState {
    running: bool,
    pending: bool,
}

impl ReadStore {
    /// Build the first generation synchronously during startup. Failure here
    /// fails server startup rather than pushing a cold build onto users.
    pub fn build_initial(db: &DbHandle) -> Result<Self> {
        let mut conn = db.connection()?;
        let generation = Arc::new(build_generation(&mut conn)?);
        Ok(Self {
            inner: Arc::new(RwLock::new(ReadStoreInner {
                active: generation,
                previous: None,
            })),
            rebuild: Arc::new(AsyncMutex::new(RebuildState::default())),
        })
    }

    /// Current generation.
    pub fn active(&self) -> Arc<ReadGeneration> {
        Arc::clone(&self.inner.read().active)
    }

    /// Find the active or retained previous generation by id.
    pub fn generation(&self, id: &str) -> Option<Arc<ReadGeneration>> {
        let inner = self.inner.read();
        if inner.active.id == id {
            return Some(Arc::clone(&inner.active));
        }
        inner
            .previous
            .as_ref()
            .filter(|generation| generation.id == id)
            .map(Arc::clone)
    }

    /// Schedule a background rebuild after ingest. The active generation is
    /// retained until the rebuild succeeds, and repeated ingests coalesce into
    /// at most one follow-up rebuild.
    pub async fn schedule_rebuild(&self, db: DbHandle) {
        let build: RebuildTask = Arc::new(move || {
            let db = db.clone();
            Box::pin(async move { db::run_read_blocking(&db, build_generation).await })
        });
        self.schedule_rebuild_with(build).await;
    }

    async fn schedule_rebuild_with(&self, build: RebuildTask) {
        let mut state = self.rebuild.lock().await;
        if state.running {
            state.pending = true;
            return;
        }
        state.running = true;
        let store = self.clone();
        tokio::spawn(async move {
            store.rebuild_loop(build).await;
        });
    }

    async fn rebuild_loop(self, build: RebuildTask) {
        loop {
            match build().await {
                Ok(generation) => self.install(generation),
                Err(err) => {
                    tracing::error!(error = ?err, "read model rebuild failed");
                }
            }

            let mut state = self.rebuild.lock().await;
            if state.pending {
                state.pending = false;
                continue;
            }
            state.running = false;
            break;
        }
    }

    fn install(&self, generation: ReadGeneration) {
        let mut inner = self.inner.write();
        let previous = Arc::clone(&inner.active);
        inner.active = Arc::new(generation);
        inner.previous = Some(previous);
    }
}

/// One immutable read snapshot.
pub struct ReadGeneration {
    id: String,
    groups: Arc<Vec<Group>>,
    filter_universe: Arc<FilterUniverse>,
    groups_artifact: EncodedArtifact,
    chart_artifacts: HashMap<String, EncodedArtifact>,
    group_artifacts: HashMap<String, EncodedArtifact>,
    group_shards: HashMap<GroupShardKey, EncodedArtifact>,
    group_shard_counts: HashMap<String, usize>,
    chart_payloads: HashMap<String, Arc<ChartResponse>>,
}

impl ReadGeneration {
    /// Content-derived generation id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Structured groups for HTML rendering.
    pub fn groups(&self) -> Arc<Vec<Group>> {
        Arc::clone(&self.groups)
    }

    /// Structured filter universe for HTML rendering.
    pub fn filter_universe(&self) -> Arc<FilterUniverse> {
        Arc::clone(&self.filter_universe)
    }

    /// Materialized `/api/groups` body.
    pub fn groups_artifact(&self) -> &EncodedArtifact {
        &self.groups_artifact
    }

    /// Materialized latest-100 `/api/chart/{slug}` body.
    pub fn chart_artifact(&self, slug: &str) -> Option<&EncodedArtifact> {
        self.chart_artifacts.get(slug)
    }

    /// Materialized latest-100 `/api/group/{slug}` compatibility body.
    pub fn group_artifact(&self, slug: &str) -> Option<&EncodedArtifact> {
        self.group_artifacts.get(slug)
    }

    /// Materialized latest-100 shard body for landing/group hydration.
    pub fn group_shard_artifact(&self, slug: &str, index: usize) -> Option<&EncodedArtifact> {
        self.group_shards.get(&GroupShardKey {
            slug: slug.to_string(),
            index,
        })
    }

    /// Number of materialized shards for a group.
    pub fn group_shard_count(&self, slug: &str) -> usize {
        self.group_shard_counts.get(slug).copied().unwrap_or(0)
    }

    /// Structured latest-100 chart payload for single-chart HTML rendering.
    pub fn chart_payload(&self, slug: &str) -> Option<Arc<ChartResponse>> {
        self.chart_payloads.get(slug).map(Arc::clone)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct GroupShardKey {
    slug: String,
    index: usize,
}

#[derive(Serialize)]
struct GroupShardResponse {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<Summary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    window: u32,
    shard_index: usize,
    shard_count: usize,
    charts: Vec<NamedChartResponse>,
}

struct RawArtifact {
    key: String,
    kind: RawArtifactKind,
    bytes: Vec<u8>,
}

enum RawArtifactKind {
    Groups,
    Chart { slug: String },
    Group { slug: String },
    GroupShard { slug: String, index: usize },
}

fn build_generation(conn: &mut Connection) -> Result<ReadGeneration> {
    api::read_transaction(conn, build_generation_from_snapshot)
}

fn build_generation_from_snapshot(conn: &Connection) -> Result<ReadGeneration> {
    let groups = Arc::new(api::collect_groups(conn)?);
    let filter_universe = Arc::new(api::collect_filter_universe(conn)?);
    let window = CommitWindow::Last(
        NonZeroU32::new(api::DEFAULT_COMMIT_WINDOW).expect("default window is non-zero"),
    );

    let mut raw = Vec::new();
    raw.push(RawArtifact {
        key: "api:groups".to_string(),
        kind: RawArtifactKind::Groups,
        bytes: serde_json::to_vec(&GroupsResponse {
            groups: Arc::clone(&groups),
        })
        .context("serialize groups artifact")?,
    });

    let mut chart_payloads = HashMap::new();
    let mut group_shard_counts = HashMap::new();

    for group in groups.iter() {
        let mut charts = Vec::with_capacity(group.charts.len());
        for link in &group.charts {
            let chart = if let Some(chart) = chart_payloads.get(&link.slug) {
                Arc::clone(chart)
            } else {
                let key = ChartKey::from_slug(&link.slug)
                    .with_context(|| format!("invalid chart slug in group: {}", link.slug))?;
                let Some(chart) = api::chart_payload(conn, &key, &window)? else {
                    continue;
                };
                let chart = Arc::new(chart);
                raw.push(RawArtifact {
                    key: format!("api:chart:{}:100", link.slug),
                    kind: RawArtifactKind::Chart {
                        slug: link.slug.clone(),
                    },
                    bytes: serde_json::to_vec(chart.as_ref())
                        .with_context(|| format!("serialize chart artifact {}", link.slug))?,
                });
                chart_payloads.insert(link.slug.clone(), Arc::clone(&chart));
                chart
            };
            charts.push(NamedChartResponse {
                name: link.name.clone(),
                slug: link.slug.clone(),
                chart,
            });
        }

        if charts.is_empty() {
            group_shard_counts.insert(group.slug.clone(), 0);
            continue;
        }

        let group_response = GroupChartsResponse {
            name: group.name.clone(),
            summary: group.summary.clone(),
            description: group.description.clone(),
            charts: charts.clone(),
        };
        raw.push(RawArtifact {
            key: format!("api:group:{}:100", group.slug),
            kind: RawArtifactKind::Group {
                slug: group.slug.clone(),
            },
            bytes: serde_json::to_vec(&group_response)
                .with_context(|| format!("serialize group artifact {}", group.slug))?,
        });

        let shard_count = charts.len().div_ceil(GROUP_SHARD_CHARTS);
        group_shard_counts.insert(group.slug.clone(), shard_count);
        for (shard_index, chunk) in charts.chunks(GROUP_SHARD_CHARTS).enumerate() {
            let shard = GroupShardResponse {
                name: group.name.clone(),
                summary: group.summary.clone(),
                description: group.description.clone(),
                window: api::DEFAULT_COMMIT_WINDOW,
                shard_index,
                shard_count,
                charts: chunk.to_vec(),
            };
            raw.push(RawArtifact {
                key: format!("api:group-shard:{}:{shard_index}:100", group.slug),
                kind: RawArtifactKind::GroupShard {
                    slug: group.slug.clone(),
                    index: shard_index,
                },
                bytes: serde_json::to_vec(&shard).with_context(|| {
                    format!(
                        "serialize group shard artifact {}#{shard_index}",
                        group.slug
                    )
                })?,
            });
        }
    }

    let id = generation_id(&raw);
    let mut groups_artifact = None;
    let mut chart_artifacts = HashMap::new();
    let mut group_artifacts = HashMap::new();
    let mut group_shards = HashMap::new();

    for artifact in raw {
        let encoded = EncodedArtifact::new(&id, artifact.bytes)
            .with_context(|| format!("encode artifact {}", artifact.key))?;
        match artifact.kind {
            RawArtifactKind::Groups => groups_artifact = Some(encoded),
            RawArtifactKind::Chart { slug } => {
                chart_artifacts.insert(slug, encoded);
            }
            RawArtifactKind::Group { slug } => {
                group_artifacts.insert(slug, encoded);
            }
            RawArtifactKind::GroupShard { slug, index } => {
                group_shards.insert(GroupShardKey { slug, index }, encoded);
            }
        }
    }

    Ok(ReadGeneration {
        id,
        groups,
        filter_universe,
        groups_artifact: groups_artifact.context("groups artifact missing")?,
        chart_artifacts,
        group_artifacts,
        group_shards,
        group_shard_counts,
        chart_payloads,
    })
}

fn generation_id(raw: &[RawArtifact]) -> String {
    let mut sorted: Vec<_> = raw.iter().collect();
    sorted.sort_by(|a, b| a.key.cmp(&b.key));
    let mut hash = XxHash64::with_seed(0);
    for artifact in sorted {
        hash.write_u64(artifact.key.len() as u64);
        hash.write(artifact.key.as_bytes());
        hash.write_u64(artifact.bytes.len() as u64);
        hash.write(&artifact.bytes);
    }
    format!("{:016x}", hash.finish())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use axum::http::header;
    use tokio::sync::Notify;
    use tokio::time::sleep;

    use super::*;

    fn raw_artifact(key: &str, bytes: &[u8]) -> RawArtifact {
        RawArtifact {
            key: key.to_string(),
            kind: RawArtifactKind::Groups,
            bytes: bytes.to_vec(),
        }
    }

    fn empty_generation(id: &str) -> Result<ReadGeneration> {
        Ok(ReadGeneration {
            id: id.to_string(),
            groups: Arc::new(Vec::new()),
            filter_universe: Arc::new(FilterUniverse::default()),
            groups_artifact: EncodedArtifact::new(id, br#"{"groups":[]}"#.to_vec())?,
            chart_artifacts: HashMap::new(),
            group_artifacts: HashMap::new(),
            group_shards: HashMap::new(),
            group_shard_counts: HashMap::new(),
            chart_payloads: HashMap::new(),
        })
    }

    fn test_store(id: &str) -> Result<ReadStore> {
        Ok(ReadStore {
            inner: Arc::new(RwLock::new(ReadStoreInner {
                active: Arc::new(empty_generation(id)?),
                previous: None,
            })),
            rebuild: Arc::new(AsyncMutex::new(RebuildState::default())),
        })
    }

    async fn wait_for_rebuild_idle(store: &ReadStore) {
        for _ in 0..100 {
            if !store.rebuild.lock().await.running {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("read model rebuild did not become idle");
    }

    async fn wait_for_calls(calls: &AtomicUsize, expected: usize) {
        for _ in 0..100 {
            if calls.load(Ordering::SeqCst) >= expected {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("read model rebuild did not start");
    }

    #[test]
    fn generation_ids_are_content_derived_and_order_stable() {
        let a = vec![raw_artifact("b", b"two"), raw_artifact("a", b"one")];
        let b = vec![raw_artifact("a", b"one"), raw_artifact("b", b"two")];
        let c = vec![raw_artifact("a", b"one"), raw_artifact("b", b"changed")];

        assert_eq!(generation_id(&a), generation_id(&b));
        assert_ne!(generation_id(&a), generation_id(&c));
    }

    #[test]
    fn encoded_artifact_negotiates_precompressed_variants() -> Result<()> {
        let artifact = EncodedArtifact::new("abc123", br#"{"ok":true}"#.to_vec())?;
        let mut headers = HeaderMap::new();
        headers.insert(header::ACCEPT_ENCODING, HeaderValue::from_static("gzip"));

        let resp = artifact.response(&headers, ArtifactCachePolicy::Immutable);
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_ENCODING)
                .and_then(|v| v.to_str().ok()),
            Some("gzip")
        );
        assert_eq!(
            resp.headers()
                .get(header::VARY)
                .and_then(|v| v.to_str().ok()),
            Some("Accept-Encoding")
        );
        assert!(
            resp.headers()
                .get(header::CACHE_CONTROL)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.contains("immutable"))
        );
        Ok(())
    }

    #[test]
    fn encoded_artifact_returns_304_for_matching_etag() -> Result<()> {
        let artifact = EncodedArtifact::new("abc123", br#"{"ok":true}"#.to_vec())?;
        let mut headers = HeaderMap::new();
        headers.insert(
            header::IF_NONE_MATCH,
            HeaderValue::from_static("\"abc123\""),
        );

        let resp = artifact.response(&headers, ArtifactCachePolicy::Revalidate);
        assert_eq!(resp.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(
            resp.headers()
                .get(header::ETAG)
                .and_then(|v| v.to_str().ok()),
            Some("\"abc123\"")
        );
        Ok(())
    }

    #[tokio::test]
    async fn failed_rebuild_keeps_old_generation_active() -> Result<()> {
        let store = test_store("old")?;
        let build: RebuildTask = Arc::new(|| Box::pin(async { anyhow::bail!("boom") }));

        store.schedule_rebuild_with(build).await;
        wait_for_rebuild_idle(&store).await;

        assert_eq!(store.active().id(), "old");
        assert!(store.generation("old").is_some());
        Ok(())
    }

    #[tokio::test]
    async fn concurrent_rebuild_requests_coalesce() -> Result<()> {
        let store = test_store("old")?;
        let calls = Arc::new(AtomicUsize::new(0));
        let release_first = Arc::new(Notify::new());
        let build: RebuildTask = Arc::new({
            let calls = Arc::clone(&calls);
            let release_first = Arc::clone(&release_first);
            move || {
                let calls = Arc::clone(&calls);
                let release_first = Arc::clone(&release_first);
                Box::pin(async move {
                    let call = calls.fetch_add(1, Ordering::SeqCst) + 1;
                    if call == 1 {
                        release_first.notified().await;
                    }
                    empty_generation(&format!("gen{call}"))
                })
            }
        });

        store.schedule_rebuild_with(Arc::clone(&build)).await;
        wait_for_calls(&calls, 1).await;
        store.schedule_rebuild_with(Arc::clone(&build)).await;
        store.schedule_rebuild_with(build).await;
        release_first.notify_one();
        wait_for_rebuild_idle(&store).await;

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(store.active().id(), "gen2");
        assert!(store.generation("gen1").is_some());
        assert!(store.generation("old").is_none());
        Ok(())
    }
}
