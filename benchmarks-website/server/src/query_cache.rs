// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-memory query result cache for the read API.
//!
//! Every chart payload, group discovery result, and filter universe in the v3
//! site is derived from a fixed DuckDB snapshot — the database only changes
//! when `/api/ingest` lands a new envelope (a few dozen times a day). Without
//! a cache every concurrent request re-runs the same SQL on a freshly cloned
//! DuckDB connection, which serialises on the engine's internal locks; with
//! many tabs / clients open at once the read API ends up pegged behind the
//! shared engine even though the underlying answer hasn't changed in hours.
//!
//! [`QueryCache`] sits between the HTTP handlers and the DB layer:
//! - reads check the cache first and return the cached `Arc<T>` directly when
//!   present, never touching DuckDB on the hot path;
//! - misses are coalesced via [`tokio::sync::OnceCell`] so a thundering herd
//!   of concurrent cold-start requests for the same key triggers a single
//!   `spawn_blocking` round-trip — every other waiter receives the same
//!   `Arc<T>` once that single computation finishes;
//! - [`QueryCache::invalidate`] is called from the ingest handler after a
//!   successful commit so the next read recomputes from the fresh snapshot.
//!
//! Cached values are wrapped in [`std::sync::Arc`] and never cloned on the
//! cache-hit path; [`axum::Json`] serialises through the `Arc` so the bytes
//! that go on the wire are produced once per unique value, then served to
//! every concurrent reader.

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;
use parking_lot::Mutex;
use tokio::sync::OnceCell;

use crate::api::ChartResponse;
use crate::api::CommitWindow;
use crate::api::FilterUniverse;
use crate::api::Group;
use crate::api::GroupChartsResponse;

/// One slot of the cache: an [`Arc`]-wrapped [`OnceCell`] so concurrent
/// initialisers all observe the same single-flight result.
type Cell<T> = Arc<OnceCell<T>>;

fn new_cell<T>() -> Cell<T> {
    Arc::new(OnceCell::new())
}

/// Composite cache for every read-side DuckDB query.
///
/// Cheap to clone via [`Arc`]; one instance is owned by [`crate::app::AppState`]
/// for the lifetime of the server. All entries are cleared by
/// [`Self::invalidate`] when ingest changes the underlying snapshot.
#[derive(Default)]
pub struct QueryCache {
    groups: Mutex<Cell<Arc<Vec<Group>>>>,
    filter_universe: Mutex<Cell<Arc<FilterUniverse>>>,
    chart_payloads: DashMap<ChartCacheKey, Cell<Option<Arc<ChartResponse>>>>,
    group_charts: DashMap<GroupCacheKey, Cell<Option<Arc<GroupChartsResponse>>>>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct ChartCacheKey {
    slug: String,
    window: CommitWindow,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct GroupCacheKey {
    slug: String,
    window: CommitWindow,
}

impl QueryCache {
    /// Build an empty cache. Equivalent to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the cached `Arc<Vec<Group>>` from `/api/groups`, or run `compute`
    /// once if the slot is empty.
    pub async fn groups<F, Fut>(&self, compute: F) -> Result<Arc<Vec<Group>>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Vec<Group>>>,
    {
        let cell = self.groups.lock().clone();
        let value = cell
            .get_or_try_init(|| async { compute().await.map(Arc::new) })
            .await?;
        Ok(Arc::clone(value))
    }

    /// Get the cached `Arc<FilterUniverse>` for the global filter bar, or
    /// run `compute` once if the slot is empty.
    pub async fn filter_universe<F, Fut>(&self, compute: F) -> Result<Arc<FilterUniverse>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<FilterUniverse>>,
    {
        let cell = self.filter_universe.lock().clone();
        let value = cell
            .get_or_try_init(|| async { compute().await.map(Arc::new) })
            .await?;
        Ok(Arc::clone(value))
    }

    /// Get the cached chart payload for `(slug, window)`, or run `compute`
    /// once if the entry is empty. Returns `None` when the chart has no data
    /// and the caller should respond with 404.
    pub async fn chart_payload<F, Fut>(
        &self,
        slug: &str,
        window: &CommitWindow,
        compute: F,
    ) -> Result<Option<Arc<ChartResponse>>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Option<ChartResponse>>>,
    {
        let key = ChartCacheKey {
            slug: slug.to_string(),
            window: *window,
        };
        let cell = self
            .chart_payloads
            .entry(key)
            .or_insert_with(new_cell)
            .clone();
        let value = cell
            .get_or_try_init(|| async { compute().await.map(|opt| opt.map(Arc::new)) })
            .await?;
        Ok(value.clone())
    }

    /// Get the cached per-group payload for `(slug, window)`, or run
    /// `compute` once if the entry is empty.
    pub async fn group_charts<F, Fut>(
        &self,
        slug: &str,
        window: &CommitWindow,
        compute: F,
    ) -> Result<Option<Arc<GroupChartsResponse>>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Option<GroupChartsResponse>>>,
    {
        let key = GroupCacheKey {
            slug: slug.to_string(),
            window: *window,
        };
        let cell = self
            .group_charts
            .entry(key)
            .or_insert_with(new_cell)
            .clone();
        let value = cell
            .get_or_try_init(|| async { compute().await.map(|opt| opt.map(Arc::new)) })
            .await?;
        Ok(value.clone())
    }

    /// Drop every cached value. Called from the ingest handler after a
    /// successful commit so the next read sees the fresh snapshot.
    ///
    /// In-flight initialisers continue to completion against their own
    /// (now-detached) [`OnceCell`]; their results are returned to the awaiter
    /// that triggered them. Subsequent calls observe the freshly-cleared slot
    /// and trigger a new `compute`. This is the standard "stale read for
    /// in-flight, fresh read for new requests" trade-off.
    pub fn invalidate(&self) {
        *self.groups.lock() = new_cell();
        *self.filter_universe.lock() = new_cell();
        self.chart_payloads.clear();
        self.group_charts.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use anyhow::anyhow;

    use super::*;
    use crate::api::FilterUniverse;

    fn empty_universe() -> FilterUniverse {
        FilterUniverse::default()
    }

    #[tokio::test]
    async fn singleton_caches_and_returns_same_arc() -> Result<()> {
        let cache = QueryCache::new();
        let calls = AtomicUsize::new(0);

        let a = cache
            .filter_universe(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(empty_universe()) }
            })
            .await?;
        let b = cache
            .filter_universe(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(empty_universe()) }
            })
            .await?;

        assert_eq!(calls.load(Ordering::SeqCst), 1, "compute should run once");
        assert!(Arc::ptr_eq(&a, &b), "cache returns the same Arc");
        Ok(())
    }

    #[tokio::test]
    async fn invalidate_clears_singleton() -> Result<()> {
        let cache = QueryCache::new();
        let calls = AtomicUsize::new(0);

        let a = cache
            .filter_universe(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(empty_universe()) }
            })
            .await?;
        cache.invalidate();
        let b = cache
            .filter_universe(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(empty_universe()) }
            })
            .await?;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "invalidate should force a recompute"
        );
        assert!(
            !Arc::ptr_eq(&a, &b),
            "post-invalidate read should produce a fresh Arc"
        );
        Ok(())
    }

    #[tokio::test]
    async fn chart_payload_keyed_by_slug_and_window() -> Result<()> {
        let cache = QueryCache::new();
        let calls = AtomicUsize::new(0);

        let make = |display_name: &str| {
            let display_name = display_name.to_string();
            ChartResponse {
                display_name,
                unit_kind: crate::api::UnitKind::TimeNs,
                commits: Vec::new(),
                series: serde_json::Map::new(),
                series_meta: std::collections::BTreeMap::new(),
            }
        };

        let one = cache
            .chart_payload("a", &CommitWindow::All, || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(Some(make("first"))) }
            })
            .await?
            .expect("Some");
        let _two = cache
            .chart_payload("a", &CommitWindow::All, || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(Some(make("second"))) }
            })
            .await?
            .expect("Some");
        // Different window — should be a separate cache slot.
        let three = cache
            .chart_payload(
                "a",
                &CommitWindow::Last(std::num::NonZeroU32::new(10).unwrap()),
                || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    async { Ok(Some(make("third"))) }
                },
            )
            .await?
            .expect("Some");

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(one.display_name, "first");
        assert_eq!(three.display_name, "third");
        Ok(())
    }

    #[tokio::test]
    async fn errors_do_not_populate_cache() -> Result<()> {
        let cache = QueryCache::new();
        let calls = AtomicUsize::new(0);

        let res = cache
            .filter_universe(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Err::<FilterUniverse, _>(anyhow!("boom")) }
            })
            .await;
        assert!(res.is_err(), "error path bubbles up");

        cache
            .filter_universe(|| {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(empty_universe()) }
            })
            .await?;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "second call must rerun after an errored first call",
        );
        Ok(())
    }
}
