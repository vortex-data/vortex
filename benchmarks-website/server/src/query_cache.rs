// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-memory query result cache for the read API.
//!
//! Every chart payload, group discovery result, and filter universe in the v3
//! site is a deterministic function of the DuckDB snapshot — and that
//! snapshot only changes when `/api/ingest` lands a new envelope (~30 times a
//! day). Without a cache, every concurrent request re-runs the same SQL
//! against the engine, which serialises on DuckDB's internal locks; many tabs
//! / clients open at once peg the read API behind that lock even though the
//! underlying answer hasn't changed in hours.
//!
//! The cache is a cache-aside store of [`Arc`]-wrapped payloads, one
//! [`DashMap`] per result type:
//! - reads check the slot, clone the [`Arc`] out, and return — no DuckDB
//!   round-trip on the hot path;
//! - misses run `compute`, wrap the result in [`Arc`], and stash it for the
//!   next reader;
//! - [`QueryCache::invalidate`] is called from [`crate::ingest`] after a
//!   successful commit so the next read recomputes against the fresh
//!   snapshot.
//!
//! The two unkeyed slots — `/api/groups` and the filter universe — use a
//! [`DashMap`] with `()` as the key, so every slot in the cache is the same
//! primitive. Concurrent misses for the same slot each run `compute` and the
//! last writer wins. Both writers produce identical data because the DB is
//! read-only between invalidations, so the only cost is the redundant work
//! in the brief window between an invalidate and the first repopulation.
//!
//! Cached values are wrapped in [`std::sync::Arc`] and never cloned on the
//! cache-hit path; [`axum::Json`] serialises through the [`Arc`] so the JSON
//! bytes on the wire are produced once per cached value.

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;

use crate::api::ChartResponse;
use crate::api::CommitWindow;
use crate::api::FilterUniverse;
use crate::api::Group;
use crate::api::GroupChartsResponse;

/// Composite cache for every read-side DuckDB query.
///
/// Cheap to clone via [`Arc`]; one instance is owned by [`crate::app::AppState`]
/// for the lifetime of the server. All entries are cleared by
/// [`Self::invalidate`] when ingest changes the underlying snapshot.
#[derive(Default)]
pub struct QueryCache {
    /// `/api/groups` discovery result. Keyed by `()` because there is only
    /// ever one group list per snapshot.
    groups: DashMap<(), Arc<Vec<Group>>>,
    /// Global filter universe (engines + formats). Also unkeyed.
    filter_universe: DashMap<(), Arc<FilterUniverse>>,
    /// Per-chart payloads, keyed by `(slug, window)`.
    chart_payloads: DashMap<ChartCacheKey, Option<Arc<ChartResponse>>>,
    /// Per-group payloads, keyed by `(slug, window)`.
    group_charts: DashMap<GroupCacheKey, Option<Arc<GroupChartsResponse>>>,
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
    /// if the slot is empty and store the result.
    pub async fn groups<F, Fut>(&self, compute: F) -> Result<Arc<Vec<Group>>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Vec<Group>>>,
    {
        if let Some(entry) = self.groups.get(&()) {
            return Ok(entry.value().clone());
        }
        let fresh = Arc::new(compute().await?);
        self.groups.insert((), Arc::clone(&fresh));
        Ok(fresh)
    }

    /// Get the cached `Arc<FilterUniverse>` for the global filter bar, or
    /// run `compute` if the slot is empty and store the result.
    pub async fn filter_universe<F, Fut>(&self, compute: F) -> Result<Arc<FilterUniverse>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<FilterUniverse>>,
    {
        if let Some(entry) = self.filter_universe.get(&()) {
            return Ok(entry.value().clone());
        }
        let fresh = Arc::new(compute().await?);
        self.filter_universe.insert((), Arc::clone(&fresh));
        Ok(fresh)
    }

    /// Get the cached chart payload for `(slug, window)`, or run `compute`
    /// if the entry is absent and store the result. The cached value is
    /// `Option<Arc<ChartResponse>>` so a confirmed "no data for this slug"
    /// answer is cached too — repeated 404s do not re-hit DuckDB.
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
        if let Some(entry) = self.chart_payloads.get(&key) {
            return Ok(entry.value().clone());
        }
        let arc_opt = compute().await?.map(Arc::new);
        self.chart_payloads.insert(key, arc_opt.clone());
        Ok(arc_opt)
    }

    /// Get the cached per-group payload for `(slug, window)`, or run
    /// `compute` if the entry is absent and store the result.
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
        if let Some(entry) = self.group_charts.get(&key) {
            return Ok(entry.value().clone());
        }
        let arc_opt = compute().await?.map(Arc::new);
        self.group_charts.insert(key, arc_opt.clone());
        Ok(arc_opt)
    }

    /// Drop every cached value. Called from the ingest handler after a
    /// successful commit so the next read sees the fresh snapshot.
    pub fn invalidate(&self) {
        self.groups.clear();
        self.filter_universe.clear();
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
    async fn chart_payload_caches_negative_result() -> Result<()> {
        let cache = QueryCache::new();
        let calls = AtomicUsize::new(0);

        let none1 = cache
            .chart_payload("missing", &CommitWindow::All, || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(None) }
            })
            .await?;
        let none2 = cache
            .chart_payload("missing", &CommitWindow::All, || {
                calls.fetch_add(1, Ordering::SeqCst);
                async { Ok(None) }
            })
            .await?;

        assert!(none1.is_none() && none2.is_none());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "the second read for a missing slug should hit the cache, not re-query"
        );
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
