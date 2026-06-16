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
//! The cache is a generation-keyed, single-flight store of [`Arc`]-wrapped
//! payloads, one [`DashMap`] per result type:
//! - reads check the slot, clone the [`Arc`] out, and return — no DuckDB
//!   round-trip on the hot path;
//! - the first miss for a key runs `compute` while concurrent waiters share the
//!   same async slot;
//! - [`QueryCache::invalidate`] is called from [`crate::ingest`] after a
//!   successful commit; it advances the generation and clears the visible
//!   maps so old in-flight computes cannot repopulate the current snapshot.
//!
//! The two unkeyed slots — `/api/groups` and the filter universe — use a
//! [`DashMap`] with `()` as the logical key, so every slot in the cache is the
//! same primitive.
//!
//! Cached values are wrapped in [`std::sync::Arc`] and never deep-cloned on
//! the cache-hit path. The JSON bytes are still serialized per fallback
//! response; the materialized latest-100 hot path lives in
//! [`crate::read_model`].

use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use dashmap::DashMap;
use tokio::sync::Mutex as AsyncMutex;

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
    /// Monotonically advances whenever ingest invalidates the read snapshot.
    generation: AtomicU64,
    /// `/api/groups` discovery result. Keyed by `()` because there is only
    /// ever one group list per snapshot.
    groups: DashMap<VersionedKey<()>, CacheSlot<Arc<Vec<Group>>>>,
    /// Global filter universe (engines + formats). Also unkeyed.
    filter_universe: DashMap<VersionedKey<()>, CacheSlot<Arc<FilterUniverse>>>,
    /// Per-chart payloads, keyed by `(slug, window)`.
    chart_payloads: DashMap<VersionedKey<ChartCacheKey>, CacheSlot<Option<Arc<ChartResponse>>>>,
    /// Per-group payloads, keyed by `(slug, window)`.
    group_charts: DashMap<VersionedKey<GroupCacheKey>, CacheSlot<Option<Arc<GroupChartsResponse>>>>,
}

type CacheSlot<T> = Arc<AsyncMutex<Option<T>>>;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct VersionedKey<K> {
    generation: u64,
    key: K,
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

    async fn get_or_compute<K, V, Raw, F, Fut, Wrap>(
        &self,
        map: &DashMap<VersionedKey<K>, CacheSlot<V>>,
        key: K,
        compute: F,
        wrap: Wrap,
    ) -> Result<V>
    where
        K: Clone + Eq + Hash,
        V: Clone,
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Raw>>,
        Wrap: FnOnce(Raw) -> V,
    {
        let generation = self.generation.load(Ordering::Acquire);
        let cache_key = VersionedKey { generation, key };
        let slot = map
            .entry(cache_key.clone())
            .or_insert_with(|| Arc::new(AsyncMutex::new(None)))
            .clone();

        let mut guard = slot.lock().await;
        if let Some(value) = guard.as_ref() {
            return Ok(value.clone());
        }

        let fresh = wrap(compute().await?);
        if self.generation.load(Ordering::Acquire) == generation {
            *guard = Some(fresh.clone());
        } else {
            map.remove(&cache_key);
        }
        Ok(fresh)
    }

    /// Get the cached `Arc<Vec<Group>>` from `/api/groups`, or run `compute`
    /// if the slot is empty and store the result.
    pub async fn groups<F, Fut>(&self, compute: F) -> Result<Arc<Vec<Group>>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Vec<Group>>>,
    {
        self.get_or_compute(&self.groups, (), compute, Arc::new)
            .await
    }

    /// Get the cached `Arc<FilterUniverse>` for the global filter bar, or
    /// run `compute` if the slot is empty and store the result.
    pub async fn filter_universe<F, Fut>(&self, compute: F) -> Result<Arc<FilterUniverse>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<FilterUniverse>>,
    {
        self.get_or_compute(&self.filter_universe, (), compute, Arc::new)
            .await
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
        self.get_or_compute(&self.chart_payloads, key, compute, |value| {
            value.map(Arc::new)
        })
        .await
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
        self.get_or_compute(&self.group_charts, key, compute, |value| {
            value.map(Arc::new)
        })
        .await
    }

    /// Drop every cached value. Called from the ingest handler after a
    /// successful commit so the next read sees the fresh snapshot.
    pub fn invalidate(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
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
    use std::time::Duration;

    use anyhow::anyhow;
    use tokio::sync::oneshot;

    use super::*;
    use crate::api::FilterUniverse;

    fn empty_universe() -> FilterUniverse {
        FilterUniverse::default()
    }

    fn universe_with_engine(engine: &str) -> FilterUniverse {
        FilterUniverse {
            engines: vec![engine.to_string()],
            formats: Vec::new(),
        }
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
    async fn concurrent_singleton_misses_share_one_compute() -> Result<()> {
        let cache = Arc::new(QueryCache::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();

        for _ in 0..16 {
            let cache = Arc::clone(&cache);
            let calls = Arc::clone(&calls);
            tasks.push(tokio::spawn(async move {
                cache
                    .filter_universe(|| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        async {
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            Ok(empty_universe())
                        }
                    })
                    .await
            }));
        }

        for task in tasks {
            task.await??;
        }

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "concurrent misses for one key should collapse to one compute"
        );
        Ok(())
    }

    #[tokio::test]
    async fn stale_in_flight_compute_does_not_repopulate_after_invalidate() -> Result<()> {
        let cache = Arc::new(QueryCache::new());
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();

        let stale_cache = Arc::clone(&cache);
        let stale_task = tokio::spawn(async move {
            stale_cache
                .filter_universe(|| {
                    started_tx.send(()).expect("test receiver is alive");
                    async {
                        release_rx
                            .await
                            .expect("test sender releases stale compute");
                        Ok(universe_with_engine("stale"))
                    }
                })
                .await
        });

        started_rx.await?;
        cache.invalidate();

        let fresh = cache
            .filter_universe(|| async { Ok(universe_with_engine("fresh")) })
            .await?;
        assert_eq!(fresh.engines, ["fresh"]);

        release_tx.send(()).expect("stale compute is waiting");
        let stale = stale_task.await??;
        assert_eq!(stale.engines, ["stale"]);

        let cached = cache
            .filter_universe(|| async { Ok(universe_with_engine("unexpected")) })
            .await?;
        assert_eq!(
            cached.engines,
            ["fresh"],
            "old in-flight computations must not populate the new generation"
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
                history: crate::api::ChartHistory {
                    total_commits: 0,
                    start_index: 0,
                    loaded_commits: 0,
                    complete: true,
                },
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
