// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for intern pool implementations.
//!
//! Compares different data structures and synchronization strategies for
//! mapping `&str -> u64` with ~200 interned strings.
//!
//! ## What is measured
//!
//! - **algo**: Single-key lookup latency across hash algorithms (no sync overhead).
//! - **sync**: Single-key lookup with different concurrency wrappers at varying thread counts.
//! - **zipf**: Throughput of 1M lookups with Zipfian key distribution (realistic hot/cold access).

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
// We intentionally benchmark std HashMap (SipHash) against alternatives.
#![allow(clippy::disallowed_types)]

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::OnceLock;

use arc_swap::ArcSwap;
use arc_swap::cache::Cache as ArcSwapCache;
use dashmap::DashMap;
use divan::Bencher;
use divan::counter::ItemsCount;
use intern_pool_bench::CompactPool;
use intern_pool_bench::StringId;
use parking_lot::RwLock;
use rand::prelude::*;
use rand_distr::Zipf;
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashMap;

fn main() {
    divan::main();
}

// ─── Constants ───────────────────────────────────────────────────────────────

const NUM_STRINGS: usize = 200;
const ZIPF_NUM_LOOKUPS: usize = 1_000_000;
const ZIPF_EXPONENT: f64 = 1.07;

// ─── Type aliases ────────────────────────────────────────────────────────────

type FoldhashMap<K, V> = hashbrown::HashMap<K, V>;
type AhashMap<K, V> = hashbrown::HashMap<K, V, ahash::RandomState>;
type PoolCache =
    ArcSwapCache<&'static ArcSwap<FxHashMap<&'static str, u64>>, Arc<FxHashMap<&'static str, u64>>>;

/// Wrapper for `UnsafeCell` that is `Sync` for use in `thread_local!` + `bench()`.
///
/// SAFETY: Only used inside `thread_local!` where access is guaranteed single-threaded.
struct SyncCache(UnsafeCell<Option<PoolCache>>);
// SAFETY: This is only accessed inside thread_local!, which guarantees single-threaded access.
unsafe impl Sync for SyncCache {}

impl SyncCache {
    const fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    fn with_cache<R>(&self, f: impl FnOnce(&mut PoolCache) -> R) -> R {
        // SAFETY: thread_local access is single-threaded by definition.
        let cache = unsafe { &mut *self.0.get() };
        let cache = cache.get_or_insert_with(|| ArcSwapCache::new(arcswap_pool()));
        f(cache)
    }
}

// ─── Test Data ───────────────────────────────────────────────────────────────

fn test_strings() -> &'static [&'static str] {
    static STRINGS: OnceLock<Vec<&'static str>> = OnceLock::new();
    STRINGS.get_or_init(|| {
        (0..NUM_STRINGS)
            .map(|i| &*Box::leak(format!("vortex.enc.{i}").into_boxed_str()))
            .collect()
    })
}

fn zipf_keys() -> &'static [&'static str] {
    static KEYS: OnceLock<Vec<&'static str>> = OnceLock::new();
    KEYS.get_or_init(|| {
        let strings = test_strings();
        let mut rng = StdRng::seed_from_u64(42);
        let zipf = Zipf::new(NUM_STRINGS as f64, ZIPF_EXPONENT).unwrap();
        (0..ZIPF_NUM_LOOKUPS)
            .map(|_| {
                let idx: f64 = rng.sample(zipf);
                strings[(idx as usize).saturating_sub(1).min(NUM_STRINGS - 1)]
            })
            .collect()
    })
}

// ─── Pool Constructors ───────────────────────────────────────────────────────

fn std_hashmap() -> &'static HashMap<&'static str, u64> {
    static MAP: OnceLock<HashMap<&'static str, u64>> = OnceLock::new();
    MAP.get_or_init(|| {
        test_strings()
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u64))
            .collect()
    })
}

fn fx_hashmap() -> &'static FxHashMap<&'static str, u64> {
    static MAP: OnceLock<FxHashMap<&'static str, u64>> = OnceLock::new();
    MAP.get_or_init(|| {
        test_strings()
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u64))
            .collect()
    })
}

fn foldhash_hashmap() -> &'static FoldhashMap<&'static str, u64> {
    static MAP: OnceLock<FoldhashMap<&'static str, u64>> = OnceLock::new();
    MAP.get_or_init(|| {
        test_strings()
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u64))
            .collect()
    })
}

fn ahash_hashmap() -> &'static AhashMap<&'static str, u64> {
    static MAP: OnceLock<AhashMap<&'static str, u64>> = OnceLock::new();
    MAP.get_or_init(|| {
        test_strings()
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u64))
            .collect()
    })
}

fn sorted_table() -> &'static [(&'static str, u64)] {
    static TABLE: OnceLock<Vec<(&'static str, u64)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut v: Vec<_> = test_strings()
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u64))
            .collect();
        v.sort_by_key(|(s, _)| *s);
        v
    })
}

fn arcswap_pool() -> &'static ArcSwap<FxHashMap<&'static str, u64>> {
    static POOL: OnceLock<ArcSwap<FxHashMap<&'static str, u64>>> = OnceLock::new();
    POOL.get_or_init(|| {
        let map: FxHashMap<_, _> = test_strings()
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u64))
            .collect();
        ArcSwap::new(Arc::new(map))
    })
}

fn rwlock_pool() -> &'static RwLock<FxHashMap<&'static str, u64>> {
    static POOL: OnceLock<RwLock<FxHashMap<&'static str, u64>>> = OnceLock::new();
    POOL.get_or_init(|| {
        let map: FxHashMap<_, _> = test_strings()
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i as u64))
            .collect();
        RwLock::new(map)
    })
}

fn dashmap_pool() -> &'static DashMap<&'static str, u64, FxBuildHasher> {
    static POOL: OnceLock<DashMap<&'static str, u64, FxBuildHasher>> = OnceLock::new();
    POOL.get_or_init(|| {
        let map = DashMap::with_hasher(FxBuildHasher);
        for (i, s) in test_strings().iter().enumerate() {
            map.insert(*s, i as u64);
        }
        map
    })
}

fn compact_pool() -> &'static CompactPool {
    static POOL: OnceLock<CompactPool> = OnceLock::new();
    POOL.get_or_init(|| {
        CompactPool::new(
            test_strings()
                .iter()
                .enumerate()
                .map(|(i, s)| (*s, i as u64)),
        )
    })
}

/// Pre-computed `StringId`s for the Zipfian key set.
/// Simulates the real use case: hash each encoding name once at init, then resolve by ID.
fn zipf_ids() -> &'static [StringId] {
    static IDS: OnceLock<Vec<StringId>> = OnceLock::new();
    IDS.get_or_init(|| {
        let pool = compact_pool();
        zipf_keys().iter().map(|k| pool.id(k)).collect()
    })
}

// ─── Bench A: Algorithm Comparison (single-threaded, no sync overhead) ───────

mod algo {
    use super::*;

    #[divan::bench]
    fn std_hashmap_siphash(bencher: Bencher) {
        let map = std_hashmap();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| black_box(map.get(key)));
    }

    #[divan::bench]
    fn fx_hashmap(bencher: Bencher) {
        let map = super::fx_hashmap();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| black_box(map.get(key)));
    }

    #[divan::bench]
    fn foldhash_hashmap(bencher: Bencher) {
        let map = super::foldhash_hashmap();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| black_box(map.get(key)));
    }

    #[divan::bench]
    fn ahash_hashmap(bencher: Bencher) {
        let map = super::ahash_hashmap();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| black_box(map.get(key)));
    }

    /// CompactPool: hash-only comparison, no `bcmp` key compare call.
    #[divan::bench]
    fn compact_pool(bencher: Bencher) {
        let pool = super::compact_pool();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| black_box(pool.get(key)));
    }

    /// CompactPool with compile-time StringId: zero hashing at runtime.
    #[divan::bench]
    fn compact_pool_prehashed(bencher: Bencher) {
        let pool = super::compact_pool();
        // Hash computed at compile time — this is a constant in the binary.
        const ID: StringId = StringId::of("vortex.enc.100");
        bencher.bench(|| black_box(pool.resolve(ID)));
    }

    #[divan::bench]
    fn sorted_binary_search(bencher: Bencher) {
        let table = sorted_table();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| {
            black_box(
                table
                    .binary_search_by_key(&key, |(k, _)| *k)
                    .map(|i| table[i].1),
            )
        });
    }
}

// ─── Bench B: Sync Wrapper Overhead (multi-threaded, single-key lookup) ──────

mod sync_wrapper {
    use super::*;

    #[divan::bench(threads = [1, 2, 4])]
    fn static_ref(bencher: Bencher) {
        let map = fx_hashmap();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| black_box(map.get(key)));
    }

    #[divan::bench(threads = [1, 2, 4])]
    fn arcswap(bencher: Bencher) {
        let pool = arcswap_pool();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| {
            let guard = pool.load();
            black_box(guard.get(key).copied())
        });
    }

    /// ArcSwap with per-thread `Cache`: avoids full atomic load on each read.
    /// Only checks if pointer changed (cheap Relaxed load), reuses cached Arc otherwise.
    #[divan::bench(threads = [1, 2, 4])]
    fn arcswap_cached(bencher: Bencher) {
        let key = test_strings()[NUM_STRINGS / 2];
        thread_local! {
            static CACHE: SyncCache = const { SyncCache::new() };
        }
        bencher.bench(|| {
            CACHE.with(|sc| {
                sc.with_cache(|cache| {
                    let map = cache.load();
                    black_box(map.get(key).copied())
                })
            })
        });
    }

    #[divan::bench(threads = [1, 2, 4])]
    fn rwlock(bencher: Bencher) {
        let pool = rwlock_pool();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| {
            let guard = pool.read();
            black_box(guard.get(key).copied())
        });
    }

    #[divan::bench(threads = [1, 2, 4])]
    fn dashmap(bencher: Bencher) {
        let map = dashmap_pool();
        let key = test_strings()[NUM_STRINGS / 2];
        bencher.bench(|| black_box(map.get(key).map(|v| *v)));
    }
}

// ─── Bench C: Zipfian Throughput (1M lookups, reports lookups/sec) ───────────

mod zipf_throughput {
    use super::*;

    // ─── Sync wrapper comparison (all use FxHash) ─────────────────────────

    #[divan::bench(threads = [1, 2, 4])]
    fn static_ref(bencher: Bencher) {
        let map = fx_hashmap();
        let keys = zipf_keys();
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            for key in keys {
                black_box(map.get(key));
            }
        });
    }

    #[divan::bench(threads = [1, 2, 4])]
    fn arcswap(bencher: Bencher) {
        let pool = arcswap_pool();
        let keys = zipf_keys();
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            let guard = pool.load();
            for key in keys {
                black_box(guard.get(key));
            }
        });
    }

    /// ArcSwap with per-thread Cache — avoids re-loading the Arc each iteration.
    #[divan::bench(threads = [1, 2, 4])]
    fn arcswap_cached(bencher: Bencher) {
        let keys = zipf_keys();
        thread_local! {
            static CACHE: SyncCache = const { SyncCache::new() };
        }
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            CACHE.with(|sc| {
                sc.with_cache(|cache| {
                    let map = cache.load();
                    for key in keys {
                        black_box(map.get(key));
                    }
                })
            })
        });
    }

    #[divan::bench(threads = [1, 2, 4])]
    fn rwlock(bencher: Bencher) {
        let pool = rwlock_pool();
        let keys = zipf_keys();
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            let guard = pool.read();
            for key in keys {
                black_box(guard.get(key));
            }
        });
    }

    #[divan::bench(threads = [1, 2, 4])]
    fn dashmap(bencher: Bencher) {
        let map = dashmap_pool();
        let keys = zipf_keys();
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            for key in keys {
                black_box(map.get(key).map(|v| *v));
            }
        });
    }

    // ─── Hash algorithm comparison (all use OnceLock / no sync) ──────────

    /// CompactPool: hash-only lookup (no key comparison). Uses foldhash internally.
    #[divan::bench(threads = [1, 2, 4])]
    fn compact_pool(bencher: Bencher) {
        let pool = super::compact_pool();
        let keys = zipf_keys();
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            for key in keys {
                black_box(pool.get(key));
            }
        });
    }

    /// CompactPool with pre-computed StringIds: zero hashing, zero key comparison.
    /// This is the theoretical speed-of-light for this workload.
    #[divan::bench(threads = [1, 2, 4])]
    fn compact_pool_prehashed(bencher: Bencher) {
        let pool = super::compact_pool();
        let ids = zipf_ids();
        bencher.counter(ItemsCount::new(ids.len())).bench(|| {
            for &id in ids {
                black_box(pool.resolve(id));
            }
        });
    }

    #[divan::bench(threads = [1, 2, 4])]
    fn foldhash(bencher: Bencher) {
        let map = foldhash_hashmap();
        let keys = zipf_keys();
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            for key in keys {
                black_box(map.get(key));
            }
        });
    }

    #[divan::bench(threads = [1, 2, 4])]
    fn ahash(bencher: Bencher) {
        let map = ahash_hashmap();
        let keys = zipf_keys();
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            for key in keys {
                black_box(map.get(key));
            }
        });
    }
}
