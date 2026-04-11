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
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use arc_swap::ArcSwap;
use arc_swap::cache::Cache as ArcSwapCache;
use dashmap::DashMap;
use divan::Bencher;
use divan::counter::ItemsCount;
use intern_pool_bench::CompactPool;
use intern_pool_bench::EncodingId;
use intern_pool_bench::RegistryBuilder;
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
fn zipf_ids() -> &'static [StringId] {
    static IDS: OnceLock<Vec<StringId>> = OnceLock::new();
    IDS.get_or_init(|| {
        let pool = compact_pool();
        zipf_keys().iter().map(|k| pool.id(k)).collect()
    })
}

/// 1M Zipfian indices (0..199) for the pre-resolved ID benchmarks.
fn zipf_indices() -> &'static [usize] {
    static INDICES: OnceLock<Vec<usize>> = OnceLock::new();
    INDICES.get_or_init(|| {
        let mut rng = StdRng::seed_from_u64(42);
        let zipf = Zipf::new(NUM_STRINGS as f64, ZIPF_EXPONENT).unwrap();
        (0..ZIPF_NUM_LOOKUPS)
            .map(|_| {
                let idx: f64 = rng.sample(zipf);
                (idx as usize).saturating_sub(1).min(NUM_STRINGS - 1)
            })
            .collect()
    })
}

// ─── Pre-resolved ID storage mechanisms ──────────────────────────────────────

/// Approach 1: Eager Vec<u64> — resolve all at init, read by index.
fn eager_vec() -> &'static [u64] {
    static VEC: OnceLock<Vec<u64>> = OnceLock::new();
    VEC.get_or_init(|| {
        let pool = compact_pool();
        test_strings()
            .iter()
            .map(|s| pool.get(s).unwrap())
            .collect()
    })
}

/// Approach 2: AtomicU64 array — resolve once, read with Relaxed load.
fn atomic_array() -> &'static [AtomicU64] {
    static ARRAY: OnceLock<Vec<AtomicU64>> = OnceLock::new();
    ARRAY.get_or_init(|| {
        let pool = compact_pool();
        test_strings()
            .iter()
            .map(|s| AtomicU64::new(pool.get(s).unwrap()))
            .collect()
    })
}

/// Approach 3: OnceLock<u64> array — lazy init per ID on first access.
fn oncelock_array() -> &'static [OnceLock<u64>] {
    static ARRAY: OnceLock<Vec<OnceLock<u64>>> = OnceLock::new();
    let arr = ARRAY.get_or_init(|| (0..NUM_STRINGS).map(|_| OnceLock::new()).collect());
    // Eagerly initialize all entries so we measure steady-state read cost.
    let pool = compact_pool();
    for (i, cell) in arr.iter().enumerate() {
        cell.get_or_init(|| pool.get(test_strings()[i]).unwrap());
    }
    arr
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

// ─── Bench D: Pre-resolved ID read cost (1M reads, reports reads/sec) ───────
//
// The ID u64 was resolved ONCE from the pool at startup.
// Now we just read it from different storage mechanisms.
// This measures the pure read overhead of each container.

mod id_read_cost {
    use super::*;

    /// Plain Vec<u64> indexed with bounds check.
    #[divan::bench(threads = [1, 2, 4])]
    fn eager_vec_index(bencher: Bencher) {
        let vec = eager_vec();
        let indices = zipf_indices();
        bencher.counter(ItemsCount::new(indices.len())).bench(|| {
            for &i in indices {
                black_box(vec[i]);
            }
        });
    }

    /// Vec<u64> with unchecked indexing — absolute minimum: one `mov` instruction.
    #[divan::bench(threads = [1, 2, 4])]
    fn eager_vec_unchecked(bencher: Bencher) {
        let vec = eager_vec();
        let indices = zipf_indices();
        bencher.counter(ItemsCount::new(indices.len())).bench(|| {
            for &i in indices {
                // SAFETY: zipf_indices generates values in 0..NUM_STRINGS, vec has NUM_STRINGS entries.
                black_box(unsafe { *vec.get_unchecked(i) });
            }
        });
    }

    /// AtomicU64 with Relaxed load. On x86 this is a plain `mov`.
    #[divan::bench(threads = [1, 2, 4])]
    fn atomic_u64_relaxed(bencher: Bencher) {
        let arr = atomic_array();
        let indices = zipf_indices();
        bencher.counter(ItemsCount::new(indices.len())).bench(|| {
            for &i in indices {
                black_box(arr[i].load(Ordering::Relaxed));
            }
        });
    }

    /// OnceLock<u64>::get() — Acquire load + is-initialized check on every read.
    #[divan::bench(threads = [1, 2, 4])]
    fn oncelock_get(bencher: Bencher) {
        let arr = oncelock_array();
        let indices = zipf_indices();
        bencher.counter(ItemsCount::new(indices.len())).bench(|| {
            for &i in indices {
                black_box(arr[i].get().copied());
            }
        });
    }

    /// CompactPool::resolve(StringId) — pre-hashed, probes flat array.
    #[divan::bench(threads = [1, 2, 4])]
    fn compact_pool_resolve(bencher: Bencher) {
        let pool = compact_pool();
        let ids = zipf_ids();
        bencher.counter(ItemsCount::new(ids.len())).bench(|| {
            for &id in ids {
                black_box(pool.resolve(id));
            }
        });
    }

    /// FxHashMap::get(&str) — full hash + key comparison every time.
    /// This is the "no pre-resolution" baseline.
    #[divan::bench(threads = [1, 2, 4])]
    fn fxhashmap_str_lookup(bencher: Bencher) {
        let map = fx_hashmap();
        let keys = zipf_keys();
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            for key in keys {
                black_box(map.get(key));
            }
        });
    }
}

// ─── Bench E: AtomicU16+sentinel vs OnceLock head-to-head ────────────────────
//
// 200 IDs, pre-initialized. 1M Zipfian reads.
// Measures the PURE read cost of each caching strategy.

mod atomic_vs_oncelock {
    use intern_pool_bench::CachedIdAtomic;

    use super::*;

    // 200 CachedIdAtomic statics
    static ATOMIC_IDS: [CachedIdAtomic; NUM_STRINGS] =
        [const { CachedIdAtomic::new() }; NUM_STRINGS];

    fn init_atomic_ids() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            for (i, id) in ATOMIC_IDS.iter().enumerate() {
                id.get_or_init(|| i as u16);
            }
        });
    }

    // 200 OnceLock<u16> statics
    static ONCELOCK_IDS: [OnceLock<u16>; NUM_STRINGS] = [const { OnceLock::new() }; NUM_STRINGS];

    fn init_oncelock_ids() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            for (i, id) in ONCELOCK_IDS.iter().enumerate() {
                id.get_or_init(|| i as u16);
            }
        });
    }

    /// AtomicU16 Relaxed load — plain `mov` on x86, `ldr` on ARM.
    #[divan::bench(threads = [1, 2, 4])]
    fn atomic_u16_relaxed(bencher: Bencher) {
        init_atomic_ids();
        let indices = zipf_indices();
        bencher.counter(ItemsCount::new(indices.len())).bench(|| {
            for &i in indices {
                black_box(ATOMIC_IDS[i].get());
            }
        });
    }

    /// OnceLock<u16>::get() — Acquire load + initialized check.
    #[divan::bench(threads = [1, 2, 4])]
    fn oncelock_u16_get(bencher: Bencher) {
        init_oncelock_ids();
        let indices = zipf_indices();
        bencher.counter(ItemsCount::new(indices.len())).bench(|| {
            for &i in indices {
                black_box(ONCELOCK_IDS[i].get().copied());
            }
        });
    }

    /// AtomicU16 get_or_init (steady state — already initialized).
    #[divan::bench(threads = [1, 2, 4])]
    fn atomic_u16_get_or_init(bencher: Bencher) {
        init_atomic_ids();
        let indices = zipf_indices();
        bencher.counter(ItemsCount::new(indices.len())).bench(|| {
            for &i in indices {
                black_box(ATOMIC_IDS[i].get_or_init(|| i as u16));
            }
        });
    }

    /// OnceLock<u16>::get_or_init (steady state — already initialized).
    #[divan::bench(threads = [1, 2, 4])]
    fn oncelock_u16_get_or_init(bencher: Bencher) {
        init_oncelock_ids();
        let indices = zipf_indices();
        bencher.counter(ItemsCount::new(indices.len())).bench(|| {
            for &i in indices {
                black_box(ONCELOCK_IDS[i].get_or_init(|| i as u16));
            }
        });
    }
}

// ─── Bench F: Global InternedRegistry (auto-assigned ordinals) ───────────────
//
// Simulates the proposed design: each encoding declares a `static EncodingId`,
// ordinals are auto-assigned at init, reads are a plain array index.

mod global_registry {
    use intern_pool_bench::InternedRegistry;

    use super::*;

    // 200 global static EncodingIds — one per encoding.
    // In the real code, each encoding crate would own its own static.
    static ENCODING_IDS: [EncodingId; NUM_STRINGS] = {
        // const array init — each starts as UNSET
        [const { EncodingId::unset() }; NUM_STRINGS]
    };

    /// Build the registry once, assigning ordinals 0..199 to each encoding.
    fn frozen_registry() -> &'static InternedRegistry<u64> {
        static REG: OnceLock<InternedRegistry<u64>> = OnceLock::new();
        REG.get_or_init(|| {
            let strings = test_strings();
            let mut builder = RegistryBuilder::new();
            for (i, &name) in strings.iter().enumerate() {
                builder.register(&ENCODING_IDS[i], name, i as u64);
            }
            builder.freeze()
        })
    }

    /// Zipfian ordinals — read from the static EncodingIds.
    fn zipf_ordinals() -> &'static [u16] {
        static ORDS: OnceLock<Vec<u16>> = OnceLock::new();
        // Make sure registry is initialized first.
        let _ = frozen_registry();
        ORDS.get_or_init(|| {
            zipf_indices()
                .iter()
                .map(|&i| ENCODING_IDS[i].get_unchecked())
                .collect()
        })
    }

    /// Full hot path: read static EncodingId → index into frozen registry.
    /// This is what production code would do.
    #[divan::bench(threads = [1, 2, 4])]
    fn static_id_to_registry(bencher: Bencher) {
        let reg = frozen_registry();
        let indices = zipf_indices();
        bencher.counter(ItemsCount::new(indices.len())).bench(|| {
            for &i in indices {
                // 1. Read ordinal from global static (AtomicU16 Relaxed)
                let ord = ENCODING_IDS[i].get_unchecked();
                // 2. Index into frozen registry (array access)
                black_box(reg.get_unchecked(ord));
            }
        });
    }

    /// Pre-resolved ordinals: skip the AtomicU16 load, just index directly.
    #[divan::bench(threads = [1, 2, 4])]
    fn preresolved_ordinal(bencher: Bencher) {
        let reg = frozen_registry();
        let ords = zipf_ordinals();
        bencher.counter(ItemsCount::new(ords.len())).bench(|| {
            for &ord in ords {
                black_box(reg.get_unchecked(ord));
            }
        });
    }

    /// Compare: FxHashMap string lookup (what vortex does today).
    #[divan::bench(threads = [1, 2, 4])]
    fn dashmap_str_baseline(bencher: Bencher) {
        let map = dashmap_pool();
        let keys = zipf_keys();
        bencher.counter(ItemsCount::new(keys.len())).bench(|| {
            for key in keys {
                black_box(map.get(key).map(|v| *v));
            }
        });
    }
}
