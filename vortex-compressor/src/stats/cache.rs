// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-compression-site statistics cache and the [`ArrayAndStats`] bundle.

use std::any::Any;
use std::any::TypeId;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Bool;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::VarBinView;
use vortex_error::VortexExpect;

use super::BoolStats;
use super::FloatStats;
use super::GenerateStatsOptions;
use super::IntegerStats;
use super::StringStats;
use crate::trace;

/// A single cache entry: a concrete [`TypeId`] paired with a type-erased value.
type StatsEntry = (TypeId, Arc<dyn Any + Send + Sync>);

/// Cache for compression statistics, keyed by concrete type.
///
/// The cache is interior-mutable: entries can be inserted through a shared [`&StatsCache`]
/// borrow. Values are stored as [`Arc<dyn Any + Send + Sync>`] so that cached entries can be
/// cloned out of the lock cheaply and handed back to callers as [`Arc<T>`].
struct StatsCache {
    // TODO(connor): We could further optimize this with a `SmallVec` here.
    /// The cache entries, keyed by [`TypeId`].
    ///
    /// The total number of statistics types in this stats should be relatively small, so we use a
    /// vector instead of a hash map.
    entries: Arc<Mutex<Vec<StatsEntry>>>,
}

impl StatsCache {
    /// Creates a new empty cache.
    fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns a cached value, computing it on first access.
    fn get_or_insert_with<T: Send + Sync + 'static>(&self, f: impl FnOnce() -> T) -> Arc<T> {
        let type_id = TypeId::of::<T>();
        let mut guard = self.entries.lock();

        if let Some(pos) = guard.iter().position(|(id, _)| *id == type_id) {
            Arc::clone(&guard[pos].1)
                .downcast::<T>()
                .ok()
                .vortex_expect("we just checked the TypeID")
        } else {
            let new_arc: Arc<T> = {
                let _span = trace::generate_stats_span(std::any::type_name::<T>()).entered();
                Arc::new(f())
            };
            guard.push((type_id, Arc::clone(&new_arc) as Arc<dyn Any + Send + Sync>));
            new_arc
        }
    }
}

/// An array bundled with its lazily-computed statistics cache.
///
/// The cache is guaranteed to correspond to the array. When a scheme creates a derived array (e.g.
/// FoR bias subtraction), it must create a new [`ArrayAndStats`] so that stale stats from the
/// original array are not reused.
///
/// Built-in stats are accessed via typed methods ([`integer_stats`], [`float_stats`],
/// [`string_stats`]) which generate stats lazily on first access using the stored
/// [`GenerateStatsOptions`].
///
/// Extension schemes can use [`get_or_insert_with`] for custom stats types.
///
/// [`integer_stats`]: ArrayAndStats::integer_stats
/// [`float_stats`]: ArrayAndStats::float_stats
/// [`string_stats`]: ArrayAndStats::string_stats
/// [`get_or_insert_with`]: ArrayAndStats::get_or_insert_with
pub struct ArrayAndStats {
    /// The array. This is always in canonical form.
    array: ArrayRef,
    /// The stats cache.
    cache: StatsCache,
    /// The stats generation options.
    opts: GenerateStatsOptions,
}

impl ArrayAndStats {
    /// Creates a new bundle with the given stats generation options.
    ///
    /// Stats are generated lazily on first access via the typed accessor methods.
    ///
    /// # Panics
    ///
    /// Panics if the array is not canonical.
    pub fn new(array: ArrayRef, opts: GenerateStatsOptions) -> Self {
        assert!(
            array.is_canonical(),
            "ArrayAndStats should only be created with canonical arrays"
        );

        Self {
            array,
            cache: StatsCache::new(),
            opts,
        }
    }

    /// Returns a reference to the array.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Returns the array as an [`ArrayView<Primitive>`].
    ///
    /// # Panics
    ///
    /// Panics if the array is not a primitive array.
    pub fn array_as_primitive(&self) -> ArrayView<'_, Primitive> {
        self.array
            .as_opt::<Primitive>()
            .vortex_expect("the array is guaranteed to already be canonical by construction")
    }

    /// Returns the array as an [`ArrayView<VarBinView>`].
    ///
    /// # Panics
    ///
    /// Panics if the array is not a UTF-8 string array.
    pub fn array_as_utf8(&self) -> ArrayView<'_, VarBinView> {
        self.array
            .as_opt::<VarBinView>()
            .vortex_expect("the array is guaranteed to already be canonical by construction")
    }

    /// Consumes the bundle and returns the array.
    pub fn into_array(self) -> ArrayRef {
        self.array
    }

    /// Returns the length of the array.
    pub fn array_len(&self) -> usize {
        self.array.len()
    }

    /// Returns bool stats, generating them lazily on first access.
    pub fn bool_stats(&self, ctx: &mut ExecutionCtx) -> Arc<BoolStats> {
        let array = self.array.clone();
        self.cache.get_or_insert_with::<BoolStats>(|| {
            let bool_array = array
                .as_opt::<Bool>()
                .vortex_expect("the array is guaranteed to already be canonical by construction")
                .into_owned();
            BoolStats::generate(&bool_array, ctx).vortex_expect("BoolStats shouldn't fail")
        })
    }

    /// Returns integer stats, generating them lazily on first access.
    pub fn integer_stats(&self, ctx: &mut ExecutionCtx) -> Arc<IntegerStats> {
        let array = self.array.clone();
        let opts = self.opts;
        self.cache.get_or_insert_with::<IntegerStats>(|| {
            let primitive = array
                .as_opt::<Primitive>()
                .vortex_expect("the array is guaranteed to already be canonical by construction")
                .into_owned();
            IntegerStats::generate_opts(&primitive, opts, ctx)
        })
    }

    /// Returns float stats, generating them lazily on first access.
    pub fn float_stats(&self, ctx: &mut ExecutionCtx) -> Arc<FloatStats> {
        let array = self.array.clone();
        let opts = self.opts;
        self.cache.get_or_insert_with::<FloatStats>(|| {
            let primitive = array
                .as_opt::<Primitive>()
                .vortex_expect("the array is guaranteed to already be canonical by construction")
                .into_owned();
            FloatStats::generate_opts(&primitive, opts, ctx)
        })
    }

    /// Returns string stats, generating them lazily on first access.
    pub fn string_stats(&self, ctx: &mut ExecutionCtx) -> Arc<StringStats> {
        let array = self.array.clone();
        let opts = self.opts;
        self.cache.get_or_insert_with::<StringStats>(|| {
            let varbinview = array
                .as_opt::<VarBinView>()
                .vortex_expect("the array is guaranteed to already be canonical by construction")
                .into_owned();
            StringStats::generate_opts(&varbinview, opts, ctx)
        })
    }

    /// For extension schemes with custom stats types.
    pub fn get_or_insert_with<T: Send + Sync + 'static>(&self, f: impl FnOnce() -> T) -> Arc<T> {
        self.cache.get_or_insert_with::<T>(f)
    }
}
