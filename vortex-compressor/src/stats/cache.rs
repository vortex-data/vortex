// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-compression-site statistics cache and the [`ArrayAndStats`] bundle.

use std::any::Any;
use std::any::TypeId;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::VarBinView;
use vortex_error::VortexExpect;

use super::BoolStats;
use super::FloatStats;
use super::GenerateStatsOptions;
use super::IntegerStats;
use super::StringStats;

/// Cache for compression statistics, keyed by concrete type.
struct StatsCache {
    // TODO(connor): We could further optimize this with a `SmallVec` here.
    /// The cache entries, keyed by [`TypeId`].
    ///
    /// The total number of statistics types in this stats should be relatively small, so we use a
    /// vector instead of a hash map.
    entries: Vec<(TypeId, Box<dyn Any>)>,
}

impl StatsCache {
    /// Creates a new empty cache.
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Returns a cached value, computing it on first access.
    fn get_or_insert_with<T: 'static>(&mut self, f: impl FnOnce() -> T) -> &T {
        let type_id = TypeId::of::<T>();
        let pos = self.entries.iter().position(|(id, _)| *id == type_id);

        if let Some(pos) = pos {
            self.entries[pos]
                .1
                .downcast_ref::<T>()
                .vortex_expect("we just checked the TypeID")
        } else {
            self.entries.push((type_id, Box::new(f())));
            self.entries
                .last()
                .vortex_expect("just pushed")
                .1
                .downcast_ref::<T>()
                .vortex_expect("we just checked the TypeID")
        }
    }
}

/// An array bundled with its lazily-computed statistics cache.
///
/// The cache is guaranteed to correspond to the array. When a scheme creates a derived array (e.g.
/// FoR bias subtraction), it must create a new [`ArrayAndStats`] so that stale stats from the
/// original array are not reused.
///
/// Built-in stats are accessed via typed methods (`integer_stats`, `float_stats`, `string_stats`)
/// which generate stats lazily on first access using the stored [`GenerateStatsOptions`].
///
/// Extension schemes can use `get_or_insert_with` for custom stats types.
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
    pub fn bool_stats(&mut self) -> &BoolStats {
        let array = self.array.clone();

        self.cache.get_or_insert_with::<BoolStats>(|| {
            #[expect(deprecated)]
            let bool_array = array.to_bool();
            BoolStats::generate(&bool_array).vortex_expect("BoolStats shouldn't fail")
        })
    }

    // TODO(connor): These should all have interior mutability instead!!!

    /// Returns integer stats, generating them lazily on first access.
    pub fn integer_stats(&mut self) -> &IntegerStats {
        let array = self.array.clone();
        let opts = self.opts;

        self.cache.get_or_insert_with::<IntegerStats>(|| {
            #[expect(deprecated)]
            let primitive = array.to_primitive();
            IntegerStats::generate_opts(&primitive, opts)
        })
    }

    /// Returns float stats, generating them lazily on first access.
    pub fn float_stats(&mut self) -> &FloatStats {
        let array = self.array.clone();
        let opts = self.opts;

        self.cache.get_or_insert_with::<FloatStats>(|| {
            #[expect(deprecated)]
            let primitive = array.to_primitive();
            FloatStats::generate_opts(&primitive, opts)
        })
    }

    /// Returns string stats, generating them lazily on first access.
    pub fn string_stats(&mut self) -> &StringStats {
        let array = self.array.clone();
        let opts = self.opts;

        self.cache.get_or_insert_with::<StringStats>(|| {
            #[expect(deprecated)]
            let varbinview = array.to_varbinview();
            StringStats::generate_opts(&varbinview, opts)
        })
    }

    /// For extension schemes with custom stats types.
    pub fn get_or_insert_with<T: 'static>(&mut self, f: impl FnOnce() -> T) -> &T {
        self.cache.get_or_insert_with::<T>(f)
    }
}
