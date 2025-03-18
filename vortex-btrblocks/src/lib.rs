#![feature(array_chunks)]

use std::any::TypeId;
use std::fmt::Debug;
use std::hash::Hash;

use float::{FloatScheme, FloatStats};
use integer::{IntegerScheme, IntegerStats};
use parking_lot::Mutex;
use string::{StringScheme, StringStats};
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::arcref::ArcRef;
use vortex_array::arrays::{ExtensionArray, ListArray, StructArray, TemporalArray};
use vortex_array::nbytes::NBytes;
use vortex_array::variants::{ExtensionArrayTrait, PrimitiveArrayTrait, StructArrayTrait};
use vortex_array::{Array, ArrayRef, Canonical};
use vortex_dtype::datetime::TemporalMetadata;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult};

pub use crate::float::FloatCompressor;
pub use crate::integer::IntCompressor;
pub use crate::string::StringCompressor;
pub use crate::temporal::compress_temporal;

mod downscale;
mod float;
pub mod integer;
mod patches;
mod sample;
mod string;
mod temporal;

pub struct GenerateStatsOptions {
    pub count_distinct_values: bool,
    // pub count_runs: bool,
    // should this be scheme-specific?
}

impl Default for GenerateStatsOptions {
    fn default() -> Self {
        Self {
            count_distinct_values: true,
            // count_runs: true,
        }
    }
}

/// Stats for the compressor.
pub trait CompressorStats: Debug + Clone {
    type ArrayType: Array;

    // Generate with options.
    fn generate(input: &Self::ArrayType) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default())
    }

    fn generate_opts(input: &Self::ArrayType, opts: GenerateStatsOptions) -> Self;

    fn source(&self) -> &Self::ArrayType;

    fn sample(&self, sample_size: u16, sample_count: u16) -> Self {
        self.sample_opts(sample_size, sample_count, GenerateStatsOptions::default())
    }

    fn sample_opts(&self, sample_size: u16, sample_count: u16, opts: GenerateStatsOptions) -> Self;

    fn is_similar(&self, _other: &Self) -> bool {
        false
    }
}

/// Top-level compression scheme trait.
///
/// Variants are specialized for each data type, e.g. see `IntegerScheme`, `FloatScheme`, etc.
pub trait Scheme: Debug {
    type StatsType: CompressorStats;
    type CodeType: Copy + Eq + Hash;

    /// Scheme unique identifier.
    fn code(&self) -> Self::CodeType;

    /// True if this is the singular Constant scheme for this data type.
    fn is_constant(&self) -> bool {
        false
    }

    /// Estimate the compression ratio for running this scheme (and its children)
    /// for the given input.
    ///
    /// Depth is the depth in the encoding tree we've already reached before considering this
    /// scheme.
    ///
    /// Returns the estimated compression ratio as well as the tree of compressors to use.
    fn expected_compression_ratio(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[Self::CodeType],
    ) -> VortexResult<f64> {
        estimate_compression_ratio_with_sampling(
            self,
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    /// Compress the input with this scheme, yielding a new array.
    fn compress(
        &self,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[Self::CodeType],
        state: Option<&CompressorState>,
    ) -> VortexResult<ArrayRef>;
}

pub struct SchemeTree {
    /// Scheme to use for the array.
    ///
    /// This is in the type-specific code space, for example either the `IntCompressor` or
    /// `FloatCompressor` code space.
    pub scheme: u8,
    /// Specified schemes to use for children.
    pub children: Vec<SchemeTree>,
}

pub fn estimate_compression_ratio_with_sampling<T: Scheme + ?Sized>(
    compressor: &T,
    stats: &T::StatsType,
    is_sample: bool,
    allowed_cascading: usize,
    excludes: &[T::CodeType],
) -> VortexResult<f64> {
    let sample = if is_sample {
        stats.clone()
    } else {
        stats.sample(64, 10)
    };

    let after = compressor
        .compress(&sample, true, allowed_cascading, excludes, None)?
        .nbytes();
    let before = sample.source().nbytes();

    log::debug!(
        "estimate_compression_ratio_with_sampling(compressor={compressor:#?} is_sample={is_sample}, allowed_cascading={allowed_cascading}) = {}",
        before as f64 / after as f64
    );

    Ok(before as f64 / after as f64)
}

const MAX_CASCADE: usize = 3;

/// A compressor for a particular input type.
///
/// The `Input` type should be one of the canonical array variants, e.g. `PrimitiveArray`.
///
/// Compressors expose a `compress` function.
pub trait Compressor {
    type ArrayType: Array;
    type SchemeType: Scheme<StatsType = Self::StatsType> + ?Sized;

    // Stats type instead?
    type StatsType: CompressorStats<ArrayType = Self::ArrayType>;

    fn schemes() -> &'static [&'static Self::SchemeType];
    fn default_scheme() -> &'static Self::SchemeType;
    fn dict_scheme_code() -> <Self::SchemeType as Scheme>::CodeType;

    fn compress(
        array: &Self::ArrayType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[<Self::SchemeType as Scheme>::CodeType],
        state: Option<&CompressorState>,
    ) -> VortexResult<ArrayRef>
    where
        Self::SchemeType: 'static,
        Self: Sized + 'static,
    {
        // Avoid compressing empty arrays.
        if array.is_empty() {
            return Ok(array.to_array());
        }

        // Generate stats on the array directly.
        let stats = if excludes.contains(&Self::dict_scheme_code()) {
            Self::StatsType::generate_opts(
                array,
                GenerateStatsOptions {
                    count_distinct_values: false,
                },
            )
        } else {
            Self::StatsType::generate(array)
        };

        let similar_stats = state.and_then(|state| state.get_similar_to::<Self>(&stats));

        let best_scheme = match similar_stats {
            Some(stats) => stats,
            None => {
                let scheme = ArcRef::new_ref(Self::choose_scheme(
                    &stats,
                    is_sample,
                    allowed_cascading,
                    excludes,
                )?);

                if let Some(state) = state {
                    state.remember::<Self>(stats.clone(), scheme.clone());
                }
                scheme
            }
        };

        let output = best_scheme.compress(&stats, is_sample, allowed_cascading, excludes, state)?;
        if output.nbytes() < array.nbytes() {
            Ok(output)
        } else {
            log::debug!("resulting tree too large: {}", output.tree_display());
            Ok(array.to_array())
        }
    }

    fn choose_scheme(
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[<Self::SchemeType as Scheme>::CodeType],
    ) -> VortexResult<&'static Self::SchemeType> {
        let mut best_ratio = 1.0;
        let mut best_scheme: Option<&'static Self::SchemeType> = None;

        // logging helpers
        let depth = MAX_CASCADE - allowed_cascading;

        for scheme in Self::schemes().iter() {
            if excludes.contains(&scheme.code()) {
                continue;
            }

            // We never choose Constant for a sample
            if is_sample && scheme.is_constant() {
                continue;
            }

            log::debug!("depth={depth} is_sample={is_sample} trying scheme: {scheme:#?}",);

            let ratio =
                scheme.expected_compression_ratio(stats, is_sample, allowed_cascading, excludes)?;
            log::debug!("depth={depth} is_sample={is_sample} scheme: {scheme:#?} ratio = {ratio}");

            if ratio > best_ratio {
                best_ratio = ratio;
                let _ = best_scheme.insert(*scheme);
            }
        }

        log::trace!("depth={depth} best scheme = {best_scheme:#?}  ratio = {best_ratio}");

        if let Some(best) = best_scheme {
            Ok(best)
        } else {
            Ok(Self::default_scheme())
        }
    }
}

#[derive(Debug, Default)]
pub struct BtrBlocksCompressor {
    inner: CompressorState,
}

impl BtrBlocksCompressor {
    pub fn empty() -> Self {
        Self {
            inner: CompressorState::empty(),
        }
    }
}

#[derive(Debug)]
pub struct CompressorState {
    map: Mutex<HashMap<TypeId, Box<dyn std::any::Any + Send>>>,
}

impl Default for CompressorState {
    fn default() -> Self {
        let mut map = HashMap::default();
        map.insert(
            TypeId::of::<IntCompressor>(),
            Box::new(Vec::<(IntegerStats, ArcRef<dyn IntegerScheme>)>::default()) as _,
        );
        map.insert(
            TypeId::of::<FloatCompressor>(),
            Box::new(Vec::<(FloatStats, ArcRef<dyn FloatScheme>)>::default()) as _,
        );
        map.insert(
            TypeId::of::<StringCompressor>(),
            Box::new(Vec::<(StringStats, ArcRef<dyn StringScheme>)>::default()) as _,
        );

        let map = Mutex::new(map);

        Self { map }
    }
}

impl CompressorState {
    pub fn empty() -> Self {
        Self {
            map: Default::default(),
        }
    }

    fn get_similar_to<C: Compressor + 'static>(
        &self,
        stats: &C::StatsType,
    ) -> Option<ArcRef<C::SchemeType>> {
        log::debug!("Looking for compression scheme");
        let guard = self.map.lock();
        let v = guard.get(&TypeId::of::<C>())?;

        let v = v
            .downcast_ref::<Vec<(C::StatsType, ArcRef<C::SchemeType>)>>()
            .vortex_expect("Must be initialized.");

        let stats = v
            .iter()
            .find(|(stored, _)| stored.is_similar(stats))
            .map(|v| v.1.clone());

        if stats.is_some() {
            log::debug!("Found existing compression scheme");
        } else {
            log::debug!("Couldn't find a similar compression scheme");
        }

        stats
    }

    fn remember<C: Compressor + 'static>(
        &self,
        stats: C::StatsType,
        scheme: ArcRef<C::SchemeType>,
    ) {
        log::debug!("remembering compression scheme");
        let mut guard = self.map.lock();

        if let Some(v) = guard.get_mut(&TypeId::of::<C>()) {
            let v = v
                .downcast_mut::<Vec<(C::StatsType, ArcRef<C::SchemeType>)>>()
                .vortex_expect("Must be initialized.");

            v.push((stats, scheme));
        }
    }
}

impl BtrBlocksCompressor {
    #[allow(clippy::only_used_in_recursion)]
    pub fn compress(&self, array: &dyn Array) -> VortexResult<ArrayRef> {
        let state = Some(&self.inner);
        match array.to_canonical()? {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            // TODO(aduffy): Sparse, other bool compressors.
            Canonical::Bool(bool_array) => Ok(bool_array.into_array()),
            Canonical::Primitive(primitive) => {
                if primitive.ptype().is_int() {
                    IntCompressor::compress(&primitive, false, MAX_CASCADE, &[], state)
                } else {
                    FloatCompressor::compress(&primitive, false, MAX_CASCADE, &[], state)
                }
            }
            Canonical::Struct(struct_array) => {
                let mut fields = Vec::new();
                for idx in 0..struct_array.nfields() {
                    let field = struct_array
                        .maybe_null_field_by_idx(idx)
                        .vortex_expect("field access");
                    let compressed = self.compress(&field)?;
                    fields.push(compressed);
                }

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity().clone(),
                )?
                .into_array())
            }
            Canonical::List(list_array) => {
                // Compress the inner
                let compressed_elems = self.compress(list_array.elements())?;
                let compressed_offsets = self.compress(list_array.offsets())?;

                Ok(ListArray::try_new(
                    compressed_elems,
                    compressed_offsets,
                    list_array.validity().clone(),
                )?
                .into_array())
            }
            Canonical::VarBinView(strings) => {
                if strings
                    .dtype()
                    .eq_ignore_nullability(&DType::Utf8(Nullability::NonNullable))
                {
                    StringCompressor::compress(&strings, false, MAX_CASCADE, &[], state)
                } else {
                    // Binary arrays do not compress
                    Ok(strings.into_array())
                }
            }
            Canonical::Extension(ext_array) => {
                // We compress Timestamp-level arrays with DateTimeParts compression
                if let Ok(temporal_array) =
                    TemporalArray::try_from(ext_array.to_array().into_array())
                {
                    if let TemporalMetadata::Timestamp(..) = temporal_array.temporal_metadata() {
                        return compress_temporal(temporal_array);
                    }
                }

                // Compress the underlying storage array.
                let compressed_storage = self.compress(ext_array.storage())?;

                Ok(
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array(),
                )
            }
        }
    }
}
