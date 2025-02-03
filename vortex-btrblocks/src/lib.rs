use std::fmt::Debug;
use std::ops::Deref;

use vortex_array::array::{ListArray, StructArray};
use vortex_array::variants::{PrimitiveArrayTrait, StructArrayTrait};
use vortex_array::{Array, Canonical, IntoArray, IntoCanonical};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult};

use crate::float::FloatCompressor;
use crate::integer::IntCompressor;
use crate::string::StringCompressor;

mod downscale;
mod float;
pub mod integer;
mod sample;
mod string;

/// Stats for the compressor.
pub trait CompressorStats: Clone {
    type ArrayType: Deref<Target = Array>;

    fn generate(input: &Self::ArrayType) -> Self;

    fn source(&self) -> &Self::ArrayType;

    fn sample(&self, sample_size: u16, sample_count: u16) -> Self;
}

/// Top-level compression scheme trait.
///
/// Variants are specialized for each data type, e.g. see `IntegerScheme`, `FloatScheme`, etc.
pub trait Scheme: Debug {
    type StatsType: CompressorStats;

    /// Scheme unique identifier.
    fn code(&self) -> u8;

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
        excludes: &[u8],
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
        excludes: &[u8],
    ) -> VortexResult<Array>;
}

pub fn estimate_compression_ratio_with_sampling<T: Scheme + ?Sized>(
    compressor: &T,
    stats: &T::StatsType,
    is_sample: bool,
    allowed_cascading: usize,
    excludes: &[u8],
) -> VortexResult<f64> {
    let sample = if is_sample {
        stats.clone()
    } else {
        stats.sample(64, 10)
    };

    let after = compressor
        .compress(&sample, true, allowed_cascading, excludes)?
        .nbytes();
    let before = stats.source().nbytes();

    Ok(before as f64 / after as f64)
}

const MAX_CASCADE: usize = 3;

/// A compressor for a particular input type.
///
/// The `Input` type should be one of the canonical array variants, e.g. `PrimitiveArray`.
///
/// Compressors expose a `compress` function.
pub trait Compressor {
    type ArrayType: Deref<Target = Array>;
    type SchemeType: Scheme<StatsType = Self::StatsType> + ?Sized;

    // Stats type instead?
    type StatsType: CompressorStats<ArrayType = Self::ArrayType>;

    fn schemes() -> &'static [&'static Self::SchemeType];
    fn default_scheme() -> &'static Self::SchemeType;

    fn compress(
        array: &Self::ArrayType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[u8],
    ) -> VortexResult<Array>
    where
        Self::SchemeType: 'static,
    {
        // Generate stats on the array directly.
        let stats = Self::StatsType::generate(array);
        let best_scheme = Self::choose_scheme(&stats, is_sample, allowed_cascading, excludes)?;

        let output = best_scheme.compress(&stats, is_sample, allowed_cascading, excludes)?;
        if output.nbytes() < array.nbytes() {
            Ok(output)
        } else {
            Ok(array.deref().clone())
        }
    }

    fn choose_scheme(
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[u8],
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

            log::trace!("depth={depth} is_sample={is_sample} trying scheme: {scheme:#?}",);

            let ratio =
                scheme.expected_compression_ratio(stats, is_sample, allowed_cascading, excludes)?;
            log::trace!("depth={depth} is_sample={is_sample} scheme: {scheme:#?} ratio = {ratio}");

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

pub struct BtrBlocksCompressor;

impl BtrBlocksCompressor {
    #[allow(clippy::only_used_in_recursion)]
    pub fn compress(&self, array: Array) -> VortexResult<Array> {
        match array.into_canonical()? {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            Canonical::Bool(bool_array) => Ok(bool_array.into_array()),
            Canonical::Primitive(primitive) => {
                if primitive.ptype().is_int() {
                    IntCompressor::compress(&primitive, false, MAX_CASCADE, &[])
                } else {
                    FloatCompressor::compress(&primitive, false, MAX_CASCADE, &[])
                }
            }
            Canonical::Struct(struct_array) => {
                let mut fields = Vec::new();
                for idx in 0..struct_array.nfields() {
                    let field = struct_array
                        .maybe_null_field_by_idx(idx)
                        .vortex_expect("field access");
                    let compressed = self.compress(field)?;
                    fields.push(compressed);
                }

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity(),
                )?
                .into_array())
            }
            Canonical::List(list_array) => {
                // Compress the inner
                let compressed_elems = self.compress(list_array.elements())?;
                let compressed_offsets = self.compress(list_array.offsets())?;

                Ok(
                    ListArray::try_new(
                        compressed_elems,
                        compressed_offsets,
                        list_array.validity(),
                    )?
                    .into_array(),
                )
            }
            Canonical::VarBinView(strings) => {
                if strings
                    .dtype()
                    .eq_ignore_nullability(&DType::Utf8(Nullability::NonNullable))
                {
                    StringCompressor::compress(&strings, false, MAX_CASCADE, &[])
                } else {
                    // Binary arrays do not compress
                    Ok(strings.into_array())
                }
            }
            Canonical::Extension(ext_array) => {
                // Canonicalize chunked array.
                Ok(ext_array.into_array())
            }
        }
    }
}
