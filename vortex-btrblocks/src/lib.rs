// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Vortex's [BtrBlocks]-inspired adaptive compression framework.
//!
//! This crate provides a sophisticated multi-level compression system that adaptively selects
//! optimal compression schemes based on data characteristics. The compressor analyzes arrays
//! to determine the best encoding strategy, supporting cascaded compression with multiple
//! encoding layers for maximum efficiency.
//!
//! # Key Features
//!
//! - **Adaptive Compression**: Automatically selects the best compression scheme based on data patterns
//! - **Type-Specific Compressors**: Specialized compression for integers, floats, strings, and temporal data
//! - **Cascaded Encoding**: Multiple compression layers can be applied for optimal results
//! - **Statistical Analysis**: Uses data sampling and statistics to predict compression ratios
//! - **Recursive Structure Handling**: Compresses nested structures like structs and lists
//!
//! # Example
//!
//! ```rust
//! use vortex_btrblocks::{BtrBlocksCompressor, BtrBlocksCompressorBuilder, IntCode};
//! use vortex_array::Array;
//!
//! // Default compressor with all schemes enabled
//! let compressor = BtrBlocksCompressor::default();
//!
//! // Configure with builder to exclude specific schemes
//! let compressor = BtrBlocksCompressorBuilder::new()
//!     .exclude_int([IntCode::Dict])
//!     .build();
//! ```
//!
//! [BtrBlocks]: https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::list_from_list_view;
use vortex_array::compute::Cost;
use vortex_array::compute::IsConstantOpts;
use vortex_array::compute::is_constant_opts;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityHelper;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::datetime::Timestamp;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::decimal::compress_decimal;
pub use crate::float::FloatCode;
pub use crate::float::FloatCompressor;
pub use crate::float::FloatStats;
pub use crate::float::dictionary::dictionary_encode as float_dictionary_encode;
pub use crate::integer::IntCode;
pub use crate::integer::IntCompressor;
pub use crate::integer::IntegerStats;
pub use crate::integer::dictionary::dictionary_encode as integer_dictionary_encode;
pub use crate::string::StringCode;
pub use crate::string::StringCompressor;
pub use crate::string::StringStats;
pub use crate::temporal::compress_temporal;

mod builder;
mod decimal;
mod float;
mod integer;
mod patches;
mod rle;
mod sample;
mod string;
mod temporal;

pub use builder::BtrBlocksCompressorBuilder;

use crate::float::FloatScheme;
use crate::integer::IntegerScheme;
use crate::string::StringScheme;

/// Holds references to exclude lists for each compression code type.
///
/// This struct is passed through recursive compression calls to specify
/// which schemes should be excluded at each level.
#[derive(Debug, Clone, Copy, Default)]
pub struct Excludes<'a> {
    /// Integer schemes to exclude.
    pub int: &'a [IntCode],
    /// Float schemes to exclude.
    pub float: &'a [FloatCode],
    /// String schemes to exclude.
    pub string: &'a [StringCode],
}

impl<'a> Excludes<'a> {
    /// Creates an empty excludes (no exclusions).
    pub const fn none() -> Self {
        Self {
            int: &[],
            float: &[],
            string: &[],
        }
    }

    /// Creates excludes with only integer exclusions.
    pub const fn int_only(int: &'a [IntCode]) -> Self {
        Self {
            int,
            float: &[],
            string: &[],
        }
    }

    /// Creates excludes with only float exclusions.
    pub const fn float_only(float: &'a [FloatCode]) -> Self {
        Self {
            int: &[],
            float,
            string: &[],
        }
    }

    /// Creates excludes with only string exclusions.
    pub const fn string_only(string: &'a [StringCode]) -> Self {
        Self {
            int: &[],
            float: &[],
            string,
        }
    }
}

/// Configures how stats are generated.
pub struct GenerateStatsOptions {
    /// Should distinct values should be counted during stats generation.
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

/// The size of each sampled run.
const SAMPLE_SIZE: u32 = 64;
/// The number of sampled runs.
///
/// # Warning
///
/// The product of SAMPLE_SIZE and SAMPLE_COUNT should be (roughly) a multiple of 1024 so that
/// fastlanes bitpacking of sampled vectors does not introduce (large amounts of) padding.
const SAMPLE_COUNT: u32 = 16;

/// Stats for the compressor.
pub trait CompressorStats: Debug + Clone {
    /// The type of the underlying source array vtable.
    type ArrayVTable: VTable;

    /// Generates stats with default options.
    fn generate(input: &<Self::ArrayVTable as VTable>::Array) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default())
    }

    /// Generates stats with provided options.
    fn generate_opts(
        input: &<Self::ArrayVTable as VTable>::Array,
        opts: GenerateStatsOptions,
    ) -> Self;

    /// Returns the underlying source array that statistics were generated from.
    fn source(&self) -> &<Self::ArrayVTable as VTable>::Array;

    /// Sample the array with default options.
    fn sample(&self, sample_size: u32, sample_count: u32) -> Self {
        self.sample_opts(sample_size, sample_count, GenerateStatsOptions::default())
    }

    /// Sample the array with provided options.
    fn sample_opts(&self, sample_size: u32, sample_count: u32, opts: GenerateStatsOptions) -> Self;
}

/// Top-level compression scheme trait.
///
/// Variants are specialized for each data type, e.g. see `IntegerScheme`, `FloatScheme`, etc.
pub trait Scheme: Debug {
    /// Type of the stats generated by the compression scheme.
    type StatsType: CompressorStats;
    /// Type of the code used to uniquely identify the compression scheme.
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
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[Self::CodeType],
    ) -> VortexResult<f64> {
        estimate_compression_ratio_with_sampling(
            self,
            compressor,
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    /// Compress the input with this scheme, yielding a new array.
    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[Self::CodeType],
    ) -> VortexResult<ArrayRef>;
}

impl<C: Copy + Eq + Hash, V: CompressorStats> PartialEq for dyn Scheme<CodeType = C, StatsType = V> {
    fn eq(&self, other: &Self) -> bool {
        self.code() == other.code()
    }
}
impl<C: Copy + Eq + Hash, V: CompressorStats> Eq for dyn Scheme<CodeType = C, StatsType = V> {}
impl<C: Copy + Eq + Hash, V: CompressorStats> Hash for dyn Scheme<CodeType = C, StatsType = V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.code().hash(state)
    }
}

fn estimate_compression_ratio_with_sampling<T: Scheme + ?Sized>(
    scheme: &T,
    btr_blocks_compressor: &BtrBlocksCompressor,
    stats: &T::StatsType,
    is_sample: bool,
    allowed_cascading: usize,
    excludes: &[T::CodeType],
) -> VortexResult<f64> {
    let sample = if is_sample {
        stats.clone()
    } else {
        // We want to sample about 1% of data
        let source_len = stats.source().len();

        // We want to sample about 1% of data, while keeping a minimal sample of 1024 values.
        let approximately_one_percent = (source_len / 100)
            / usize::try_from(SAMPLE_SIZE).vortex_expect("SAMPLE_SIZE must fit in usize");
        let sample_count = u32::max(
            u32::next_multiple_of(
                approximately_one_percent
                    .try_into()
                    .vortex_expect("sample count must fit in u32"),
                16,
            ),
            SAMPLE_COUNT,
        );

        tracing::trace!(
            "Sampling {} values out of {}",
            SAMPLE_SIZE as u64 * sample_count as u64,
            source_len
        );

        stats.sample(SAMPLE_SIZE, sample_count)
    };

    let after = scheme
        .compress(
            btr_blocks_compressor,
            &sample,
            true,
            allowed_cascading,
            excludes,
        )?
        .nbytes();
    let before = sample.source().nbytes();

    tracing::debug!(
        "estimate_compression_ratio_with_sampling(compressor={scheme:#?} is_sample={is_sample}, allowed_cascading={allowed_cascading}) = {}",
        before as f64 / after as f64
    );

    Ok(before as f64 / after as f64)
}

const MAX_CASCADE: usize = 3;

/// A compressor for a particular input type.
///
/// This trait defines the interface for type-specific compressors that can adaptively
/// choose and apply compression schemes based on data characteristics. Compressors
/// analyze input arrays, select optimal compression schemes, and handle cascading
/// compression with multiple encoding layers.
///
/// The compressor works by generating statistics on the input data, evaluating
/// available compression schemes, and selecting the one with the best compression ratio.
pub trait Compressor {
    /// The VTable type for arrays this compressor operates on.
    type ArrayVTable: VTable;
    /// The compression scheme type used by this compressor.
    type SchemeType: Scheme<StatsType = Self::StatsType> + ?Sized;
    /// The statistics type used to analyze arrays for compression.
    type StatsType: CompressorStats<ArrayVTable = Self::ArrayVTable>;

    /// Generates statistics for the given array to guide compression scheme selection.
    fn gen_stats(&self, array: &<Self::ArrayVTable as VTable>::Array) -> Self::StatsType;

    /// Returns all available compression schemes for this compressor.
    fn schemes(&self) -> &[&'static Self::SchemeType];
    /// Returns the default fallback compression scheme.
    fn default_scheme(&self) -> &'static Self::SchemeType;

    /// Selects the best compression scheme based on expected compression ratios.
    ///
    /// Evaluates all available schemes against the provided statistics and returns
    /// the one with the highest compression ratio. Falls back to the default scheme
    /// if no scheme provides compression benefits.
    #[allow(clippy::cognitive_complexity)]
    fn choose_scheme(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[<Self::SchemeType as Scheme>::CodeType],
    ) -> VortexResult<&'static Self::SchemeType> {
        let mut best_ratio = 1.0;
        let mut best_scheme: Option<&'static Self::SchemeType> = None;

        // logging helpers
        let depth = MAX_CASCADE - allowed_cascading;

        for scheme in self.schemes().iter() {
            // Skip excluded schemes
            if excludes.contains(&scheme.code()) {
                continue;
            }

            // We never choose Constant for a sample
            if is_sample && scheme.is_constant() {
                continue;
            }

            tracing::trace!(
                is_sample,
                depth,
                is_constant = scheme.is_constant(),
                ?scheme,
                "Trying compression scheme"
            );

            let ratio = scheme.expected_compression_ratio(
                compressor,
                stats,
                is_sample,
                allowed_cascading,
                excludes,
            )?;
            tracing::trace!(
                is_sample,
                depth,
                ratio,
                ?scheme,
                "Expected compression result"
            );

            if !(ratio.is_subnormal() || ratio.is_infinite() || ratio.is_nan()) {
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_scheme = Some(*scheme);
                }
            } else {
                tracing::trace!(
                    "Calculated invalid compression ratio {ratio} for scheme: {scheme:?}. Must not be sub-normal, infinite or nan."
                );
            }
        }

        tracing::trace!(depth, scheme = ?best_scheme, ratio = best_ratio, "best scheme found");

        if let Some(best) = best_scheme {
            Ok(best)
        } else {
            Ok(self.default_scheme())
        }
    }
}

/// Compresses an array using the given compressor.
///
/// Generates statistics on the input array, selects the best compression scheme,
/// and applies it. Returns the original array if compression would increase size.
pub fn compress<C: Compressor>(
    c: &C,
    compressor: &BtrBlocksCompressor,
    array: &<<C as Compressor>::ArrayVTable as VTable>::Array,
    is_sample: bool,
    allowed_cascading: usize,
    excludes: &[<C::SchemeType as Scheme>::CodeType],
) -> VortexResult<ArrayRef>
where
    <C as Compressor>::SchemeType: 'static,
{
    // Avoid compressing empty arrays.
    if array.is_empty() {
        return Ok(array.to_array());
    }

    // Generate stats on the array directly.
    let stats = c.gen_stats(array);
    let best_scheme =
        c.choose_scheme(compressor, &stats, is_sample, allowed_cascading, excludes)?;

    let output =
        best_scheme.compress(compressor, &stats, is_sample, allowed_cascading, excludes)?;
    if output.nbytes() < array.nbytes() {
        Ok(output)
    } else {
        tracing::debug!("resulting tree too large: {}", output.display_tree());
        Ok(array.to_array())
    }
}

/// Trait for compressors that can compress canonical arrays.
///
/// Provides access to configured compression schemes and the ability to
/// compress canonical arrays recursively.
pub trait CanonicalCompressor {
    /// Compresses a canonical array with the specified options.
    fn compress_canonical(
        &self,
        array: Canonical,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: Excludes,
    ) -> VortexResult<ArrayRef>;

    /// Returns the enabled integer compression schemes.
    fn int_schemes(&self) -> &[&'static dyn IntegerScheme];

    /// Returns the enabled float compression schemes.
    fn float_schemes(&self) -> &[&'static dyn FloatScheme];

    /// Returns the enabled string compression schemes.
    fn string_schemes(&self) -> &[&'static dyn StringScheme];
}

/// The main compressor type implementing BtrBlocks-inspired compression.
///
/// This compressor applies adaptive compression schemes to arrays based on their data types
/// and characteristics. It recursively compresses nested structures like structs and lists,
/// and chooses optimal compression schemes for primitive types.
///
/// The compressor works by:
/// 1. Canonicalizing input arrays to a standard representation
/// 2. Analyzing data characteristics to choose optimal compression schemes
/// 3. Recursively compressing nested structures
/// 4. Applying type-specific compression for primitives, strings, and temporal data
///
/// Use [`BtrBlocksCompressorBuilder`] to configure which compression schemes are enabled.
///
/// # Examples
///
/// ```rust
/// use vortex_btrblocks::{BtrBlocksCompressor, BtrBlocksCompressorBuilder, IntCode};
///
/// // Default compressor - all schemes allowed
/// let compressor = BtrBlocksCompressor::default();
///
/// // Exclude specific schemes using the builder
/// let compressor = BtrBlocksCompressorBuilder::new()
///     .exclude_int([IntCode::Dict])
///     .build();
/// ```
#[derive(Clone)]
pub struct BtrBlocksCompressor {
    /// Integer compressor with configured schemes.
    pub int_schemes: Vec<&'static dyn IntegerScheme>,

    /// Float compressor with configured schemes.
    pub float_schemes: Vec<&'static dyn FloatScheme>,

    /// String compressor with configured schemes.
    pub string_schemes: Vec<&'static dyn StringScheme>,
}

impl Default for BtrBlocksCompressor {
    fn default() -> Self {
        BtrBlocksCompressorBuilder::new().build()
    }
}

impl BtrBlocksCompressor {
    /// Creates a new compressor with default settings (all schemes allowed).
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns an iterator over the enabled integer compression scheme codes.
    pub fn int_codes(&self) -> impl Iterator<Item = IntCode> + '_ {
        self.int_schemes.iter().map(|s| s.code())
    }

    /// Returns an iterator over the enabled float compression scheme codes.
    pub fn float_codes(&self) -> impl Iterator<Item = FloatCode> + '_ {
        self.float_schemes.iter().map(|s| s.code())
    }

    /// Returns an iterator over the enabled string compression scheme codes.
    pub fn string_codes(&self) -> impl Iterator<Item = StringCode> + '_ {
        self.string_schemes.iter().map(|s| s.code())
    }

    /// Compresses an array using BtrBlocks-inspired compression.
    ///
    /// First canonicalizes and compacts the array, then applies optimal compression schemes.
    pub fn compress(&self, array: &dyn Array) -> VortexResult<ArrayRef> {
        // Canonicalize the array
        let canonical = array.to_canonical()?;

        // Compact it, removing any wasted space before we attempt to compress it
        let compact = canonical.compact()?;

        self.compress_canonical(compact, false, MAX_CASCADE, Excludes::none())
    }
}

impl CanonicalCompressor for BtrBlocksCompressor {
    /// Compresses a canonical array by dispatching to type-specific compressors.
    ///
    /// Recursively compresses nested structures and applies optimal schemes for each data type.
    fn compress_canonical<'a>(
        &self,
        array: Canonical,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: Excludes<'a>,
    ) -> VortexResult<ArrayRef> {
        match array {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            // TODO(aduffy): Sparse, other bool compressors.
            Canonical::Bool(bool_array) => Ok(bool_array.into_array()),
            Canonical::Primitive(primitive) => {
                if primitive.ptype().is_int() {
                    compress(
                        &IntCompressor {
                            btr_blocks_compressor: self,
                        },
                        self,
                        &primitive,
                        is_sample,
                        allowed_cascading,
                        excludes.int,
                    )
                } else {
                    compress(
                        &FloatCompressor {
                            btr_blocks_compressor: self,
                        },
                        self,
                        &primitive,
                        is_sample,
                        allowed_cascading,
                        excludes.float,
                    )
                }
            }
            Canonical::Decimal(decimal) => compress_decimal(self, &decimal),
            Canonical::Struct(struct_array) => {
                let fields = struct_array
                    .unmasked_fields()
                    .iter()
                    .map(|field| self.compress(field))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity().clone(),
                )?
                .into_array())
            }
            Canonical::List(list_view_array) => {
                // TODO(joe): We might want to write list views in the future and chose between
                // list and list view.
                let list_array = list_from_list_view(list_view_array)?;

                // Reset the offsets to remove garbage data that might prevent us from narrowing our
                // offsets (there could be a large amount of trailing garbage data that the current
                // views do not reference at all).
                let list_array = list_array.reset_offsets(true)?;

                let compressed_elems = self.compress(list_array.elements())?;

                // Note that since the type of our offsets are not encoded in our `DType`, and since
                // we guarantee above that all elements are referenced by offsets, we may narrow the
                // widths.

                let compressed_offsets = self.compress_canonical(
                    Canonical::Primitive(list_array.offsets().to_primitive().narrow()?),
                    is_sample,
                    allowed_cascading,
                    Excludes::int_only(&[IntCode::Dict]),
                )?;

                Ok(ListArray::try_new(
                    compressed_elems,
                    compressed_offsets,
                    list_array.validity().clone(),
                )?
                .into_array())
            }
            Canonical::FixedSizeList(fsl_array) => {
                let compressed_elems = self.compress(fsl_array.elements())?;

                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl_array.list_size(),
                    fsl_array.validity().clone(),
                    fsl_array.len(),
                )?
                .into_array())
            }
            Canonical::VarBinView(strings) => {
                if strings
                    .dtype()
                    .eq_ignore_nullability(&DType::Utf8(Nullability::NonNullable))
                {
                    compress(
                        &StringCompressor {
                            btr_blocks_compressor: self,
                        },
                        self,
                        &strings,
                        is_sample,
                        allowed_cascading,
                        excludes.string,
                    )
                } else {
                    // Binary arrays do not compress
                    Ok(strings.into_array())
                }
            }
            Canonical::Extension(ext_array) => {
                // We compress Timestamp-level arrays with DateTimeParts compression
                if ext_array.ext_dtype().is::<Timestamp>() {
                    if is_constant_opts(
                        ext_array.as_ref(),
                        &IsConstantOpts {
                            cost: Cost::Canonicalize,
                        },
                    )?
                    .unwrap_or_default()
                    {
                        return Ok(ConstantArray::new(
                            ext_array.as_ref().scalar_at(0)?,
                            ext_array.len(),
                        )
                        .into_array());
                    }

                    let temporal_array = TemporalArray::try_from(ext_array)?;
                    return compress_temporal(self, temporal_array);
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

    fn int_schemes(&self) -> &[&'static dyn IntegerScheme] {
        &self.int_schemes
    }

    fn float_schemes(&self) -> &[&'static dyn FloatScheme] {
        &self.float_schemes
    }

    fn string_schemes(&self) -> &[&'static dyn StringScheme] {
        &self.string_schemes
    }
}

/// Context passed through recursive compression calls.
#[derive(Debug, Clone, Copy)]
pub struct CompressorContext<'a> {
    /// Whether we're compressing a sample (for ratio estimation).
    pub is_sample: bool,
    /// Remaining cascade depth allowed.
    pub allowed_cascading: usize,
    /// Schemes to exclude at this level.
    pub excludes: Excludes<'a>,
}

impl<'a> CompressorContext<'a> {
    /// Creates a new context for top-level compression.
    pub fn new(allowed_cascading: usize) -> Self {
        Self {
            is_sample: false,
            allowed_cascading,
            excludes: Excludes::none(),
        }
    }

    /// Creates a context for sample-based compression ratio estimation.
    pub fn for_sample(allowed_cascading: usize) -> Self {
        Self {
            is_sample: true,
            allowed_cascading,
            excludes: Excludes::none(),
        }
    }

    /// Returns a new context with decremented cascade depth.
    pub fn decrement_cascade(self) -> Self {
        Self {
            allowed_cascading: self.allowed_cascading.saturating_sub(1),
            ..self
        }
    }

    /// Returns a new context with additional integer excludes.
    pub fn with_int_excludes(self, int: &'a [IntCode]) -> Self {
        Self {
            excludes: Excludes {
                int,
                ..self.excludes
            },
            ..self
        }
    }
}
