// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod dictionary;
pub(super) mod stats;

use std::hash::Hash;
use std::hash::Hasher;

use enum_iterator::Sequence;
pub use stats::IntegerStats;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_fastlanes::FoRArray;
use vortex_fastlanes::bitpack_compress::bit_width_histogram;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::find_best_bit_width;
use vortex_runend::RunEndArray;
use vortex_runend::compress::runend_encode;
use vortex_sequence::sequence_encode;
use vortex_sparse::Sparse;
use vortex_sparse::SparseArray;
use vortex_zigzag::ZigZagArray;
use vortex_zigzag::zigzag_encode;

use self::dictionary::dictionary_encode;
use crate::BtrBlocksCompressor;
use crate::CanonicalCompressor;
use crate::Compressor;
use crate::CompressorContext;
use crate::CompressorStats;
use crate::Excludes;
use crate::GenerateStatsOptions;
use crate::Scheme;
use crate::SchemeExt;
use crate::compressor::patches::compress_patches;
use crate::compressor::rle;
use crate::compressor::rle::RLEScheme;

/// All available integer compression schemes.
pub const ALL_INT_SCHEMES: &[&dyn IntegerScheme] = &[
    &ConstantScheme,
    &FORScheme,
    &ZigZagScheme,
    &BitPackingScheme,
    &SparseScheme,
    &DictScheme,
    &RunEndScheme,
    &SequenceScheme,
    &RLE_INTEGER_SCHEME,
    #[cfg(feature = "pco")]
    &PcoScheme,
];

/// [`Compressor`] for signed and unsigned integers.
#[derive(Clone, Copy)]
pub struct IntCompressor<'a> {
    /// Reference to the parent compressor.
    pub btr_blocks_compressor: &'a dyn CanonicalCompressor,
}

impl<'a> Compressor for IntCompressor<'a> {
    type ArrayVTable = Primitive;
    type SchemeType = dyn IntegerScheme;
    type StatsType = IntegerStats;

    fn schemes(&self) -> &[&'static dyn IntegerScheme] {
        self.btr_blocks_compressor.int_schemes()
    }

    fn default_scheme(&self) -> &'static Self::SchemeType {
        &UncompressedScheme
    }

    fn gen_stats(&self, array: &<Self::ArrayVTable as VTable>::Array) -> Self::StatsType {
        if self
            .btr_blocks_compressor
            .int_schemes()
            .iter()
            .any(|s| s.code() == IntCode::Dict)
        {
            IntegerStats::generate_opts(
                array,
                GenerateStatsOptions {
                    count_distinct_values: true,
                },
            )
        } else {
            IntegerStats::generate_opts(
                array,
                GenerateStatsOptions {
                    count_distinct_values: false,
                },
            )
        }
    }
}

pub trait IntegerScheme:
    Scheme<StatsType = IntegerStats, CodeType = IntCode> + Send + Sync
{
}

// Auto-impl
impl<T> IntegerScheme for T where
    T: Scheme<StatsType = IntegerStats, CodeType = IntCode> + Send + Sync
{
}

impl PartialEq for dyn IntegerScheme {
    fn eq(&self, other: &Self) -> bool {
        self.code() == other.code()
    }
}

impl Eq for dyn IntegerScheme {}

impl Hash for dyn IntegerScheme {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.code().hash(state)
    }
}

/// Unique identifier for integer compression schemes.
///
/// NOTE: Variant order matters for tie-breaking; `For` must precede `BitPacking` to avoid unnecessary patches.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Sequence, Ord, PartialOrd)]
pub enum IntCode {
    /// No compression applied.
    Uncompressed,
    /// Constant encoding for arrays with a single distinct value.
    Constant,
    /// Frame of Reference encoding - subtracts minimum value then bitpacks.
    For,
    /// BitPacking encoding - compresses non-negative integers by reducing bit width.
    BitPacking,
    /// ZigZag encoding - transforms negative integers to positive for better bitpacking.
    ZigZag,
    /// Sparse encoding - optimizes null-dominated or single-value-dominated arrays.
    Sparse,
    /// Dictionary encoding - creates a dictionary of unique values.
    Dict,
    /// Run-end encoding - run-length encoding with end positions.
    RunEnd,
    /// Sequence encoding - detects sequential patterns.
    Sequence,
    /// RLE encoding - generic run-length encoding.
    Rle,
    /// Pco (pcodec) compression for integers.
    Pco,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]

pub struct UncompressedScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]

pub struct ConstantScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]

pub struct FORScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ZigZagScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BitPackingScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SparseScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DictScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RunEndScheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SequenceScheme;

/// Pco (pcodec) compression for integers.
#[cfg(feature = "pco")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct PcoScheme;

/// Threshold for the average run length in an array before we consider run-end encoding.
const RUN_END_THRESHOLD: u32 = 4;

/// Configuration for integer RLE compression.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntRLEConfig;

impl rle::RLEConfig for IntRLEConfig {
    type Stats = IntegerStats;
    type Code = IntCode;

    const CODE: IntCode = IntCode::Rle;

    fn compress_values(
        compressor: &BtrBlocksCompressor,
        values: &PrimitiveArray,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        compressor.compress_canonical(Canonical::Primitive(values.clone()), ctx, excludes.into())
    }
}

/// RLE scheme for integer compression.
pub const RLE_INTEGER_SCHEME: RLEScheme<IntRLEConfig> = RLEScheme::new();

impl Scheme for UncompressedScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::Uncompressed
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &BtrBlocksCompressor,
        _stats: &IntegerStats,
        _ctx: CompressorContext,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // no compression
        Ok(1.0)
    }

    fn compress(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        _ctx: CompressorContext,
        _excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        Ok(stats.source().clone().into_array())
    }
}

impl Scheme for ConstantScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::Constant
    }

    fn is_constant(&self) -> bool {
        true
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // Never yield ConstantScheme for a sample, it could be a false-positive.
        if ctx.is_sample {
            return Ok(0.0);
        }

        // Only arrays with one distinct values can be constant compressed.
        if stats.distinct_values_count > 1 {
            return Ok(0.0);
        }

        Ok(stats.value_count as f64)
    }

    fn compress(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        _ctx: CompressorContext,
        _excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        let scalar_idx =
            (0..stats.source().len()).position(|idx| stats.source().is_valid(idx).unwrap_or(false));

        match scalar_idx {
            Some(idx) => {
                let scalar = stats.source().scalar_at(idx)?;
                let const_arr = ConstantArray::new(scalar, stats.src.len()).into_array();
                if !stats.source().all_valid()? {
                    Ok(MaskedArray::try_new(const_arr, stats.src.validity().clone())?.into_array())
                } else {
                    Ok(const_arr)
                }
            }
            None => Ok(ConstantArray::new(
                Scalar::null(stats.src.dtype().clone()),
                stats.src.len(),
            )
            .into_array()),
        }
    }
}

impl Scheme for FORScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::For
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // Only apply if we are not at the leaf
        if ctx.allowed_cascading == 0 {
            return Ok(0.0);
        }

        // All-null cannot be FOR compressed.
        if stats.value_count == 0 {
            return Ok(0.0);
        }

        // Only apply when the min is not already zero.
        if stats.typed.min_is_zero() {
            return Ok(0.0);
        }

        // Difference between max and min
        let full_width: u32 = stats
            .src
            .ptype()
            .bit_width()
            .try_into()
            .vortex_expect("bit width must fit in u32");
        let for_bw = match stats.typed.max_minus_min().checked_ilog2() {
            Some(l) => l + 1,
            // If max-min == 0, it we should use a different compression scheme
            // as we don't want to bitpack down to 0 bits.
            None => return Ok(0.0),
        };

        // If BitPacking could apply (non-negative values) and FOR doesn't reduce bit width
        // compared to BitPacking, don't use FOR since it has overhead (storing reference).
        // Only skip FOR when min >= 0, otherwise BitPacking can't apply directly.
        if let Some(max_log) = stats
            .typed
            .max_ilog2()
            .filter(|_| !stats.typed.min_is_negative())
        {
            let bitpack_bw = max_log + 1;
            if for_bw >= bitpack_bw {
                return Ok(0.0);
            }
        }

        Ok(full_width as f64 / for_bw as f64)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        let for_array = FoRArray::encode(stats.src.clone())?;
        let biased = for_array.encoded().to_primitive();
        let biased_stats = IntegerStats::generate_opts(
            &biased,
            GenerateStatsOptions {
                count_distinct_values: false,
            },
        );

        // Immediately bitpack. If any other scheme was preferable, it would be chosen instead
        // of bitpacking.
        // NOTE: we could delegate in the future if we had another downstream codec that performs
        //  as well.
        let leaf_ctx = CompressorContext {
            is_sample: ctx.is_sample,
            allowed_cascading: 0,
        };
        let compressed =
            BitPackingScheme.compress(compressor, &biased_stats, leaf_ctx, excludes)?;

        let for_compressed = FoRArray::try_new(compressed, for_array.reference_scalar().clone())?;
        for_compressed
            .as_ref()
            .statistics()
            .inherit_from(for_array.as_ref().statistics());
        Ok(for_compressed.into_array())
    }
}

impl Scheme for ZigZagScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::ZigZag
    }

    fn expected_compression_ratio(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // ZigZag is only useful when we cascade it with another encoding
        if ctx.allowed_cascading == 0 {
            return Ok(0.0);
        }

        // Don't try and compress all-null arrays
        if stats.value_count == 0 {
            return Ok(0.0);
        }

        // ZigZag is only useful when there are negative values.
        if !stats.typed.min_is_negative() {
            return Ok(0.0);
        }

        // Run compression on a sample to see how it performs.
        self.estimate_compression_ratio_with_sampling(compressor, stats, ctx, excludes)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        // Zigzag encode the values, then recursively compress the inner values.
        let zag = zigzag_encode(stats.src.clone())?;
        let encoded = zag.encoded().to_primitive();

        // ZigZag should be after Dict, RunEnd or Sparse.
        // We should only do these "container" style compressors once.
        let mut new_excludes = vec![
            ZigZagScheme.code(),
            DictScheme.code(),
            RunEndScheme.code(),
            SparseScheme.code(),
        ];
        new_excludes.extend_from_slice(excludes);

        let compressed = compressor.compress_canonical(
            Canonical::Primitive(encoded),
            ctx.descend(),
            Excludes::int_only(&new_excludes),
        )?;

        tracing::debug!("zigzag output: {}", compressed.encoding_id());

        Ok(ZigZagArray::try_new(compressed)?.into_array())
    }
}

impl Scheme for BitPackingScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::BitPacking
    }

    fn expected_compression_ratio(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // BitPacking only works for non-negative values
        if stats.typed.min_is_negative() {
            return Ok(0.0);
        }

        // Don't compress all-null arrays
        if stats.value_count == 0 {
            return Ok(0.0);
        }

        self.estimate_compression_ratio_with_sampling(compressor, stats, ctx, excludes)
    }

    fn compress(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        _ctx: CompressorContext,
        _excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        let histogram = bit_width_histogram(stats.source())?;
        let bw = find_best_bit_width(stats.source().ptype(), &histogram)?;
        // If best bw is determined to be the current bit-width, return the original array.
        if bw as usize == stats.source().ptype().bit_width() {
            return Ok(stats.source().clone().into_array());
        }
        let mut packed = bitpack_encode(stats.source(), bw, Some(&histogram))?;

        let patches = packed.patches().map(compress_patches).transpose()?;
        packed.replace_patches(patches);

        Ok(packed.into_array())
    }
}

impl Scheme for SparseScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::Sparse
    }

    // We can avoid asserting the encoding tree instead.
    fn expected_compression_ratio(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // Only use `SparseScheme` if we can cascade.
        if ctx.allowed_cascading == 0 {
            return Ok(0.0);
        }

        if stats.value_count == 0 {
            // All nulls should use ConstantScheme
            return Ok(0.0);
        }

        // If the majority is null, will compress well.
        if stats.null_count as f64 / stats.src.len() as f64 > 0.9 {
            return Ok(stats.src.len() as f64 / stats.value_count as f64);
        }

        // See if the top value accounts for >= 90% of the set values.
        let (_, top_count) = stats.typed.top_value_and_count();

        if top_count == stats.value_count {
            // top_value is the only value, should use ConstantScheme instead
            return Ok(0.0);
        }

        let freq = top_count as f64 / stats.value_count as f64;
        if freq >= 0.9 {
            // We only store the positions of the non-top values.
            return Ok(stats.value_count as f64 / (stats.value_count - top_count) as f64);
        }

        Ok(0.0)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        assert!(ctx.allowed_cascading > 0);
        let (top_pvalue, top_count) = stats.typed.top_value_and_count();
        if top_count as usize == stats.src.len() {
            // top_value is the only value, use ConstantScheme
            return Ok(ConstantArray::new(
                Scalar::primitive_value(
                    top_pvalue,
                    top_pvalue.ptype(),
                    stats.src.dtype().nullability(),
                ),
                stats.src.len(),
            )
            .into_array());
        }

        let sparse_encoded = SparseArray::encode(
            &stats.src.clone().into_array(),
            Some(Scalar::primitive_value(
                top_pvalue,
                top_pvalue.ptype(),
                stats.src.dtype().nullability(),
            )),
        )?;

        if let Some(sparse) = sparse_encoded.as_opt::<Sparse>() {
            // Compress the values
            let mut new_excludes = vec![SparseScheme.code(), IntCode::Dict];
            new_excludes.extend_from_slice(excludes);

            let compressed_values = compressor.compress_canonical(
                Canonical::Primitive(sparse.patches().values().to_primitive()),
                ctx.descend(),
                Excludes::int_only(&new_excludes),
            )?;

            let indices = sparse.patches().indices().to_primitive().narrow()?;

            let compressed_indices = compressor.compress_canonical(
                Canonical::Primitive(indices),
                ctx.descend(),
                Excludes::int_only(&new_excludes),
            )?;

            SparseArray::try_new(
                compressed_indices,
                compressed_values,
                sparse.len(),
                sparse.fill_scalar().clone(),
            )
            .map(|a| a.into_array())
        } else {
            Ok(sparse_encoded)
        }
    }
}

impl Scheme for DictScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::Dict
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // Dict should not be terminal.
        if ctx.allowed_cascading == 0 {
            return Ok(0.0);
        }

        if stats.value_count == 0 {
            return Ok(0.0);
        }

        // If > 50% of the values are distinct, skip dict.
        if stats.distinct_values_count > stats.value_count / 2 {
            return Ok(0.0);
        }

        // Ignore nulls encoding for the estimate. We only focus on values.
        let values_size = stats.source().ptype().bit_width() * stats.distinct_values_count as usize;

        // Assume codes are compressed RLE + BitPacking.
        let codes_bw = usize::BITS - stats.distinct_values_count.leading_zeros();

        let n_runs = (stats.value_count / stats.average_run_length) as usize;

        // Assume that codes will either be BitPack or RLE-BitPack
        let codes_size_bp = (codes_bw * stats.value_count) as usize;
        let codes_size_rle_bp = usize::checked_mul((codes_bw + 32) as usize, n_runs);

        let codes_size = usize::min(codes_size_bp, codes_size_rle_bp.unwrap_or(usize::MAX));

        let before = stats.value_count as usize * stats.source().ptype().bit_width();

        Ok(before as f64 / (values_size + codes_size) as f64)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        assert!(ctx.allowed_cascading > 0);

        // TODO(aduffy): we can be more prescriptive: we know that codes will EITHER be
        //    RLE or FOR + BP. Cascading probably wastes some time here.

        let dict = dictionary_encode(stats);

        // Cascade the codes child
        // Don't allow SequenceArray as the codes child as it merely adds extra indirection without actually compressing data.
        let mut new_excludes = vec![IntCode::Dict, IntCode::Sequence];
        new_excludes.extend_from_slice(excludes);

        let compressed_codes = compressor.compress_canonical(
            Canonical::Primitive(dict.codes().to_primitive().narrow()?),
            ctx.descend(),
            Excludes::int_only(&new_excludes),
        )?;

        // SAFETY: compressing codes does not change their values
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, dict.values().clone())
                    .set_all_values_referenced(dict.has_all_values_referenced())
                    .into_array(),
            )
        }
    }
}

impl Scheme for RunEndScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::RunEnd
    }

    fn expected_compression_ratio(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // If the run length is below the threshold, drop it.
        if stats.average_run_length < RUN_END_THRESHOLD {
            return Ok(0.0);
        }

        if ctx.allowed_cascading == 0 {
            return Ok(0.0);
        }

        // Run compression on a sample, see how it performs.
        self.estimate_compression_ratio_with_sampling(compressor, stats, ctx, excludes)
    }

    fn compress(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &IntegerStats,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        assert!(ctx.allowed_cascading > 0);

        // run-end encode the ends
        let (ends, values) = runend_encode(&stats.src);

        let mut new_excludes = vec![RunEndScheme.code(), DictScheme.code()];
        new_excludes.extend_from_slice(excludes);

        let compressed_ends = compressor.compress_canonical(
            Canonical::Primitive(ends.to_primitive()),
            ctx.descend(),
            Excludes::int_only(&new_excludes),
        )?;

        let compressed_values = compressor.compress_canonical(
            Canonical::Primitive(values.to_primitive()),
            ctx.descend(),
            Excludes::int_only(&new_excludes),
        )?;

        // SAFETY: compression doesn't affect invariants
        unsafe {
            Ok(
                RunEndArray::new_unchecked(compressed_ends, compressed_values, 0, stats.src.len())
                    .into_array(),
            )
        }
    }
}

impl Scheme for SequenceScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> Self::CodeType {
        IntCode::Sequence
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        _ctx: CompressorContext,
        _excludes: &[Self::CodeType],
    ) -> VortexResult<f64> {
        if stats.null_count > 0 {
            return Ok(0.0);
        }

        // If the distinct_values_count was computed (!= u32::MAX)
        // Then all values in a sequence must be unique.
        if stats.distinct_values_count != u32::MAX
            && stats.distinct_values_count as usize != stats.src.len()
        {
            return Ok(0.0);
        }

        // Since two values are required to store base and multiplier the
        // compression ratio is divided by 2.
        Ok(sequence_encode(&stats.src)?
            .map(|_| stats.src.len() as f64 / 2.0)
            .unwrap_or(0.0))
    }

    fn compress(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        _ctx: CompressorContext,
        _excludes: &[Self::CodeType],
    ) -> VortexResult<ArrayRef> {
        if stats.null_count > 0 {
            vortex_bail!("sequence encoding does not support nulls");
        }
        sequence_encode(&stats.src)?.ok_or_else(|| vortex_err!("cannot sequence encode array"))
    }
}

#[cfg(feature = "pco")]
impl Scheme for PcoScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        IntCode::Pco
    }

    fn expected_compression_ratio(
        &self,
        compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        ctx: CompressorContext,
        excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // Pco does not support I8 or U8.
        if matches!(
            stats.src.ptype(),
            vortex_array::dtype::PType::I8 | vortex_array::dtype::PType::U8
        ) {
            return Ok(0.0);
        }

        self.estimate_compression_ratio_with_sampling(compressor, stats, ctx, excludes)
    }

    fn compress(
        &self,
        _compressor: &BtrBlocksCompressor,
        stats: &Self::StatsType,
        _ctx: CompressorContext,
        _excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        Ok(vortex_pco::PcoArray::from_primitive(
            stats.source(),
            pco::DEFAULT_COMPRESSION_LEVEL,
            8192,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use itertools::Itertools;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_sequence::Sequence;
    use vortex_sparse::Sparse;

    use super::IntegerStats;
    use super::RLE_INTEGER_SCHEME;
    use super::SequenceScheme;
    use super::SparseScheme;
    use crate::BtrBlocksCompressor;
    use crate::CompressorContext;
    use crate::CompressorExt;
    use crate::CompressorStats;
    use crate::Scheme;

    #[test]
    fn test_empty() -> VortexResult<()> {
        // Make sure empty array compression does not fail
        let btr = BtrBlocksCompressor::default();
        let result = btr.integer_compressor().compress(
            &btr,
            &PrimitiveArray::new(Buffer::<i32>::empty(), Validity::NonNullable),
            CompressorContext::default(),
            &[],
        )?;

        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_dict_encodable() -> VortexResult<()> {
        let mut codes = BufferMut::<i32>::with_capacity(65_535);
        // Write some runs of length 3 of a handful of different values. Interrupted by some
        // one-off values.

        let numbers = [0, 10, 50, 100, 1000, 3000]
            .into_iter()
            .map(|i| 12340 * i) // must be big enough to not prefer fastlanes.bitpacked
            .collect_vec();

        let mut rng = StdRng::seed_from_u64(1u64);
        while codes.len() < 64000 {
            let run_length = rng.next_u32() % 5;
            let value = numbers[rng.next_u32() as usize % numbers.len()];
            for _ in 0..run_length {
                codes.push(value);
            }
        }

        let primitive = codes.freeze().into_array().to_primitive();
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.integer_compressor().compress(
            &btr,
            &primitive,
            CompressorContext::default(),
            &[],
        )?;
        assert!(compressed.is::<Dict>());
        Ok(())
    }

    #[test]
    fn sparse_with_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![189u8, 189, 189, 0, 46],
            Validity::from_iter(vec![true, true, true, true, false]),
        );
        let btr = BtrBlocksCompressor::default();
        let compressed = SparseScheme.compress(
            &btr,
            &IntegerStats::generate(&array),
            CompressorContext::default(),
            &[],
        )?;
        assert!(compressed.is::<Sparse>());
        let decoded = compressed.clone();
        let expected =
            PrimitiveArray::new(buffer![189u8, 189, 189, 0, 0], array.validity().clone())
                .into_array();
        assert_arrays_eq!(decoded.as_ref(), expected.as_ref());
        Ok(())
    }

    #[test]
    fn sparse_mostly_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![189u8, 189, 189, 189, 189, 189, 189, 189, 189, 0, 46],
            Validity::from_iter(vec![
                false, false, false, false, false, false, false, false, false, false, true,
            ]),
        );
        let btr = BtrBlocksCompressor::default();
        let compressed = SparseScheme.compress(
            &btr,
            &IntegerStats::generate(&array),
            CompressorContext::default(),
            &[],
        )?;
        assert!(compressed.is::<Sparse>());
        let decoded = compressed.clone();
        let expected = PrimitiveArray::new(
            buffer![0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46],
            array.validity().clone(),
        )
        .into_array();
        assert_arrays_eq!(decoded.as_ref(), expected.as_ref());
        Ok(())
    }

    #[test]
    fn nullable_sequence() -> VortexResult<()> {
        let values = (0i32..20).step_by(7).collect_vec();
        let array = PrimitiveArray::from_option_iter(values.clone().into_iter().map(Some));
        let btr = BtrBlocksCompressor::default();
        let compressed = SequenceScheme.compress(
            &btr,
            &IntegerStats::generate(&array),
            CompressorContext::default(),
            &[],
        )?;
        assert!(compressed.is::<Sequence>());
        let decoded = compressed;
        let expected = PrimitiveArray::from_option_iter(values.into_iter().map(Some)).into_array();
        assert_arrays_eq!(decoded.as_ref(), expected.as_ref());
        Ok(())
    }

    #[test]
    fn test_rle_compression() -> VortexResult<()> {
        let mut values = Vec::new();
        values.extend(iter::repeat_n(42i32, 100));
        values.extend(iter::repeat_n(123i32, 200));
        values.extend(iter::repeat_n(987i32, 150));

        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = RLE_INTEGER_SCHEME.compress(
            &btr,
            &IntegerStats::generate(&array),
            CompressorContext::default(),
            &[],
        )?;

        let decoded = compressed;
        let expected = Buffer::copy_from(&values).into_array();
        assert_arrays_eq!(decoded.as_ref(), expected.as_ref());
        Ok(())
    }

    #[test_with::env(CI)]
    #[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
    fn compress_large_int() -> VortexResult<()> {
        const NUM_LISTS: usize = 10_000;
        const ELEMENTS_PER_LIST: usize = 5_000;

        let prim = (0..NUM_LISTS)
            .flat_map(|list_idx| {
                (0..ELEMENTS_PER_LIST).map(move |elem_idx| (list_idx * 1000 + elem_idx) as f64)
            })
            .collect::<PrimitiveArray>()
            .into_array();

        let btr = BtrBlocksCompressor::default();
        drop(btr.compress(&prim)?);

        Ok(())
    }
}

/// Tests to verify that each integer compression scheme produces the expected encoding.
#[cfg(test)]
mod scheme_selection_tests {
    use std::iter;

    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPacked;
    use vortex_fastlanes::FoR;
    use vortex_fastlanes::RLE;
    use vortex_runend::RunEnd;
    use vortex_sequence::Sequence;
    use vortex_sparse::Sparse;

    use crate::BtrBlocksCompressor;
    use crate::CompressorContext;
    use crate::CompressorExt;
    use crate::IntCode;

    #[test]
    fn test_constant_compressed() -> VortexResult<()> {
        let values: Vec<i32> = iter::repeat_n(42, 100).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.integer_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn test_for_compressed() -> VortexResult<()> {
        let values: Vec<i32> = (0..1000).map(|i| 1_000_000 + ((i * 37) % 100)).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.integer_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<FoR>());
        Ok(())
    }

    #[test]
    fn test_bitpacking_compressed() -> VortexResult<()> {
        let values: Vec<u32> = (0..1000).map(|i| i % 16).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.integer_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<BitPacked>());
        Ok(())
    }

    #[test]
    fn test_sparse_compressed() -> VortexResult<()> {
        let mut values: Vec<i32> = Vec::new();
        for i in 0..1000 {
            if i % 20 == 0 {
                values.push(2_000_000 + (i * 7) % 1000);
            } else {
                values.push(1_000_000);
            }
        }
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.integer_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<Sparse>());
        Ok(())
    }

    #[test]
    fn test_dict_compressed() -> VortexResult<()> {
        let mut codes = Vec::with_capacity(65_535);
        let numbers: Vec<i32> = [0, 10, 50, 100, 1000, 3000]
            .into_iter()
            .map(|i| 12340 * i) // must be big enough to not prefer fastlanes.bitpacked
            .collect();

        let mut rng = StdRng::seed_from_u64(1u64);
        while codes.len() < 64000 {
            let run_length = rng.next_u32() % 5;
            let value = numbers[rng.next_u32() as usize % numbers.len()];
            for _ in 0..run_length {
                codes.push(value);
            }
        }

        let array = PrimitiveArray::new(Buffer::copy_from(&codes), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.integer_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<Dict>());
        Ok(())
    }

    #[test]
    fn test_runend_compressed() -> VortexResult<()> {
        let mut values: Vec<i32> = Vec::new();
        for i in 0..100 {
            values.extend(iter::repeat_n((i32::MAX - 50).wrapping_add(i), 10));
        }
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.integer_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<RunEnd>());
        Ok(())
    }

    #[test]
    fn test_sequence_compressed() -> VortexResult<()> {
        let values: Vec<i32> = (0..1000).map(|i| i * 7).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed =
            btr.integer_compressor()
                .compress(&btr, &array, CompressorContext::default(), &[])?;
        assert!(compressed.is::<Sequence>());
        Ok(())
    }

    #[test]
    fn test_rle_compressed() -> VortexResult<()> {
        let mut values: Vec<i32> = Vec::new();
        for i in 0..1024 {
            values.extend(iter::repeat_n(i, 10));
        }
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let btr = BtrBlocksCompressor::default();
        let compressed = btr.integer_compressor().compress(
            &btr,
            &array,
            CompressorContext::default(),
            &[IntCode::RunEnd],
        )?;
        assert!(compressed.is::<RLE>());
        Ok(())
    }
}
