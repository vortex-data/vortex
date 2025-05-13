pub mod dictionary;
mod stats;

use std::fmt::Debug;
use std::hash::Hash;

pub use stats::IntegerStats;
use vortex_array::arrays::{ConstantArray, PrimitiveArray, PrimitiveVTable};
use vortex_array::compress::downscale_integer_array;
use vortex_array::{ArrayExt, ArrayRef, IntoArray, ToCanonical};
use vortex_dict::DictArray;
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap};
use vortex_fastlanes::{FoRArray, bit_width_histogram, bitpack_encode, find_best_bit_width};
use vortex_runend::RunEndArray;
use vortex_runend::compress::runend_encode;
use vortex_scalar::Scalar;
use vortex_sparse::{SparseArray, SparseVTable};
use vortex_zigzag::{ZigZagArray, zigzag_encode};

use crate::integer::dictionary::dictionary_encode;
use crate::patches::compress_patches;
use crate::{
    Compressor, CompressorStats, GenerateStatsOptions, Scheme,
    estimate_compression_ratio_with_sampling,
};

pub struct IntCompressor;

impl Compressor for IntCompressor {
    type ArrayVTable = PrimitiveVTable;
    type SchemeType = dyn IntegerScheme;
    type StatsType = IntegerStats;

    fn schemes() -> &'static [&'static dyn IntegerScheme] {
        &[
            &ConstantScheme,
            &FORScheme,
            &ZigZagScheme,
            &BitPackingScheme,
            &SparseScheme,
            &DictScheme,
            &RunEndScheme,
        ]
    }

    fn default_scheme() -> &'static Self::SchemeType {
        &UncompressedScheme
    }

    fn dict_scheme_code() -> IntCode {
        DICT_SCHEME
    }
}

impl IntCompressor {
    pub fn compress_no_dict(
        array: &PrimitiveArray,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        let stats = IntegerStats::generate_opts(
            array,
            GenerateStatsOptions {
                count_distinct_values: false,
            },
        );

        let scheme = Self::choose_scheme(&stats, is_sample, allowed_cascading, excludes)?;
        let output = scheme.compress(&stats, is_sample, allowed_cascading, excludes)?;

        if output.nbytes() < array.nbytes() {
            Ok(output)
        } else {
            log::debug!("resulting tree too large: {}", output.tree_display());
            Ok(array.to_array())
        }
    }
}

pub trait IntegerScheme: Scheme<StatsType = IntegerStats, CodeType = IntCode> {}

// Auto-impl
impl<T> IntegerScheme for T where T: Scheme<StatsType = IntegerStats, CodeType = IntCode> {}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct IntCode(u8);

const UNCOMPRESSED_SCHEME: IntCode = IntCode(0);
const CONSTANT_SCHEME: IntCode = IntCode(1);
const FOR_SCHEME: IntCode = IntCode(2);
const ZIGZAG_SCHEME: IntCode = IntCode(3);
const BITPACKING_SCHEME: IntCode = IntCode(4);
const SPARSE_SCHEME: IntCode = IntCode(5);
const DICT_SCHEME: IntCode = IntCode(6);
const RUNEND_SCHEME: IntCode = IntCode(7);

#[derive(Debug, Copy, Clone)]
pub struct UncompressedScheme;

#[derive(Debug, Copy, Clone)]
pub struct ConstantScheme;

#[derive(Debug, Copy, Clone)]
pub struct FORScheme;

#[derive(Debug, Copy, Clone)]
pub struct ZigZagScheme;

#[derive(Debug, Copy, Clone)]
pub struct BitPackingScheme;

#[derive(Debug, Copy, Clone)]
pub struct SparseScheme;

#[derive(Debug, Copy, Clone)]
pub struct DictScheme;

#[derive(Debug, Copy, Clone)]
pub struct RunEndScheme;

/// Threshold for the average run length in an array before we consider run-end encoding.
const RUN_END_THRESHOLD: u32 = 4;

impl Scheme for UncompressedScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        UNCOMPRESSED_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        _stats: &IntegerStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // no compression
        Ok(1.0)
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        Ok(stats.source().to_array())
    }
}

impl Scheme for ConstantScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        CONSTANT_SCHEME
    }

    fn is_constant(&self) -> bool {
        true
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // Never yield ConstantScheme for a sample, it could be a false-positive.
        if is_sample {
            return Ok(0.0);
        }

        // Only arrays with one distinct values can be constant compressed.
        if stats.distinct_values_count != 1 {
            return Ok(0.0);
        }

        // Cannot have mix of nulls and non-nulls
        if stats.null_count > 0 && stats.value_count > 0 {
            return Ok(0.0);
        }

        Ok(stats.value_count as f64)
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        // We only use Constant encoding if the entire array is constant, never if one of
        // the child arrays yields a constant value.
        let scalar = stats
            .source()
            .as_constant()
            .vortex_expect("constant array expected");

        Ok(ConstantArray::new(scalar, stats.src.len()).into_array())
    }
}

impl Scheme for FORScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        FOR_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // Only apply if we are not at the leaf
        if allowed_cascading == 0 {
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
        let full_width: u32 = stats.src.ptype().bit_width().try_into().vortex_unwrap();
        let bw = match stats.typed.max_minus_min().checked_ilog2() {
            Some(l) => l + 1,
            // If max-min == 0, it we should use a different compression scheme
            // as we don't want to bitpack down to 0 bits.
            None => return Ok(0.0),
        };

        // If we're not saving at least 1 byte, don't bother with FOR
        if full_width - bw < 8 {
            return Ok(0.0);
        }

        Ok(full_width as f64 / bw as f64)
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        _allowed_cascading: usize,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        let for_array = FoRArray::encode(stats.src.clone())?;
        let biased = for_array.encoded().to_primitive()?;
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
        let compressed = BitPackingScheme.compress(&biased_stats, is_sample, 0, excludes)?;

        Ok(FoRArray::try_new(compressed, for_array.reference_scalar().clone())?.into_array())
    }
}

impl Scheme for ZigZagScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        ZIGZAG_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // ZigZag is only useful when we cascade it with another encoding
        if allowed_cascading == 0 {
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
        estimate_compression_ratio_with_sampling(
            self,
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        // Zigzag encode the values, then recursively compress the inner values.
        let zag = zigzag_encode(stats.src.clone())?;
        let encoded = zag.encoded().to_primitive()?;

        // ZigZag should be after Dict, RunEnd or Sparse.
        // We should only do these "container" style compressors once.
        let mut new_excludes = vec![
            ZigZagScheme.code(),
            DictScheme.code(),
            RunEndScheme.code(),
            SparseScheme.code(),
        ];
        new_excludes.extend_from_slice(excludes);

        let compressed =
            IntCompressor::compress(&encoded, is_sample, allowed_cascading - 1, &new_excludes)?;

        log::debug!("zigzag output: {}", compressed.tree_display());

        Ok(ZigZagArray::try_new(compressed)?.into_array())
    }
}

impl Scheme for BitPackingScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        BITPACKING_SCHEME
    }

    #[allow(clippy::cast_possible_truncation)]
    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
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

        estimate_compression_ratio_with_sampling(
            self,
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    #[allow(clippy::cast_possible_truncation)]
    fn compress(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        _allowed_cascading: usize,
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
        SPARSE_SCHEME
    }

    // We can avoid asserting the encoding tree instead.
    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
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
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        assert!(allowed_cascading > 0);
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
            stats.src.as_ref(),
            Some(Scalar::primitive_value(
                top_pvalue,
                top_pvalue.ptype(),
                stats.src.dtype().nullability(),
            )),
        )?;

        if let Some(sparse) = sparse_encoded.as_opt::<SparseVTable>() {
            // Compress the values
            let mut new_excludes = vec![SparseScheme.code()];
            new_excludes.extend_from_slice(excludes);

            let compressed_values = IntCompressor::compress_no_dict(
                &sparse.patches().values().to_primitive()?,
                is_sample,
                allowed_cascading - 1,
                &new_excludes,
            )?;

            let indices =
                downscale_integer_array(sparse.patches().indices().clone())?.to_primitive()?;

            let compressed_indices = IntCompressor::compress_no_dict(
                &indices,
                is_sample,
                allowed_cascading - 1,
                &new_excludes,
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
        DICT_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // Dict should not be terminal.
        if allowed_cascading == 0 {
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

        let n_runs = stats.value_count / stats.average_run_length;

        // Assume that codes will either be BitPack or RLE-BitPack
        let codes_size_bp = (codes_bw * stats.value_count) as usize;
        let codes_size_rle_bp = (codes_bw + 32) * n_runs;

        let codes_size = usize::min(codes_size_bp, codes_size_rle_bp as usize);

        let before = stats.value_count as usize * stats.source().ptype().bit_width();

        Ok(before as f64 / (values_size + codes_size) as f64)
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        assert!(allowed_cascading > 0);

        // TODO(aduffy): we can be more prescriptive: we know that codes will EITHER be
        //    RLE or FOR + BP. Cascading probably wastes some time here.

        let dict = dictionary_encode(stats)?;

        // Cascade the codes child
        let mut new_excludes = vec![DICT_SCHEME];
        new_excludes.extend_from_slice(excludes);

        let compressed_codes = IntCompressor::compress_no_dict(
            &dict.codes().to_primitive()?,
            is_sample,
            allowed_cascading - 1,
            &new_excludes,
        )?;

        Ok(DictArray::try_new(compressed_codes, dict.values().clone())?.into_array())
    }
}

impl Scheme for RunEndScheme {
    type StatsType = IntegerStats;
    type CodeType = IntCode;

    fn code(&self) -> IntCode {
        RUNEND_SCHEME
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[IntCode],
    ) -> VortexResult<f64> {
        // If the run length is below the threshold, drop it.
        if stats.average_run_length < RUN_END_THRESHOLD {
            return Ok(0.0);
        }

        if allowed_cascading == 0 {
            return Ok(0.0);
        }

        // Run compression on a sample, see how it performs.
        estimate_compression_ratio_with_sampling(
            self,
            stats,
            is_sample,
            allowed_cascading,
            excludes,
        )
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[IntCode],
    ) -> VortexResult<ArrayRef> {
        assert!(allowed_cascading > 0);

        // run-end encode the ends
        let (ends, values) = runend_encode(&stats.src)?;

        let mut new_excludes = vec![RunEndScheme.code(), DictScheme.code()];
        new_excludes.extend_from_slice(excludes);

        let ends_stats = IntegerStats::generate_opts(
            &ends,
            GenerateStatsOptions {
                count_distinct_values: false,
            },
        );
        let ends_scheme = IntCompressor::choose_scheme(
            &ends_stats,
            is_sample,
            allowed_cascading - 1,
            &new_excludes,
        )?;
        let compressed_ends =
            ends_scheme.compress(&ends_stats, is_sample, allowed_cascading - 1, &new_excludes)?;

        let compressed_values = IntCompressor::compress_no_dict(
            &values.to_primitive()?,
            is_sample,
            allowed_cascading - 1,
            &new_excludes,
        )?;

        Ok(RunEndArray::try_new(compressed_ends, compressed_values)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use log::LevelFilter;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use vortex_array::aliases::hash_set::HashSet;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::vtable::ValidityHelper;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::{Buffer, BufferMut, buffer, buffer_mut};
    use vortex_sparse::SparseEncoding;

    use crate::integer::{IntCompressor, IntegerStats, SparseScheme};
    use crate::{Compressor, CompressorStats, Scheme};

    #[test]
    fn test_empty() {
        // Make sure empty array compression does not fail
        let result = IntCompressor::compress(
            &PrimitiveArray::new(Buffer::<i32>::empty(), Validity::NonNullable),
            false,
            3,
            &[],
        )
        .unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_dict_encodable() {
        let mut codes = BufferMut::<i32>::with_capacity(65_535);
        // Write some runs of length 3 of a handful of different values. Interrupted by some
        // one-off values.

        let numbers = [0, 10, 50, 100, 1000, 3000]
            .into_iter()
            .map(|i| 1234 * i)
            .collect_vec();

        let mut rng = StdRng::seed_from_u64(1u64);
        while codes.len() < 64000 {
            let run_length = rng.next_u32() % 5;
            let value = numbers[rng.next_u32() as usize % numbers.len()];
            for _ in 0..run_length {
                codes.push(value);
            }
        }

        let primitive = codes.freeze().into_array().to_primitive().unwrap();
        let compressed = IntCompressor::compress(&primitive, false, 3, &[]).unwrap();
        log::info!("compressed values: {}", compressed.tree_display());
    }

    #[test]
    fn test_window_name() {
        env_logger::builder()
            .filter(None, LevelFilter::Debug)
            .try_init()
            .ok();

        // A test that's meant to mirror the WindowName column from ClickBench.
        let mut values = buffer_mut![-1i32; 1_000_000];
        let mut visited = HashSet::new();
        let mut rng = StdRng::seed_from_u64(1u64);
        while visited.len() < 223 {
            let random = (rng.next_u32() as usize) % 1_000_000;
            if visited.contains(&random) {
                continue;
            }
            visited.insert(random);
            // Pick 100 random values to insert.
            values[random] = 5 * (rng.next_u64() % 100) as i32;
        }

        let array = values.freeze().into_array().to_primitive().unwrap();
        let compressed = IntCompressor::compress(&array, false, 3, &[]).unwrap();
        log::info!("WindowName compressed: {}", compressed.tree_display());
    }

    #[test]
    fn sparse_with_nulls() {
        let array = PrimitiveArray::new(
            buffer![189u8, 189, 189, 0, 46],
            Validity::from_iter(vec![true, true, true, true, false]),
        );
        let compressed = SparseScheme
            .compress(&IntegerStats::generate(&array), false, 3, &[])
            .unwrap();
        assert_eq!(compressed.encoding_id(), SparseEncoding.id());
        let decoded = compressed.to_primitive().unwrap();
        let expected = [189u8, 189, 189, 0, 0];
        assert_eq!(decoded.as_slice::<u8>(), &expected);
        assert_eq!(decoded.validity(), array.validity());
    }

    #[test]
    fn sparse_mostly_nulls() {
        let array = PrimitiveArray::new(
            buffer![189u8, 189, 189, 189, 189, 189, 189, 189, 189, 0, 46],
            Validity::from_iter(vec![
                false, false, false, false, false, false, false, false, false, false, true,
            ]),
        );
        let compressed = SparseScheme
            .compress(&IntegerStats::generate(&array), false, 3, &[])
            .unwrap();
        assert_eq!(compressed.encoding_id(), SparseEncoding.id());
        let decoded = compressed.to_primitive().unwrap();
        let expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46];
        assert_eq!(decoded.as_slice::<u8>(), &expected);
        assert_eq!(decoded.validity(), array.validity());
    }
}
