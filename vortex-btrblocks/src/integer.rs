mod stats;

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Not;

use num_traits::PrimInt;
use vortex_array::array::{BooleanBufferBuilder, ConstantArray, PrimitiveArray};
use vortex_array::compute::filter;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_buffer::Buffer;
use vortex_dict::{dict_encode, DictArray};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap};
use vortex_fastlanes::{bitpack_encode, find_min_patchless_bit_width, for_compress, FoRArray};
use vortex_mask::Mask;
use vortex_runend::compress::runend_encode;
use vortex_runend::RunEndArray;
use vortex_scalar::Scalar;
use vortex_sparse::SparseArray;
use vortex_zigzag::{zigzag_encode, ZigZagArray};

use self::stats::IntegerStats;
use crate::downscale::downscale_integer_array;
use crate::{estimate_compression_ratio_with_sampling, Compressor, CompressorStats, Scheme};

pub struct IntCompressor;

impl Compressor for IntCompressor {
    type ArrayType = PrimitiveArray;
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
}

pub trait IntegerScheme: Scheme<StatsType = IntegerStats> {}

// Auto-impl
impl<T> IntegerScheme for T where T: Scheme<StatsType = IntegerStats> {}

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
const RUN_END_THRESHOLD: u32 = 2;

impl Scheme for UncompressedScheme {
    type StatsType = IntegerStats;

    fn code(&self) -> u8 {
        0
    }

    fn expected_compression_ratio(
        &self,
        _stats: &IntegerStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<f64> {
        // no compression
        Ok(1.0)
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        Ok(stats.source().clone().into_array())
    }
}

impl Scheme for ConstantScheme {
    type StatsType = IntegerStats;

    fn code(&self) -> u8 {
        1
    }

    fn is_constant(&self) -> bool {
        true
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<f64> {
        // Never yield ConstantScheme for a sample, it could be a false-positive.
        if is_sample {
            return Ok(0.0);
        }

        // Only arrays with one distinct values can be constant compressed.
        if stats.typed.distinct_value_count() > 1 {
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
        _excludes: &[u8],
    ) -> VortexResult<Array> {
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

    fn code(&self) -> u8 {
        2
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<f64> {
        // Only apply if we are not at the leaf
        if allowed_cascading == 0 {
            return Ok(0.0);
        }

        // Only apply when the min is not already zero.
        if stats.typed.min_is_zero() {
            return Ok(0.0);
        }

        // Difference between max and min
        let full_width: u32 = stats.src.ptype().bit_width().try_into().vortex_unwrap();
        // Figure out how to truncate down to the PType width.
        let padding = 64 - full_width;
        let bw = full_width + padding - stats.typed.max_minus_min().leading_zeros();

        if bw == 0 {
            // All zeros -> should use constant scheme instead.
            return Ok(0.0);
        }

        Ok(full_width as f64 / bw as f64)
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        _allowed_cascading: usize,
        excludes: &[u8],
    ) -> VortexResult<Array> {
        let for_array = for_compress(stats.src.clone())?;
        let biased = for_array.encoded().into_primitive()?;
        let biased_stats = IntegerStats::generate(&biased);

        // Immediately bitpack. If any other scheme was preferable, it would be chosen instead
        // of bitpacking.
        // NOTE: we could delegate in the future if we had another downstream codec that performs
        //  as well.
        let compressed = BitPackingScheme.compress(&biased_stats, is_sample, 0, excludes)?;

        Ok(FoRArray::try_new(compressed, for_array.reference_scalar())?.into_array())
    }
}

impl Scheme for ZigZagScheme {
    type StatsType = IntegerStats;

    fn code(&self) -> u8 {
        3
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[u8],
    ) -> VortexResult<f64> {
        // ZigZag is only useful when we cascade it with another encoding
        if allowed_cascading == 0 {
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
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        // Zigzag encode the values, then recursively compress the inner values.
        let zag = zigzag_encode(stats.src.clone())?;
        let encoded = zag.encoded().into_primitive()?;

        // ZigZag should be after Dict, RunEnd or Sparse.
        // We should only do these "container" style compressors once.
        let compressed = IntCompressor::compress(
            &encoded,
            is_sample,
            allowed_cascading - 1,
            &[
                ZigZagScheme.code(),
                DictScheme.code(),
                RunEndScheme.code(),
                SparseScheme.code(),
            ],
        )?;

        Ok(ZigZagArray::try_new(compressed)?.into_array())
    }
}

impl Scheme for BitPackingScheme {
    type StatsType = IntegerStats;

    fn code(&self) -> u8 {
        4
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<f64> {
        // BitPacking only works for non-negative values

        if stats.typed.min_is_negative() {
            return Ok(0.0);
        }

        // Estimated bit width is based on difference between max and min
        let full_width: u32 = stats.src.ptype().bit_width().try_into().vortex_unwrap();
        let padding = 64 - full_width;
        let bw = full_width + padding - stats.typed.max_minus_min().leading_zeros();

        let ratio = if bw == 0 {
            // Assume 1% will end up being patch values in the real data.
            // Assume gets compressed with 10% patches.
            0.99 * stats.value_count as f64
        } else {
            stats.src.ptype().bit_width() as f64 / bw as f64
        };

        Ok(ratio)
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        _allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        let bw = find_min_patchless_bit_width(&stats.src)?;
        let packed = bitpack_encode(stats.src.clone(), bw)?;
        Ok(packed.into_array())
    }
}

impl Scheme for SparseScheme {
    type StatsType = IntegerStats;

    fn code(&self) -> u8 {
        6
    }

    // We can avoid asserting the encoding tree instead.
    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        _is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<f64> {
        // We must have at least one level of cascading after Sparse for it to be useful.
        if allowed_cascading == 0 {
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
        let (_, top_count) = stats.typed.top_value();

        if top_count == stats.value_count {
            // top_value is the only value, should use ConstantScheme instead
            return Ok(0.0);
        }

        if top_count as f64 / stats.value_count as f64 > 0.9 {
            return Ok(stats.value_count as f64 / (stats.value_count - top_count) as f64);
        }

        Ok(0.0)
    }

    fn compress(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        assert!(allowed_cascading > 0);

        let mask = stats.src.validity().to_logical(stats.src.len())?;

        // Find the top value and all positions it occurs in.
        let (top_pvalue, _) = stats.typed.top_value();

        let top_mask = match_each_integer_ptype!(stats.src.ptype(), |$T| {
            let buffer = stats.src.buffer::<$T>();
            let top_value: $T = top_pvalue.as_primitive::<$T>().vortex_expect("top value");
            value_indices(top_value, buffer.as_ref(), &mask)
        });

        // In the case where there are no other values, immediately yield ConstantArray.
        if top_mask.all_true() {
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

        let non_top_mask = top_mask.not();
        let non_top_values = filter(stats.src.as_ref(), &non_top_mask)?.into_primitive()?;

        // Compress the values
        let values_compressed = IntCompressor::compress(
            &non_top_values,
            is_sample,
            allowed_cascading - 1,
            &[SparseScheme.code()],
        )?;

        // Compress the indices
        let indices: Buffer<u64> = match non_top_mask {
            Mask::AllTrue(count) => {
                // all true -> complete slice
                (0u64..count as u64).collect()
            }
            Mask::AllFalse(_) => {
                // empty slice
                Buffer::empty()
            }
            Mask::Values(values) => values.indices().iter().map(|v| *v as u64).collect(),
        };

        let indices = downscale_integer_array(indices.into_array())?;

        let indices_compressed = IntCompressor::compress(
            // "sparse.indices",
            &indices.into_primitive()?,
            is_sample,
            allowed_cascading - 1,
            &[SparseScheme.code()],
        )?;

        Ok(SparseArray::try_new(
            indices_compressed,
            values_compressed,
            // values_compressed,
            stats.src.len(),
            Scalar::primitive_value(
                top_pvalue,
                top_pvalue.ptype(),
                stats.src.dtype().nullability(),
            ),
        )?
        .into_array())
    }
}

// We return the top value and a mask of all positions containing the top value.
fn value_indices<T: PrimInt + Hash + Into<Scalar>>(
    top_value: T,
    values: &[T],
    validity: &Mask,
) -> Mask {
    // Find all of the positions containing the top value.
    let mut buffer = BooleanBufferBuilder::new(values.len());
    for (idx, &value) in values.iter().enumerate() {
        if top_value == value && validity.value(idx) {
            buffer.append(true);
        } else {
            buffer.append(false);
        }
    }

    Mask::from_buffer(buffer.finish())
}

impl Scheme for DictScheme {
    type StatsType = IntegerStats;

    fn code(&self) -> u8 {
        7
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[u8],
    ) -> VortexResult<f64> {
        // Dict should not be terminal.
        if allowed_cascading == 0 {
            return Ok(0.0);
        }

        // If < 50% of the values are distinct, skip dict.
        if stats.typed.distinct_value_count() / 2 > (stats.value_count as usize) {
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

    fn compress(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        _excludes: &[u8],
    ) -> VortexResult<Array> {
        assert!(allowed_cascading > 0);

        let dict = dict_encode(&stats.src.clone().into_array())?;

        // Compress the codes.
        let codes = downscale_integer_array(dict.codes())?.into_primitive()?;
        let compressed_codes = IntCompressor::compress(
            // "dict.codes",
            &codes,
            is_sample,
            allowed_cascading - 1,
            &[DictScheme.code()],
        )?;

        Ok(DictArray::try_new(compressed_codes, dict.values())?.into_array())
    }
}

impl Scheme for RunEndScheme {
    type StatsType = IntegerStats;

    fn code(&self) -> u8 {
        8
    }

    fn expected_compression_ratio(
        &self,
        stats: &IntegerStats,
        is_sample: bool,
        allowed_cascading: usize,
        excludes: &[u8],
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
        excludes: &[u8],
    ) -> VortexResult<Array> {
        assert!(allowed_cascading > 0);

        // run-end encode the ends
        let (ends, values) = runend_encode(&stats.src)?;

        let compressed_ends =
            IntCompressor::compress(&ends, is_sample, allowed_cascading - 1, excludes)?;

        // We know the values will be Primitive
        let compressed_values = IntCompressor::compress(
            // "ree.values",
            &values.into_primitive()?,
            is_sample,
            allowed_cascading - 1,
            &[RunEndScheme.code(), DictScheme.code()],
        )?;

        Ok(RunEndArray::try_new(compressed_ends, compressed_values)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use vortex_array::aliases::hash_set::HashSet;
    use vortex_array::{IntoArray, IntoArrayVariant};
    use vortex_buffer::{buffer_mut, BufferMut};

    use crate::integer::IntCompressor;
    use crate::Compressor;

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

        let primitive = codes.freeze().into_array().into_primitive().unwrap();
        let compressed = IntCompressor::compress(&primitive, false, 3, &[]).unwrap();
        println!("compressed values: {}", compressed.tree_display());
    }

    #[test]
    fn test_window_name() {
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

        // Ok, now let's compress
        let array = values.freeze().into_array().into_primitive().unwrap();
        let compressed = IntCompressor::compress(&array, false, 3, &[]).unwrap();
        println!("WindowName compressed: {}", compressed.tree_display());
    }
}
