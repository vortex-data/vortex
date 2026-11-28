// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted
//! for external compatibility.

use std::hash::Hash;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::IntegerPType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::dictionary_encoded;
use crate::CascadingCompressor;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::Encoded;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::stats::IntegerErasedStats;
use crate::stats::IntegerStats;
use crate::stats::IntegerTypedStats;

/// Dictionary encoding for low-cardinality integer values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntDictScheme;

impl Scheme for IntDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Dict.id()]
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        GenerateStatsOptions {
            count_distinct_values: true,
        }
    }

    /// Children: values=0, codes=1.
    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let bit_width = data.array_as_primitive().ptype().bit_width() as u64;
        let stats = data.integer_stats(exec_ctx);
        let value_count = stats.value_count();

        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let distinct_values_count = stats.distinct_count().vortex_expect(
            "this must be present since `DictScheme` declared that we need distinct values",
        );

        // If > 50% of the values are distinct, skip dictionary scheme.
        if distinct_values_count == 0 || distinct_values_count > value_count / 2 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let array_len = u64::from(value_count) + u64::from(stats.null_count());
        let validity_bits = if stats.null_count() > 0 { array_len } else { 0 };
        let allowed_cascading = compress_ctx.allowed_cascading();

        let uncompressed_values_bits = u64::from(distinct_values_count).saturating_mul(bit_width);
        let values_bits = match allowed_cascading {
            0 => uncompressed_values_bits,
            1 if stats.erased().min_is_negative() => uncompressed_values_bits,
            1 => {
                // Only one cascade level remains: `FoRScheme` skips, but `BitPackingScheme`
                // can still pack non-negative dictionary values directly.
                let values_bw = stats.erased().max_ilog2().map_or(0, |l| u64::from(l) + 1);
                if values_bw + u64::from(super::BITPACKING_MIN_BITS_SAVED) <= bit_width {
                    let packed_values_bits = u64::from(distinct_values_count)
                        .next_multiple_of(super::FASTLANES_BLOCK_LEN)
                        .saturating_mul(values_bw);
                    uncompressed_values_bits.min(packed_values_bits)
                } else {
                    uncompressed_values_bits
                }
            }
            _ => {
                // `FoR + BitPacked` compresses `values` to the `max - min` span width (padded to
                // 1024-element FastLanes blocks plus one reference scalar) when it beats raw storage.
                let values_bw =
                    u64::from(u64::BITS - stats.erased().max_minus_min().leading_zeros());
                if values_bw + u64::from(super::BITPACKING_MIN_BITS_SAVED) <= bit_width {
                    let packed_values_bits = u64::from(distinct_values_count)
                        .next_multiple_of(super::FASTLANES_BLOCK_LEN)
                        .saturating_mul(values_bw)
                        .saturating_add(bit_width);
                    uncompressed_values_bits.min(packed_values_bits)
                } else {
                    uncompressed_values_bits
                }
            }
        };

        let histogram = (allowed_cascading > 0)
            .then(|| integer_code_bit_width_histogram(&stats, distinct_values_count).0);
        let codes_bits = super::estimate_codes_bits(
            array_len,
            u64::from(value_count),
            distinct_values_count,
            stats.run_count(),
            histogram.as_ref(),
            allowed_cascading,
        );

        let before = array_len
            .saturating_mul(bit_width)
            .saturating_add(validity_bits);
        let after = values_bits
            .saturating_add(codes_bits)
            .saturating_add(validity_bits);

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(before as f64 / after as f64))
    }

    /// Children: values=0, codes=1. The codes carry the validity.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let stats = data.integer_stats(exec_ctx);
        let dict = dictionary_encode(data.array_as_primitive(), &stats)?;
        Ok(dictionary_encoded(self, &dict))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

/// Compresses an integer array into a dictionary array according to attached stats.
///
/// Distinct values are ordered by descending occurrence count (breaking ties deterministically
/// in ascending value order) so hot values receive the smallest codes.
///
/// # Errors
///
/// Returns an error if unable to compute validity.
pub fn dictionary_encode(
    array: ArrayView<'_, Primitive>,
    stats: &IntegerStats,
) -> VortexResult<DictArray> {
    match stats.erased() {
        IntegerErasedStats::U8(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::U16(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::U32(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::U64(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::I8(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::I16(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::I32(typed) => dictionary_encode_typed(array, typed),
        IntegerErasedStats::I64(typed) => dictionary_encode_typed(array, typed),
    }
}

/// Computes the bit-width histogram of frequency-ordered dictionary codes from [`IntegerStats`].
fn integer_code_bit_width_histogram(stats: &IntegerStats, distinct_count: u32) -> ([u64; 33], u32) {
    match stats.erased() {
        IntegerErasedStats::U8(typed) => typed_code_bit_width_histogram(typed, distinct_count),
        IntegerErasedStats::U16(typed) => typed_code_bit_width_histogram(typed, distinct_count),
        IntegerErasedStats::U32(typed) => typed_code_bit_width_histogram(typed, distinct_count),
        IntegerErasedStats::U64(typed) => typed_code_bit_width_histogram(typed, distinct_count),
        IntegerErasedStats::I8(typed) => typed_code_bit_width_histogram(typed, distinct_count),
        IntegerErasedStats::I16(typed) => typed_code_bit_width_histogram(typed, distinct_count),
        IntegerErasedStats::I32(typed) => typed_code_bit_width_histogram(typed, distinct_count),
        IntegerErasedStats::I64(typed) => typed_code_bit_width_histogram(typed, distinct_count),
    }
}

/// Computes the bit-width histogram of frequency-ordered dictionary codes from typed integer stats.
fn typed_code_bit_width_histogram<T>(
    typed: &IntegerTypedStats<T>,
    distinct_count: u32,
) -> ([u64; 33], u32) {
    let distinct = typed.distinct().vortex_expect(
        "this must be present since `DictScheme` declared that we need distinct values",
    );
    super::frequency_code_bit_width_histogram(
        distinct.distinct_values().values().copied(),
        distinct_count,
    )
}

/// Encodes a typed integer array into a [`DictArray`] ordered by descending frequency, breaking
/// ties by ascending value order.
fn dictionary_encode_typed<T: IntegerPType>(
    source_array: ArrayView<'_, Primitive>,
    typed: &IntegerTypedStats<T>,
) -> VortexResult<DictArray>
where
    NativeValue<T>: Eq + Hash,
{
    let distinct = typed.distinct().vortex_expect(
        "this must be present since `DictScheme` declared that we need distinct values",
    );
    super::dictionary_encode_from_counts(
        source_array,
        distinct
            .distinct_values()
            .iter()
            .map(|(value, &count)| (value.0, count)),
    )
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Primitive;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::dict::DictArrayExt;
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::IntDictScheme;
    use super::dictionary_encode;
    use crate::builtins::dict::frequency_code_bit_width_histogram;
    use crate::builtins::dict::min_code_bit_width;
    use crate::scheme::CompressionEstimate;
    use crate::scheme::CompressorContext;
    use crate::scheme::EstimateVerdict;
    use crate::scheme::MAX_CASCADE;
    use crate::scheme::Scheme;
    use crate::scheme::SchemeExt;
    use crate::stats::ArrayAndStats;
    use crate::stats::GenerateStatsOptions;
    use crate::stats::IntegerStats;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_dict_encode_integer_stats() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let data = buffer![100i32, 200, 100, 0, 100];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(data, validity);

        let stats = IntegerStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );
        let dict_array = dictionary_encode(array.as_view(), &stats)?;
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);
        assert_eq!(
            dict_array.values().as_::<Primitive>().as_slice::<i32>(),
            &[100, 200]
        );

        let expected = PrimitiveArray::new(
            buffer![100i32, 200, 100, 100, 100],
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array()),
        )
        .into_array();
        assert_arrays_eq!(dict_array, expected, &mut ctx);
        Ok(())
    }

    /// Codes take the narrowest type that can address the dictionary, including when the
    /// dictionary exactly fills a type's range.
    #[rstest]
    #[case::fills_u8(256, PType::U8)]
    #[case::overflows_u8(257, PType::U16)]
    fn codes_are_as_narrow_as_the_dictionary_allows(
        #[case] distinct: i32,
        #[case] expected: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter(0..distinct);
        let stats = IntegerStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        let dict = dictionary_encode(array.as_view(), &stats)?;
        assert_eq!(PType::try_from(dict.codes().dtype())?, expected);
        Ok(())
    }

    #[test]
    fn test_frequency_ordered_codes_and_tie_break_determinism() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut data = Vec::with_capacity(78);
        data.push(99i64);
        data.push(-50);
        data.extend(iter::repeat_n(-1, 5));
        data.extend(iter::repeat_n(25, 20));
        data.extend(iter::repeat_n(10, 50));
        data.push(50);

        let opts = GenerateStatsOptions {
            count_distinct_values: true,
        };
        let array_a = PrimitiveArray::from_iter(data.iter().copied());
        let array_b = PrimitiveArray::from_iter(data.iter().rev().copied());
        let stats_a = IntegerStats::generate_opts(&array_a, opts, &mut ctx);
        let stats_b = IntegerStats::generate_opts(&array_b, opts, &mut ctx);

        let dict_a = dictionary_encode(array_a.as_view(), &stats_a)?;
        let dict_b = dictionary_encode(array_b.as_view(), &stats_b)?;
        assert!(dict_a.has_all_values_referenced());
        assert_eq!(
            dict_a.values().as_::<Primitive>().as_slice::<i64>(),
            &[10, 25, -1, -50, 50, 99]
        );
        assert_eq!(
            dict_a.values().as_::<Primitive>().as_slice::<i64>(),
            dict_b.values().as_::<Primitive>().as_slice::<i64>()
        );
        assert_arrays_eq!(dict_a, array_a, &mut ctx);
        assert_arrays_eq!(dict_b, array_b, &mut ctx);
        Ok(())
    }

    fn estimate_ratio(array: PrimitiveArray, compress_ctx: CompressorContext) -> f64 {
        let mut ctx = SESSION.create_execution_ctx();
        let data = ArrayAndStats::new(array.into_array(), IntDictScheme.stats_options());
        match IntDictScheme.expected_compression_ratio(&data, compress_ctx, &mut ctx) {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio)) => ratio,
            other => panic!("expected Ratio verdict, got {other:?}"),
        }
    }

    #[test]
    fn min_code_bit_width_handles_power_of_two_boundaries() {
        for (distinct, expected_bw) in [
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (256, 8),
            (257, 9),
            (65_536, 16),
            (65_537, 17),
        ] {
            assert_eq!(
                min_code_bit_width(distinct),
                expected_bw,
                "distinct={distinct}"
            );
        }
    }

    #[test]
    fn frequency_histogram_matches_sorted_reference() -> VortexResult<()> {
        let counts = [3u32, 50, 1, 20, 10, 5, 1, 20, 7, 2];
        let (hist, max_bw) = frequency_code_bit_width_histogram(
            counts.iter().copied(),
            u32::try_from(counts.len())?,
        );
        assert_eq!(max_bw, 4);

        let mut sorted = counts.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let mut expected = [0u64; 33];
        for (code, &count) in sorted.iter().enumerate() {
            let bw = if code == 0 {
                0
            } else {
                (u32::BITS - u32::try_from(code)?.leading_zeros()) as usize
            };
            expected[bw] += u64::from(count);
        }
        assert_eq!(hist, expected);
        Ok(())
    }

    #[rstest]
    #[case::two_values(2, 60.0, 64.0)]
    #[case::four_values(4, 30.0, 32.0)]
    fn power_of_two_cardinality_does_not_overestimate_code_width(
        #[case] distinct: i64,
        #[case] min_ratio: f64,
        #[case] max_ratio: f64,
    ) {
        let len = 4096usize;
        let array = PrimitiveArray::from_iter((0..len).map(|i| (i as i64 % distinct) * 1_000_000));
        let ratio = estimate_ratio(array, CompressorContext::new());
        assert!(
            ratio > min_ratio && ratio < max_ratio,
            "expected ratio in ({min_ratio}, {max_ratio}), got {ratio}"
        );
    }

    #[test]
    fn skewed_distribution_reflects_patched_bitpacking_and_values_narrowing() {
        let len = 100_000usize;
        let distinct = 2_000i64;
        // 98% of rows hit the top 16 values, 2% hit the remaining 1,984 values.
        let skewed_narrow = PrimitiveArray::from_iter((0..len).map(|i| {
            if i % 50 == 0 {
                16 + ((i / 50) as i64 % (distinct - 16))
            } else {
                (i as i64) % 16
            }
        }));
        let skewed_wide = PrimitiveArray::from_iter((0..len).map(|i| {
            let id = if i % 50 == 0 {
                16 + ((i / 50) as i64 % (distinct - 16))
            } else {
                (i as i64) % 16
            };
            id.wrapping_mul(0x0123_4567_89ab_cdef)
        }));
        let uniform_narrow = PrimitiveArray::from_iter((0..len).map(|i| i as i64 % distinct));

        let skewed_narrow_ratio = estimate_ratio(skewed_narrow, CompressorContext::new());
        let skewed_wide_ratio = estimate_ratio(skewed_wide, CompressorContext::new());
        let uniform_narrow_ratio = estimate_ratio(uniform_narrow, CompressorContext::new());

        // Patched 4-bit codes (~4.96 bpv) + narrowed 11-bit values (~0.23 bpv) => ~12.3x.
        assert!(
            skewed_narrow_ratio > 10.0,
            "expected skewed ratio > 10.0, got {skewed_narrow_ratio}"
        );
        assert!(
            skewed_narrow_ratio > skewed_wide_ratio,
            "narrow values ({skewed_narrow_ratio}) should estimate higher than wide values ({skewed_wide_ratio})"
        );
        // Uniform 2,000-value column needs 11-bit codes (~5.7x), well below skewed.
        assert!(
            uniform_narrow_ratio < 6.0,
            "expected uniform ratio < 6.0, got {uniform_narrow_ratio}"
        );
    }

    #[test]
    fn nullable_dense_array_includes_validity_cost() {
        let len = 16_384usize;
        let values = (0..len).map(|i| (i % 256) as i64).collect::<Vec<_>>();
        let validity = Validity::from_iter((0..len).map(|i| i % 10 != 0));
        let array = PrimitiveArray::new(Buffer::from(values), validity);

        // BitPacking on nullable i64 (8 bpv + 1-bit validity) achieves 65/9 = 7.222x.
        // IntDict must also pay the 1-bit validity + dictionary table, so it must stay below 7.22.
        let ratio = estimate_ratio(array, CompressorContext::new());
        assert!(
            ratio < 7.2,
            "expected nullable dense IntDict ratio < 7.2, got {ratio}"
        );
    }

    #[test]
    fn nullable_long_runs_with_scattered_nulls_do_not_over_credit_rle() {
        // 100K rows of runs of 1,000 over 4 wide values, with 10% scattered nulls.
        // `run_count` ignoring nulls is 100, but each scattered null splits a run in `codes`,
        // so RLE cannot compress to ~100 runs; codes fall back to 2-bit BitPacked + 1-bit validity
        // (~65 / 3 = 21.67x), not RLE's ~64x.
        let len = 100_000usize;
        let values: Vec<i64> = (0..len)
            .map(|i| ((i / 1_000) % 4) as i64 * 1_000_000_000)
            .collect();
        let validity = Validity::from_iter((0..len).map(|i| i % 10 != 0));
        let array = PrimitiveArray::new(Buffer::from(values), validity);

        let ratio = estimate_ratio(array, CompressorContext::new());
        assert!(
            ratio > 20.0 && ratio < 22.5,
            "expected BitPacked-bounded ratio ~21.6x, got {ratio}"
        );
    }

    #[test]
    fn bit_width_rejection_mirrors_bitpacking_argmin_order() {
        // 100 distinct values (u8 codes, max_bw = 7): 96% hit codes 0..64 (<= 6 bits), 4% hit
        // codes 64..100 (7 bits). `find_best_bit_width` compares 6 + 0.04 * 40 = 7.6 bpv vs
        // 7 + 0 = 7.0 bpv, picks 7, and `should_pack` rejects 7 (8 - 7 < 2), leaving codes as
        // uncompressed u8 (8 bpv => ~8.0x on i64, not 64 / 7.6 = 8.42x).
        let len = 100_000usize;
        let array = PrimitiveArray::from_iter((0..len).map(|i| {
            let code = if i % 25 == 0 {
                64 + ((i / 25) % 36)
            } else {
                i % 64
            };
            (code as i64).wrapping_mul(0x0123_4567_89ab_cdef)
        }));

        let ratio = estimate_ratio(array, CompressorContext::new());
        assert!(
            ratio < 8.2,
            "expected uncompressed u8 ratio (~8.0x, below patched 6-bit 8.42x), got {ratio}"
        );
    }

    #[test]
    fn finished_cascading_uses_uncompressed_narrowed_codes_and_values() {
        let len = 4096usize;
        let array = PrimitiveArray::from_iter((0..len).map(|i| (i % 2) as i64 * 1_000_000));
        let mut ctx = CompressorContext::new();
        for _ in 0..MAX_CASCADE {
            ctx = ctx.descend_with_scheme(IntDictScheme.id(), 0);
        }
        assert!(ctx.finished_cascading());

        // Without cascading, codes stay u8 (8 bpv) and values stay i64 (2 * 64 bits), so ratio ~8x.
        let ratio = estimate_ratio(array, ctx);
        assert!(
            ratio > 7.0 && ratio < 8.5,
            "expected ~8x ratio when cascading is exhausted, got {ratio}"
        );
    }

    #[test]
    fn test_expected_compression_ratio_one_cascade_level_left() {
        let len = 4096usize;
        let mut one_left = CompressorContext::new();
        for _ in 0..(MAX_CASCADE - 1) {
            one_left = one_left.descend_with_scheme(IntDictScheme.id(), 0);
        }
        assert_eq!(one_left.allowed_cascading(), 1);

        // Non-negative values: FoR is unavailable, but direct BitPacking (20-bit) still applies.
        let pos: Vec<i64> = (0..len).map(|i| 1_000_000 + (i % 1024) as i64).collect();
        let pos_full = estimate_ratio(
            PrimitiveArray::from_iter(pos.iter().copied()),
            CompressorContext::new(),
        );
        let pos_one_left = estimate_ratio(PrimitiveArray::from_iter(pos), one_left.clone());
        assert!(
            pos_one_left < pos_full && pos_one_left > 3.5,
            "expected direct-BitPacked values (~4.27x) below full FoR (~5.11x), got {pos_one_left} vs full {pos_full}"
        );

        // Negative values: neither FoR nor direct BitPacking can run, so values stay raw.
        let neg: Vec<i64> = (0..len).map(|i| -1_000_000 + (i % 1024) as i64).collect();
        let neg_one_left = estimate_ratio(PrimitiveArray::from_iter(neg), one_left.clone());
        assert!(
            neg_one_left < pos_one_left && neg_one_left < 3.0,
            "expected uncompressed negative values (~2.46x) below positive direct-packed ({pos_one_left}), got {neg_one_left}"
        );

        // Long runs over 2 values with > 65,535 rows (`u32` ends): with 1 level left, `RunEnd`
        // keeps raw `u8` codes (8 + 32 = 40 bits/run) instead of cascading (`1 + 32 = 33` bits/run).
        let run_len = 100_000usize;
        let runs: Vec<i64> = (0..run_len)
            .map(|i| ((i / 100) % 2) as i64 * 1_000_000_000)
            .collect();
        let runs_full = estimate_ratio(
            PrimitiveArray::from_iter(runs.iter().copied()),
            CompressorContext::new(),
        );
        let runs_one_left = estimate_ratio(PrimitiveArray::from_iter(runs), one_left);
        assert!(
            runs_one_left < runs_full,
            "expected 1-level-left RunEnd ({runs_one_left}) to estimate lower than full cascaded RLE ({runs_full})"
        );
    }
}
