// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for
//! external compatibility.

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
use vortex_array::dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::dictionary_encoded;
use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DescendantExclusion;
use crate::scheme::Encoded;
use crate::scheme::EstimateVerdict;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::stats::ArrayAndStats;
use crate::stats::FloatErasedStats;
use crate::stats::FloatStats;
use crate::stats::FloatTypedStats;
use crate::stats::GenerateStatsOptions;

/// Dictionary encoding for low-cardinality float values.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatDictScheme;

impl Scheme for FloatDictScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.dict"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
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

    /// Float dict codes (child 1) are compact unsigned integers that should not be
    /// dict-encoded again. Float dict values (child 0) flow through ALP into integer-land,
    /// where integer dict encoding is redundant since the values are already deduplicated at
    /// the float level.
    ///
    /// Pull rules in `vortex-btrblocks` also keep `SequenceScheme`, `FoRScheme`, `ZigZagScheme` and
    /// `DeltaScheme` off the codes (child 1), and `RunEndScheme` and the RLE schemes off the values
    /// (child 0).
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            DescendantExclusion {
                excluded: IntDictScheme.id(),
                children: ChildSelection::One(0),
            },
        ]
    }

    /// Estimates the ratio analytically from the exact distinct frequencies already computed in
    /// [`FloatStats`], like `IntDictScheme` and `DecimalDictScheme`.
    ///
    /// Sampling under-rates medium-cardinality dictionaries (e.g. thousands of distinct values in
    /// a 100K–1M row chunk) because compressing the sample charges the sampled dictionary table
    /// against far fewer rows than the full chunk.
    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let bit_width = data.array_as_primitive().ptype().bit_width() as u64;
        let stats = data.float_stats(exec_ctx);
        let value_count = stats.value_count();

        if value_count == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let distinct_values_count = stats.distinct_count().vortex_expect(
            "this must be present since `DictScheme` declared that we need distinct values",
        );

        // If > 50% of the values are distinct, skip dictionary scheme.
        if distinct_values_count > value_count / 2 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let array_len = u64::from(value_count) + u64::from(stats.null_count());
        let validity_bits = if stats.null_count() > 0 { array_len } else { 0 };
        let allowed_cascading = compress_ctx.allowed_cascading();
        let values_bits = u64::from(distinct_values_count).saturating_mul(bit_width);
        let histogram = (allowed_cascading > 0)
            .then(|| float_code_bit_width_histogram(&stats, distinct_values_count).0);
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
        let stats = data.float_stats(exec_ctx);
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

/// Computes the bit-width histogram of frequency-ordered dictionary codes from [`FloatStats`].
fn float_code_bit_width_histogram(stats: &FloatStats, distinct_count: u32) -> ([u64; 33], u32) {
    match stats.erased() {
        FloatErasedStats::F16(typed) => typed_code_bit_width_histogram(typed, distinct_count),
        FloatErasedStats::F32(typed) => typed_code_bit_width_histogram(typed, distinct_count),
        FloatErasedStats::F64(typed) => typed_code_bit_width_histogram(typed, distinct_count),
    }
}

/// Computes the bit-width histogram of frequency-ordered dictionary codes from typed float stats.
fn typed_code_bit_width_histogram<T>(
    typed: &FloatTypedStats<T>,
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

/// Compresses a floating-point array into a dictionary array according to attached stats.
///
/// Distinct values are ordered by descending occurrence count (breaking ties deterministically
/// via IEEE 754 total order) so hot values receive the smallest codes.
///
/// # Errors
///
/// Returns an error if unable to compute validity.
pub fn dictionary_encode(
    array: ArrayView<'_, Primitive>,
    stats: &FloatStats,
) -> VortexResult<DictArray> {
    match stats.erased() {
        FloatErasedStats::F16(typed) => dictionary_encode_typed(array, typed),
        FloatErasedStats::F32(typed) => dictionary_encode_typed(array, typed),
        FloatErasedStats::F64(typed) => dictionary_encode_typed(array, typed),
    }
}

/// Encodes a typed float array into a [`DictArray`] ordered by descending frequency, breaking
/// ties via IEEE 754 total order.
fn dictionary_encode_typed<T: NativePType>(
    source_array: ArrayView<'_, Primitive>,
    typed: &FloatTypedStats<T>,
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
            .map(|(&value, &count)| (value.0, count)),
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
    use vortex_array::dtype::half::f16;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::FloatDictScheme;
    use super::dictionary_encode;
    use crate::scheme::CompressionEstimate;
    use crate::scheme::CompressorContext;
    use crate::scheme::EstimateVerdict;
    use crate::scheme::MAX_CASCADE;
    use crate::scheme::Scheme;
    use crate::scheme::SchemeExt;
    use crate::stats::ArrayAndStats;
    use crate::stats::FloatStats;
    use crate::stats::GenerateStatsOptions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_float_dict_encode() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![1f32, 2f32, 2f32, 0f32, 1f32];
        let validity =
            Validity::Array(BoolArray::from_iter([true, true, true, false, true]).into_array());
        let array = PrimitiveArray::new(values, validity);

        let stats = FloatStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );
        let dict_array = dictionary_encode(array.as_view(), &stats)?;
        assert_eq!(dict_array.values().len(), 2);
        assert_eq!(dict_array.codes().len(), 5);

        let expected = PrimitiveArray::new(
            buffer![1f32, 2f32, 2f32, 1f32, 1f32],
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
        #[case] distinct: u16,
        #[case] expected: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter((0..distinct).map(|i| i as f32));
        let stats = FloatStats::generate_opts(
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
    fn test_frequency_ordered_codes() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Counts: 10.0 -> 50, 2.5 -> 20, -1.0 -> 5, 99.0 -> 1 (interleaved in non-sorted order).
        let mut data = Vec::with_capacity(76);
        data.push(99.0f64);
        data.extend(iter::repeat_n(-1.0, 5));
        data.extend(iter::repeat_n(2.5, 20));
        data.extend(iter::repeat_n(10.0, 50));

        let array = PrimitiveArray::from_iter(data);
        let stats = FloatStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        let dict = dictionary_encode(array.as_view(), &stats)?;
        assert!(dict.has_all_values_referenced());
        assert_eq!(
            dict.values().as_::<Primitive>().as_slice::<f64>(),
            &[10.0, 2.5, -1.0, 99.0]
        );
        let codes_prim = dict.codes().as_::<Primitive>();
        let codes = codes_prim.as_slice::<u8>();
        assert_eq!(codes[0], 3);
        assert!(codes[1..6].iter().all(|&code| code == 2));
        assert!(codes[6..26].iter().all(|&code| code == 1));
        assert!(codes[26..76].iter().all(|&code| code == 0));
        assert_arrays_eq!(dict, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_special_float_values_deterministic_roundtrip() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let neg_nan = f64::from_bits(0xfff8_0000_0000_0001);
        let pos_nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
        let pos_nan_b = f64::from_bits(0x7ff8_0000_0000_0002);
        let input = vec![
            pos_nan_b,
            0.0f64,
            f64::INFINITY,
            -0.0f64,
            neg_nan,
            1.0,
            pos_nan_a,
            f64::NEG_INFINITY,
            pos_nan_a,
            -0.0f64,
            f64::NEG_INFINITY,
            1.0,
            neg_nan,
            pos_nan_b,
            0.0f64,
            f64::INFINITY,
        ];
        let array_a = PrimitiveArray::from_iter(input.iter().copied());
        let array_b = PrimitiveArray::from_iter(input.iter().rev().copied());
        let opts = GenerateStatsOptions {
            count_distinct_values: true,
        };
        let stats_a = FloatStats::generate_opts(&array_a, opts, &mut ctx);
        let stats_b = FloatStats::generate_opts(&array_b, opts, &mut ctx);

        let dict_a = dictionary_encode(array_a.as_view(), &stats_a)?;
        let dict_b = dictionary_encode(array_b.as_view(), &stats_b)?;
        let bits_a: Vec<u64> = dict_a
            .values()
            .as_::<Primitive>()
            .as_slice::<f64>()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let bits_b: Vec<u64> = dict_b
            .values()
            .as_::<Primitive>()
            .as_slice::<f64>()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(bits_a, bits_b);
        assert_eq!(bits_a.len(), 8);

        let decoded = dict_a.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        let decoded_bits: Vec<u64> = decoded
            .as_slice::<f64>()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let input_bits: Vec<u64> = input.iter().map(|v| v.to_bits()).collect();
        assert_eq!(decoded_bits, input_bits);
        Ok(())
    }

    #[test]
    fn test_f16_frequency_ordering() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = [
            f16::ZERO,
            f16::NEG_ZERO,
            f16::from_f32(3.5),
            f16::from_f32(3.5),
            f16::from_f32(3.5),
        ];
        let array = PrimitiveArray::from_iter(input);
        let stats = FloatStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        let dict = dictionary_encode(array.as_view(), &stats)?;
        assert_eq!(
            dict.values().as_::<Primitive>().as_slice::<f16>()[0].to_bits(),
            f16::from_f32(3.5).to_bits()
        );
        let decoded = dict.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        let decoded_bits: Vec<u16> = decoded
            .as_slice::<f16>()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let input_bits: Vec<u16> = input.iter().map(|v| v.to_bits()).collect();
        assert_eq!(decoded_bits, input_bits);
        Ok(())
    }

    #[test]
    fn test_nulls_excluded_from_frequency_counts() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // `1.0` appears at 1 valid row and 80 null rows; `2.0` appears at 70 valid rows;
        // `999.0` appears only at null rows and must never enter the dictionary.
        let mut raw_values = Vec::with_capacity(200);
        let mut valid_bits = Vec::with_capacity(200);
        raw_values.extend(iter::repeat_n(2.0f32, 64));
        valid_bits.extend(iter::repeat_n(true, 64));
        raw_values.extend(iter::repeat_n(1.0f32, 64));
        valid_bits.extend(iter::repeat_n(false, 64));
        for idx in 0..72 {
            if idx == 0 {
                raw_values.push(1.0f32);
                valid_bits.push(true);
            } else if idx < 7 {
                raw_values.push(2.0f32);
                valid_bits.push(true);
            } else if idx < 23 {
                raw_values.push(1.0f32);
                valid_bits.push(false);
            } else {
                raw_values.push(999.0f32);
                valid_bits.push(false);
            }
        }

        let sliced_validity = Validity::from(
            BitBuffer::from_iter(iter::repeat_n(false, 5).chain(valid_bits.iter().copied()))
                .slice(5..),
        );
        let array = PrimitiveArray::new(Buffer::from(raw_values), sliced_validity);
        let stats = FloatStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        let dict = dictionary_encode(array.as_view(), &stats)?;
        assert_eq!(
            dict.values().as_::<Primitive>().as_slice::<f32>(),
            &[2.0f32, 1.0f32]
        );
        assert_arrays_eq!(dict, array, &mut ctx);
        Ok(())
    }

    fn estimate_ratio(array: PrimitiveArray, compress_ctx: CompressorContext) -> f64 {
        let mut ctx = SESSION.create_execution_ctx();
        let data = ArrayAndStats::new(array.into_array(), FloatDictScheme.stats_options());
        match FloatDictScheme.expected_compression_ratio(&data, compress_ctx, &mut ctx) {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(ratio)) => ratio,
            other => panic!("expected Ratio verdict, got {other:?}"),
        }
    }

    #[rstest]
    #[case::two_values(2, 60.0, 64.0)]
    #[case::four_values(4, 30.0, 32.0)]
    #[case::eight_values(8, 20.0, 22.0)]
    fn test_analytical_ratio_handles_power_of_two_cardinalities(
        #[case] distinct: usize,
        #[case] min_ratio: f64,
        #[case] max_ratio: f64,
    ) {
        let array = PrimitiveArray::from_iter((0..4096).map(|i| (i % distinct) as f64 * 1.25));
        let ratio = estimate_ratio(array, CompressorContext::new());
        assert!(
            ratio > min_ratio && ratio < max_ratio,
            "expected ratio in ({min_ratio}, {max_ratio}), got {ratio}"
        );
    }

    #[test]
    fn test_analytical_ratio_rewards_skewed_medium_cardinality_and_stays_conservative_on_high_cardinality()
     {
        // 100K rows, 5,000 distinct prices where 95% hit the top 16 prices (4-bit patched codes)
        // and 5% cover all 4,984 tail prices.
        let skewed = PrimitiveArray::from_iter((0..100_000usize).map(|i| {
            let cents = if i % 20 == 0 {
                16 + ((i / 20) % 4_984)
            } else {
                i % 16
            };
            cents as f64 / 100.0
        }));
        let mut ctx = SESSION.create_execution_ctx();
        let skewed_stats = FloatStats::generate_opts(
            &skewed,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );
        assert_eq!(skewed_stats.distinct_count(), Some(5_000));
        let skewed_ratio = estimate_ratio(skewed, CompressorContext::new());
        // ~6.4 bpv codes + 3.2 bpv values => ~6.65x.
        assert!(
            skewed_ratio > 6.5,
            "expected skewed medium-cardinality ratio > 6.5, got {skewed_ratio}"
        );

        // 100K rows, 45,000 distinct prices drawn uniformly (16-bit codes + 45K f64 values).
        let high_card =
            PrimitiveArray::from_iter((0..100_000usize).map(|i| (i % 45_000) as f64 / 100.0));
        let high_card_ratio = estimate_ratio(high_card, CompressorContext::new());
        assert!(
            high_card_ratio < 1.6,
            "expected high-cardinality uniform ratio < 1.6, got {high_card_ratio}"
        );
    }

    #[test]
    fn test_finished_cascading_uses_uncompressed_narrowed_codes_and_values() {
        let array = PrimitiveArray::from_iter((0..4096).map(|i| (i % 2) as f64 * 2.5));
        let mut ctx = CompressorContext::new();
        for _ in 0..MAX_CASCADE {
            ctx = ctx.descend_with_scheme(FloatDictScheme.id(), 0);
        }
        assert!(ctx.finished_cascading());

        let ratio = estimate_ratio(array, ctx);
        assert!(
            ratio > 7.5 && ratio < 8.1,
            "expected ~8x ratio when cascading is exhausted, got {ratio}"
        );
    }

    #[test]
    fn test_nullable_and_nan_leading_floats_estimate_cleanly() {
        // Leading NaN mixed with finite values estimates cleanly as 4 distinct values.
        let nan_array = PrimitiveArray::from_iter(
            (0..4096usize).map(|i| if i % 4 == 0 { f64::NAN } else { (i % 4) as f64 }),
        );
        let nan_ratio = estimate_ratio(nan_array, CompressorContext::new());
        assert!(
            nan_ratio > 30.0 && nan_ratio < 32.0,
            "expected ~32x ratio on 4-value NaN float array, got {nan_ratio}"
        );

        // An all-NaN array would have runs == len + 1 (average_run_length == 0) without bitwise
        // equality; with bitwise equality it has 1 run and estimates a high ratio.
        let all_nan = PrimitiveArray::from_iter(vec![f64::NAN; 4096]);
        let mut ctx = SESSION.create_execution_ctx();
        let all_nan_stats = FloatStats::generate_opts(
            &all_nan,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );
        assert_eq!(all_nan_stats.run_count(), 1);
        assert_eq!(all_nan_stats.average_run_length(), 4096);
        let all_nan_ratio = estimate_ratio(all_nan, CompressorContext::new());
        assert!(
            all_nan_ratio > 100.0,
            "expected high ratio on all-NaN float array, got {all_nan_ratio}"
        );

        // Nullable array must price codes across full array.len() plus the 1-bit validity bitmap.
        let len = 16_384usize;
        let values = (0..len).map(|i| (i % 256) as f64).collect::<Vec<_>>();
        let validity = Validity::from_iter((0..len).map(|i| i % 10 != 0));
        let nullable = PrimitiveArray::new(Buffer::from(values), validity);
        let nullable_ratio = estimate_ratio(nullable, CompressorContext::new());
        // Full len * 8b codes + 1b validity + 256 * 64b values => 1,064,960 / 163,840 = 6.5x.
        assert!(
            nullable_ratio > 6.4 && nullable_ratio < 6.6,
            "expected ~6.5x ratio on nullable 256-value float array, got {nullable_ratio}"
        );
    }
}
