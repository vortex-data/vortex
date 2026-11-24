// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float-specific dictionary encoding implementation.
//!
//! Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for
//! external compatibility.

use std::hash::Hash;

use rustc_hash::FxBuildHasher;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::CascadingCompressor;
use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::DeferredEstimate;
use crate::scheme::DescendantExclusion;
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

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let stats = data.float_stats(exec_ctx);

        if stats.value_count() == 0 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let distinct_values_count = stats.distinct_count().vortex_expect(
            "this must be present since `DictScheme` declared that we need distinct values",
        );

        // If > 50% of the values are distinct, skip dictionary scheme.
        if distinct_values_count > stats.value_count() / 2 {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Let sampling determine the expected ratio.
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let stats = data.float_stats(exec_ctx);
        let dict = dictionary_encode(data.array_as_primitive(), &stats)?;

        let has_all_values_referenced = dict.has_all_values_referenced();

        // Values = child 0.
        let compressed_values =
            compressor.compress_child(dict.values(), &compress_ctx, self.id(), 0, exec_ctx)?;

        // Codes = child 1. `dictionary_encode` already sizes the codes from the dictionary
        // length, so there is nothing left to narrow here.
        let compressed_codes =
            compressor.compress_child(dict.codes(), &compress_ctx, self.id(), 1, exec_ctx)?;

        // SAFETY: compressing codes or values does not alter the invariants.
        unsafe {
            Ok(
                DictArray::new_unchecked(compressed_codes, compressed_values)
                    .set_all_values_referenced(has_all_values_referenced)
                    .into_array(),
            )
        }
    }
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

/// Encodes a typed float array into a [`DictArray`] with frequency-ordered values and codes.
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

    let codes_validity = source_array.validity()?;
    let values_validity = match codes_validity {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    let mut entries: Vec<(T, u32)> = distinct
        .distinct_values()
        .iter()
        .map(|(&value, &count)| (value.0, count))
        .collect();
    entries.sort_unstable_by(|&(lhs_val, lhs_count), &(rhs_val, rhs_count)| {
        rhs_count
            .cmp(&lhs_count)
            .then_with(|| lhs_val.total_compare(rhs_val))
    });
    let values: Buffer<T> = entries.into_iter().map(|(value, _)| value).collect();

    // The largest code is one less than the dictionary length.
    let dict_len = values.len();
    let source_slice = source_array.as_slice::<T>();
    let codes = if dict_len <= u8::MAX as usize + 1 {
        let buf = encode_codes::<T, u8>(&values, source_slice);
        PrimitiveArray::new(buf, codes_validity).into_array()
    } else if dict_len <= u16::MAX as usize + 1 {
        let buf = encode_codes::<T, u16>(&values, source_slice);
        PrimitiveArray::new(buf, codes_validity).into_array()
    } else {
        let buf = encode_codes::<T, u32>(&values, source_slice);
        PrimitiveArray::new(buf, codes_validity).into_array()
    };

    let values = PrimitiveArray::new(values, values_validity).into_array();
    // SAFETY: `encode_codes` emits codes in `0..values.len()` for every valid entry
    // and preserves the exact distinct value set computed by `FloatStats`.
    Ok(unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) })
}

/// Maps `values` to their dictionary codes in `distinct`.
fn encode_codes<T: NativePType, I: UnsignedPType>(distinct: &[T], values: &[T]) -> Buffer<I>
where
    NativeValue<T>: Eq + Hash,
{
    let mut codes = HashMap::<NativeValue<T>, I, FxBuildHasher>::with_capacity_and_hasher(
        distinct.len(),
        FxBuildHasher,
    );
    for (code, &value) in distinct.iter().enumerate() {
        codes.insert(
            NativeValue(value),
            I::from(code).vortex_expect("caller ensures dictionary length fits in code type"),
        );
    }

    let mut output = BufferMut::with_capacity(values.len());
    for &value in values {
        // Any code lookups which fail are for nulls, so their value does not matter.
        // SAFETY: `output` has capacity `values.len()`.
        unsafe {
            output.push_unchecked(codes.get(&NativeValue(value)).copied().unwrap_or_default());
        }
    }

    output.freeze()
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

    use super::dictionary_encode;
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
        let array = PrimitiveArray::from_iter(input.iter().copied());
        let stats = FloatStats::generate_opts(
            &array,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        let dict_a = dictionary_encode(array.as_view(), &stats)?;
        let dict_b = dictionary_encode(array.as_view(), &stats)?;
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
}
