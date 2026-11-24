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
use vortex_array::arrays::Bool;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::half::f16;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::for_each_mask_word;
use vortex_compute::lane_kernels::low_bits_mask;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

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
        let array = data.array_as_primitive();
        let validity_mask = array.validity()?.execute_mask(array.len(), exec_ctx)?;
        let dict = dictionary_encode_with_mask(array, &stats, Some(&validity_mask))?;

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

/// Maps a floating-point type to its unsigned bit pattern for bit-exact hashing and ordering.
trait FloatBits: NativePType {
    /// Unsigned integer with the same bit width as `Self`.
    type Bits: Copy + Eq + Hash + Ord;

    /// Returns the IEEE 754 bit pattern of this value.
    fn to_bits(self) -> Self::Bits;
}

impl FloatBits for f16 {
    type Bits = u16;

    #[inline]
    fn to_bits(self) -> u16 {
        f16::to_bits(self)
    }
}

impl FloatBits for f32 {
    type Bits = u32;

    #[inline]
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }
}

impl FloatBits for f64 {
    type Bits = u64;

    #[inline]
    fn to_bits(self) -> u64 {
        f64::to_bits(self)
    }
}

/// Unsigned integer code width (`u8`, `u16`, or `u32`) for float dictionary codes.
trait DictCode: NativePType {
    /// Zero code, also used for null positions.
    const ZERO: Self;

    /// Converts a dictionary index into a code value.
    fn from_index(index: usize) -> Self;

    /// Converts a code value back into a dictionary index.
    fn into_index(self) -> usize;

    /// Allocates a buffer of `len` zero codes.
    fn zeroed_buffer(len: usize) -> Buffer<Self>;
}

impl DictCode for u8 {
    const ZERO: Self = 0;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "caller ensures dictionary length fits in u8"
    )]
    #[inline]
    fn from_index(index: usize) -> Self {
        index as u8
    }

    #[inline]
    fn into_index(self) -> usize {
        usize::from(self)
    }

    fn zeroed_buffer(len: usize) -> Buffer<Self> {
        Buffer::zeroed(len)
    }
}

impl DictCode for u16 {
    const ZERO: Self = 0;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "caller ensures dictionary length fits in u16"
    )]
    #[inline]
    fn from_index(index: usize) -> Self {
        index as u16
    }

    #[inline]
    fn into_index(self) -> usize {
        usize::from(self)
    }

    fn zeroed_buffer(len: usize) -> Buffer<Self> {
        Buffer::zeroed(len)
    }
}

impl DictCode for u32 {
    const ZERO: Self = 0;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "caller ensures dictionary length fits in u32"
    )]
    #[inline]
    fn from_index(index: usize) -> Self {
        index as u32
    }

    #[inline]
    fn into_index(self) -> usize {
        self as usize
    }

    fn zeroed_buffer(len: usize) -> Buffer<Self> {
        Buffer::zeroed(len)
    }
}

/// Compresses a floating-point array into a dictionary array according to attached stats.
///
/// Distinct values are ordered by descending occurrence count (breaking ties deterministically
/// via IEEE 754 total order and bit representation) so hot values receive the smallest codes.
///
/// # Errors
///
/// Returns an error if unable to compute validity.
pub fn dictionary_encode(
    array: ArrayView<'_, Primitive>,
    stats: &FloatStats,
) -> VortexResult<DictArray> {
    let validity_mask = match array.validity()? {
        Validity::NonNullable | Validity::AllValid => Some(Mask::new_true(array.len())),
        Validity::AllInvalid => Some(Mask::new_false(array.len())),
        Validity::Array(validity) => validity
            .as_opt::<Bool>()
            .map(|bool_array| Mask::from_buffer(bool_array.to_bit_buffer())),
    };
    dictionary_encode_with_mask(array, stats, validity_mask.as_ref())
}

/// Dictionary-encodes `array` using an optional pre-executed validity mask for frequency counting.
fn dictionary_encode_with_mask(
    array: ArrayView<'_, Primitive>,
    stats: &FloatStats,
    validity_mask: Option<&Mask>,
) -> VortexResult<DictArray> {
    match stats.erased() {
        FloatErasedStats::F16(typed) => dictionary_encode_typed(array, typed, validity_mask),
        FloatErasedStats::F32(typed) => dictionary_encode_typed(array, typed, validity_mask),
        FloatErasedStats::F64(typed) => dictionary_encode_typed(array, typed, validity_mask),
    }
}

/// Encodes a typed float array into a [`DictArray`] with frequency-ordered values and codes.
fn dictionary_encode_typed<T: FloatBits>(
    source_array: ArrayView<'_, Primitive>,
    typed: &FloatTypedStats<T>,
    validity_mask: Option<&Mask>,
) -> VortexResult<DictArray> {
    let distinct = typed.distinct().vortex_expect(
        "this must be present since `DictScheme` declared that we need distinct values",
    );

    let codes_validity = source_array.validity()?;
    let values_validity = match codes_validity {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    // The largest code is one less than the dictionary length.
    let distinct_values = distinct.distinct_values();
    let dict_len = distinct_values.len();
    let source_slice = source_array.as_slice::<T>();
    let (values, codes) = if dict_len <= u8::MAX as usize + 1 {
        let (values, buf) =
            encode_frequency_sorted::<T, u8>(distinct_values, source_slice, validity_mask);
        (
            values,
            PrimitiveArray::new(buf, codes_validity).into_array(),
        )
    } else if dict_len <= u16::MAX as usize + 1 {
        let (values, buf) =
            encode_frequency_sorted::<T, u16>(distinct_values, source_slice, validity_mask);
        (
            values,
            PrimitiveArray::new(buf, codes_validity).into_array(),
        )
    } else {
        let (values, buf) =
            encode_frequency_sorted::<T, u32>(distinct_values, source_slice, validity_mask);
        (
            values,
            PrimitiveArray::new(buf, codes_validity).into_array(),
        )
    };

    let values = PrimitiveArray::new(values, values_validity).into_array();
    // SAFETY: `encode_frequency_sorted` emits codes in `0..values.len()` for every valid entry
    // and preserves the exact distinct value set computed by `FloatStats`.
    Ok(unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) })
}

/// Assigns codes in descending frequency order in a single hash pass plus an in-place remap.
///
/// Sorting the dictionary by frequency ensures the hottest values get codes `0, 1, ...` so
/// downstream bit-packing chooses a narrow width and spills rare codes into patches.
fn encode_frequency_sorted<T: FloatBits, I: DictCode>(
    distinct: &HashSet<NativeValue<T>, FxBuildHasher>,
    values: &[T],
    validity_mask: Option<&Mask>,
) -> (Buffer<T>, Buffer<I>) {
    let dict_len = distinct.len();
    if dict_len == 0 {
        return (Buffer::empty(), I::zeroed_buffer(values.len()));
    }
    if dict_len == 1 {
        let only = distinct
            .iter()
            .next()
            .vortex_expect("single-entry distinct set is non-empty")
            .0;
        return (Buffer::from_iter([only]), I::zeroed_buffer(values.len()));
    }

    let mut temp_values = Vec::with_capacity(dict_len);
    let mut temp_codes = HashMap::with_capacity_and_hasher(dict_len, FxBuildHasher);
    for (idx, value) in distinct.iter().enumerate() {
        temp_values.push(value.0);
        temp_codes.insert(value.0.to_bits(), I::from_index(idx));
    }

    let mut counts = vec![0usize; dict_len];
    let mut output = BufferMut::<I>::with_capacity(values.len());
    match validity_mask.map(Mask::bit_buffer) {
        Some(AllOr::None) => {
            output.push_n(I::ZERO, values.len());
        }
        Some(AllOr::All) => {
            for &value in values {
                let code = temp_codes.get(&value.to_bits()).copied().unwrap_or(I::ZERO);
                counts[code.into_index()] += 1;
                output.push(code);
            }
        }
        Some(AllOr::Some(bits)) => {
            encode_with_validity(values, bits, &temp_codes, &mut counts, &mut output);
        }
        None => {
            for &value in values {
                if let Some(&code) = temp_codes.get(&value.to_bits()) {
                    counts[code.into_index()] += 1;
                    output.push(code);
                } else {
                    output.push(I::ZERO);
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..dict_len).collect();
    order.sort_unstable_by(|&lhs, &rhs| {
        counts[rhs]
            .cmp(&counts[lhs])
            .then_with(|| temp_values[lhs].total_compare(temp_values[rhs]))
            .then_with(|| temp_values[lhs].to_bits().cmp(&temp_values[rhs].to_bits()))
    });

    let sorted_values: Buffer<T> = order.iter().map(|&old_idx| temp_values[old_idx]).collect();

    if order
        .iter()
        .enumerate()
        .any(|(new_idx, &old_idx)| new_idx != old_idx)
    {
        let mut remap = vec![I::ZERO; dict_len];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            remap[old_idx] = I::from_index(new_idx);
        }
        let out_slice = output.as_mut_slice();
        match validity_mask.map(Mask::bit_buffer) {
            Some(AllOr::None) => {}
            Some(AllOr::Some(bits)) => remap_with_validity(out_slice, bits, &remap),
            Some(AllOr::All) | None => {
                for code in out_slice {
                    *code = remap[code.into_index()];
                }
            }
        }
    }

    (sorted_values, output.freeze())
}

/// Encodes temporary codes and accumulates counts over valid positions only, writing `0` at nulls.
fn encode_with_validity<T: FloatBits, I: DictCode>(
    values: &[T],
    bits: &BitBuffer,
    temp_codes: &HashMap<T::Bits, I, FxBuildHasher>,
    counts: &mut [usize],
    output: &mut BufferMut<I>,
) {
    for_each_mask_word(bits, |word, start, len| {
        let slice = &values[start..start + len];
        if word == 0 {
            output.push_n(I::ZERO, len);
        } else if word == low_bits_mask(len) {
            for &value in slice {
                let code = temp_codes.get(&value.to_bits()).copied().unwrap_or(I::ZERO);
                counts[code.into_index()] += 1;
                output.push(code);
            }
        } else {
            let mut codes = [I::ZERO; 64];
            let mut valid = word;
            while valid != 0 {
                let lane = valid.trailing_zeros() as usize;
                let code = temp_codes
                    .get(&slice[lane].to_bits())
                    .copied()
                    .unwrap_or(I::ZERO);
                counts[code.into_index()] += 1;
                codes[lane] = code;
                valid &= valid - 1;
            }
            output.extend_from_slice(&codes[..len]);
        }
    });
}

/// Remaps valid code slots in place while leaving null slots at `0`.
fn remap_with_validity<I: DictCode>(out: &mut [I], bits: &BitBuffer, remap: &[I]) {
    for_each_mask_word(bits, |word, start, len| {
        if word == 0 {
            return;
        }
        let slice = &mut out[start..start + len];
        if word == low_bits_mask(len) {
            for code in slice {
                *code = remap[code.into_index()];
            }
        } else {
            let mut valid = word;
            while valid != 0 {
                let lane = valid.trailing_zeros() as usize;
                slice[lane] = remap[slice[lane].into_index()];
                valid &= valid - 1;
            }
        }
    });
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

        let array = PrimitiveArray::from_iter(data.iter().copied());
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
    fn test_deterministic_tie_breaking_special_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Every distinct bit pattern occurs twice, forcing total-order tie-breaking.
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

        let dict = dictionary_encode(array.as_view(), &stats)?;
        assert!(dict.has_all_values_referenced());
        let expected_order = [
            neg_nan,
            f64::NEG_INFINITY,
            -0.0f64,
            0.0f64,
            1.0f64,
            f64::INFINITY,
            pos_nan_a,
            pos_nan_b,
        ];
        let actual_bits: Vec<u64> = dict
            .values()
            .as_::<Primitive>()
            .as_slice::<f64>()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let expected_bits: Vec<u64> = expected_order.iter().map(|value| value.to_bits()).collect();
        assert_eq!(actual_bits, expected_bits);

        let decoded = dict.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        let decoded_bits: Vec<u64> = decoded
            .as_slice::<f64>()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let input_bits: Vec<u64> = input.iter().map(|value| value.to_bits()).collect();
        assert_eq!(decoded_bits, input_bits);
        Ok(())
    }

    #[test]
    fn test_f16_frequency_and_tie_breaking() -> VortexResult<()> {
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
        assert!(dict.has_all_values_referenced());
        let actual_bits: Vec<u16> = dict
            .values()
            .as_::<Primitive>()
            .as_slice::<f16>()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(
            actual_bits,
            vec![
                f16::from_f32(3.5).to_bits(),
                f16::NEG_ZERO.to_bits(),
                f16::ZERO.to_bits(),
            ]
        );

        let decoded = dict.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        let decoded_bits: Vec<u16> = decoded
            .as_slice::<f16>()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let input_bits: Vec<u16> = input.iter().map(|value| value.to_bits()).collect();
        assert_eq!(decoded_bits, input_bits);
        Ok(())
    }

    #[test]
    fn test_nulls_excluded_from_frequency_counts() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Span > 192 rows with all-valid, all-null, and mixed chunks plus a sliced bit offset.
        // `1.0` appears at 1 valid row and 80 null rows; `2.0` appears at 70 valid rows;
        // `999.0` appears only at null rows and must never enter the dictionary.
        let mut raw_values = Vec::with_capacity(200);
        let mut valid_bits = Vec::with_capacity(200);
        // Chunk 0 (0..64): all valid `2.0`.
        raw_values.extend(iter::repeat_n(2.0f32, 64));
        valid_bits.extend(iter::repeat_n(true, 64));
        // Chunk 1 (64..128): all null with underlying `1.0`.
        raw_values.extend(iter::repeat_n(1.0f32, 64));
        valid_bits.extend(iter::repeat_n(false, 64));
        // Chunk 2+ (128..200): mixed valid/null with `1.0`, `2.0`, and null-only `999.0`.
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

        // Slice the validity bitmap by 5 bits to exercise non-zero bit offsets.
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
        assert!(dict.has_all_values_referenced());
        assert_eq!(
            dict.values().as_::<Primitive>().as_slice::<f32>(),
            &[2.0f32, 1.0f32]
        );
        let codes_prim = dict.codes().as_::<Primitive>();
        let codes = codes_prim.as_slice::<u8>();
        for (idx, &is_valid) in valid_bits.iter().enumerate() {
            if !is_valid {
                assert_eq!(codes[idx], 0, "null code slot {idx} should remain 0");
            }
        }
        assert_arrays_eq!(dict, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_all_null_and_single_distinct_float_dict() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let all_null = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::AllInvalid);
        let all_null_stats = FloatStats::generate_opts(
            &all_null,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );
        let all_null_dict = dictionary_encode(all_null.as_view(), &all_null_stats)?;
        assert_eq!(all_null_dict.values().len(), 0);
        assert_arrays_eq!(all_null_dict, all_null, &mut ctx);

        let single = PrimitiveArray::new(
            buffer![4.5f32, 9.0, 4.5],
            Validity::from_iter([true, false, true]),
        );
        let single_stats = FloatStats::generate_opts(
            &single,
            GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );
        let single_dict = dictionary_encode(single.as_view(), &single_stats)?;
        assert_eq!(
            single_dict.values().as_::<Primitive>().as_slice::<f32>(),
            &[4.5f32]
        );
        assert_arrays_eq!(single_dict, single, &mut ctx);
        Ok(())
    }
}
