// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dictionary encoding schemes for binary, decimal, integer, float, and string arrays.

mod binary;
mod bytes_candidate;
mod decimal;
mod float;
mod integer;
mod string;

use std::cmp::Ordering;
use std::hash::Hash;

pub use binary::BinaryDictScheme;
pub use decimal::DecimalDictScheme;
pub use float::FloatDictScheme;
pub use float::dictionary_encode as float_dictionary_encode;
pub use integer::IntDictScheme;
pub use integer::dictionary_encode as integer_dictionary_encode;
use rustc_hash::FxBuildHasher;
pub use string::StringDictScheme;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

/// Encodes a primitive array into a [`DictArray`] from precomputed `(value, count)` pairs,
/// ordering dictionary values by descending occurrence count and breaking ties with
/// [`NativePType::total_compare`].
///
/// The caller must supply the exact distinct valid value set of `source_array` so every valid
/// element maps to a code in `0..values.len()` and every dictionary value is referenced.
fn dictionary_encode_from_counts<T: NativePType>(
    source_array: ArrayView<'_, Primitive>,
    counts: impl Iterator<Item = (T, u32)>,
) -> VortexResult<DictArray>
where
    NativeValue<T>: Eq + Hash,
{
    let codes_validity = source_array.validity()?;
    let values_validity = match codes_validity {
        Validity::NonNullable => Validity::NonNullable,
        _ => Validity::AllValid,
    };

    let (values, codes) = encode_sorted_codes(
        source_array.as_slice::<T>(),
        codes_validity,
        counts,
        NativeValue,
        NativePType::total_compare,
    );

    let values = PrimitiveArray::new(values, values_validity).into_array();
    // SAFETY: `encode_sorted_codes` emits codes in `0..values.len()` for every valid entry
    // and the caller guarantees `counts` holds the exact distinct valid value set.
    Ok(unsafe { DictArray::new_unchecked(codes, values).set_all_values_referenced(true) })
}

/// Sorts `(value, count)` pairs by descending count (breaking ties with `cmp`) and encodes
/// `source_slice` into the narrowest unsigned integer codes array (`u8`, `u16`, or `u32`).
fn encode_sorted_codes<T, K>(
    source_slice: &[T],
    codes_validity: Validity,
    counts: impl Iterator<Item = (T, u32)>,
    key: impl Fn(T) -> K,
    cmp: impl Fn(T, T) -> Ordering,
) -> (Buffer<T>, ArrayRef)
where
    T: Copy,
    K: Eq + Hash,
{
    let mut entries: Vec<(T, u32)> = counts.collect();
    entries.sort_unstable_by(|&(lhs_val, lhs_count), &(rhs_val, rhs_count)| {
        rhs_count
            .cmp(&lhs_count)
            .then_with(|| cmp(lhs_val, rhs_val))
    });
    let values: Buffer<T> = entries.into_iter().map(|(value, _)| value).collect();

    // The largest code is one less than the dictionary length.
    let dict_len = values.len();
    let codes = if dict_len <= u8::MAX as usize + 1 {
        let buf = encode_codes::<T, K, u8>(&values, source_slice, &key);
        PrimitiveArray::new(buf, codes_validity).into_array()
    } else if dict_len <= u16::MAX as usize + 1 {
        let buf = encode_codes::<T, K, u16>(&values, source_slice, &key);
        PrimitiveArray::new(buf, codes_validity).into_array()
    } else {
        let buf = encode_codes::<T, K, u32>(&values, source_slice, &key);
        PrimitiveArray::new(buf, codes_validity).into_array()
    };

    (values, codes)
}

/// Maps `values` to their dictionary codes in `distinct`.
fn encode_codes<T, K, I>(distinct: &[T], values: &[T], key: impl Fn(T) -> K) -> Buffer<I>
where
    T: Copy,
    K: Eq + Hash,
    I: UnsignedPType,
{
    let mut codes =
        HashMap::<K, I, FxBuildHasher>::with_capacity_and_hasher(distinct.len(), FxBuildHasher);
    for (code, &value) in distinct.iter().enumerate() {
        codes.insert(
            key(value),
            I::from(code).vortex_expect("caller ensures dictionary length fits in code type"),
        );
    }

    let mut output = BufferMut::with_capacity(values.len());
    for &value in values {
        // Any code lookups which fail are for nulls, so their value does not matter.
        // SAFETY: `output` has capacity `values.len()`.
        unsafe {
            output.push_unchecked(codes.get(&key(value)).copied().unwrap_or_default());
        }
    }

    output.freeze()
}

/// FastLanes bit-packing aligns packed buffers to 1024-element blocks.
pub(super) const FASTLANES_BLOCK_LEN: u64 = 1024;

/// Minimum bit savings `BitPackingScheme` requires to pack a primitive array rather than leave
/// it uncompressed.
pub(super) const BITPACKING_MIN_BITS_SAVED: u64 = 2;

/// Minimum average run length before RLE / RunEnd schemes consider compressing codes.
const RLE_MIN_RUN_LENGTH: u32 = 4;

/// Minimum bit width required to represent dictionary codes `0..distinct_count`.
///
/// The largest code is `distinct_count - 1`, so power-of-two cardinalities (such as `2` -> `1`
/// bit, `4` -> `2` bits, `256` -> `8` bits) do not pay an extra bit.
pub(super) fn min_code_bit_width(distinct_count: u32) -> u32 {
    debug_assert!(distinct_count > 0);
    u32::BITS - (distinct_count - 1).leading_zeros()
}

/// Bit width of the narrowest unsigned integer type (`u8`, `u16`, or `u32`) that holds
/// `distinct_count` dictionary codes `0..distinct_count`.
pub(super) fn codes_ptype_bits(distinct_count: u32) -> u64 {
    if distinct_count <= u32::from(u8::MAX) + 1 {
        u64::from(u8::BITS)
    } else if distinct_count <= u32::from(u16::MAX) + 1 {
        u64::from(u16::BITS)
    } else {
        u64::from(u32::BITS)
    }
}

/// Computes the exact bit-width frequency histogram of frequency-ordered dictionary codes in
/// `O(K)` time via top-down selection at power-of-two cutoffs.
///
/// Entry `bw` holds the total number of valid rows assigned a code whose minimum bit width is
/// `bw` (code `0` has width `0`; codes `2^(bw-1)..2^bw` have width `bw`).
pub(super) fn frequency_code_bit_width_histogram(
    counts: impl Iterator<Item = u32>,
    distinct_count: u32,
) -> ([u64; 33], u32) {
    let max_bw = min_code_bit_width(distinct_count);
    let mut histogram = [0u64; 33];
    if max_bw == 0 {
        histogram[0] = counts.map(u64::from).sum();
        return (histogram, 0);
    }

    let mut freqs: Vec<u32> = Vec::with_capacity(distinct_count as usize);
    freqs.extend(counts);
    debug_assert_eq!(freqs.len(), distinct_count as usize);

    // Partition top-down at 2^(max_bw-1), 2^(max_bw-2), ..., 1 so total work is K + K/2 + ... < 2K.
    let mut hi = freqs.len();
    for bw in (0..max_bw).rev() {
        let cap = 1usize << bw;
        debug_assert!(cap < hi);
        freqs[..hi].select_nth_unstable_by(cap, |a, b| b.cmp(a));
        histogram[(bw + 1) as usize] = freqs[cap..hi].iter().map(|&c| u64::from(c)).sum();
        hi = cap;
    }
    histogram[0] = u64::from(freqs[0]);
    (histogram, max_bw)
}

/// Estimates the compressed bit size of dictionary `codes` (excluding validity).
///
/// Considers uncompressed narrowed codes (`u8`/`u16`/`u32`), patched FastLanes bit-packing
/// (mirroring `BitPacking::find_best_bit_width` followed by its `MIN_BITS_SAVED` check when a
/// frequency `histogram` is supplied), and RLE/RunEnd when effective runs average at least 4 rows
/// and the dictionary deduplicates them.
pub(super) fn estimate_codes_bits(
    array_len: u64,
    value_count: u64,
    distinct_count: u32,
    run_count: u32,
    histogram: Option<&[u64; 33]>,
    finished_cascading: bool,
) -> u64 {
    let ptype_bits = codes_ptype_bits(distinct_count);
    let uncompressed_bits = array_len.saturating_mul(ptype_bits);
    if finished_cascading {
        return uncompressed_bits;
    }

    let max_bw_u32 = min_code_bit_width(distinct_count);
    let max_bw = u64::from(max_bw_u32);
    let packed_len = array_len.next_multiple_of(FASTLANES_BLOCK_LEN);
    // Match `BitPacking::find_best_bit_width`: each exception stores a code value plus a u32 index.
    let bits_per_exception = ptype_bits + u64::from(u32::BITS);

    let mut best_bits = uncompressed_bits;

    match histogram {
        Some(hist) => {
            // Mirror `BitPackingScheme::compress_global`: `find_best_bit_width` first picks the
            // cheapest bit width over `0..=max_bw`, and `should_pack` then rejects widths that
            // save fewer than `BITPACKING_MIN_BITS_SAVED` bits (falling back to uncompressed).
            // Null slots are encoded as code `0`, so they never become bit-packing exceptions.
            let mut packed_valid = 0u64;
            let mut chosen_bw = 0u64;
            let mut chosen_exceptions = value_count;
            let mut best_selection_cost = u64::MAX;
            for (bw, &count) in hist.iter().enumerate().take(max_bw_u32 as usize + 1) {
                packed_valid += count;
                let bw = bw as u64;
                let exceptions = value_count.saturating_sub(packed_valid);
                let selection_cost = array_len
                    .saturating_mul(bw)
                    .saturating_add(exceptions.saturating_mul(bits_per_exception));
                if selection_cost < best_selection_cost {
                    best_selection_cost = selection_cost;
                    chosen_bw = bw;
                    chosen_exceptions = exceptions;
                }
            }
            if chosen_bw + BITPACKING_MIN_BITS_SAVED <= ptype_bits {
                let packed_bits = if chosen_bw == 0 {
                    0
                } else {
                    packed_len.saturating_mul(chosen_bw)
                };
                let candidate_bits = packed_bits
                    .saturating_add(chosen_exceptions.saturating_mul(bits_per_exception));
                best_bits = best_bits.min(candidate_bits);
            }
        }
        None => {
            if max_bw + BITPACKING_MIN_BITS_SAVED <= ptype_bits {
                let packed_bits = if max_bw == 0 {
                    0
                } else {
                    packed_len.saturating_mul(max_bw)
                };
                best_bits = best_bits.min(packed_bits);
            }
        }
    }

    // `run_count` ignores nulls, but dictionary `codes` keeps a slot per null and `RunEnd` stores
    // nulls in its values stream, so each scattered null can split a run (up to +2 runs per null).
    // Only credit RLE/RunEnd when the dictionary meaningfully deduplicates the effective runs
    // (`distinct_count <= rle_runs / 2`); otherwise direct RunEnd dominates Dict(RunEnd).
    let null_count = array_len.saturating_sub(value_count);
    let rle_runs = u64::from(run_count)
        .saturating_add(null_count.saturating_mul(2))
        .min(array_len);
    let effective_avg_run_len = array_len.checked_div(rle_runs).unwrap_or(0);
    if effective_avg_run_len >= u64::from(RLE_MIN_RUN_LENGTH)
        && u64::from(distinct_count) <= rle_runs / 2
    {
        let rle_bits = rle_runs.saturating_mul(max_bw + u64::from(u32::BITS));
        best_bits = best_bits.min(rle_bits);
    }

    best_bits
}
