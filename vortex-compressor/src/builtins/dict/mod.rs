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
