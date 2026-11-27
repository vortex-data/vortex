// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dictionary encoding schemes for binary, decimal, integer, float, and string arrays.

mod binary;
mod bytes_candidate;
mod decimal;
mod float;
mod integer;
mod string;

use std::hash::Hash;

pub use binary::BinaryDictScheme;
pub use decimal::DecimalDictScheme;
pub use float::FloatDictScheme;
pub use float::dictionary_encode as float_dictionary_encode;
pub use integer::IntDictScheme;
pub use integer::dictionary_encode as integer_dictionary_encode;
use rustc_hash::FxBuildHasher;
pub use string::StringDictScheme;
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

    let mut entries: Vec<(T, u32)> = counts.collect();
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
    // and the caller guarantees `counts` holds the exact distinct valid value set.
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
