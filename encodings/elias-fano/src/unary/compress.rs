// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::array::Unary;
use super::array::UnaryArray;
use super::array::bit_len;

/// Encode `array` as a [`Unary`] array.
///
/// Returns `None` when the values cannot, or should not, be stored this way:
/// - the dtype is not a non-nullable unsigned integer;
/// - the values sum past `u64::MAX`;
/// - the encoding would take more bits than the plain array's `len × bit width`, which happens
///   once the values average more than `bit width - 1`. This keeps large values from growing past
///   their input, and bounds what the encoder allocates.
pub fn unary_encode(array: ArrayView<'_, Primitive>) -> VortexResult<Option<UnaryArray>> {
    let &DType::Primitive(ptype, NonNullable) = array.dtype() else {
        return Ok(None);
    };
    if !ptype.is_unsigned_int() {
        return Ok(None);
    }
    let len = array.len();

    match_each_unsigned_integer_ptype!(ptype, |T| {
        let values = array.as_slice::<T>();
        let Some(num_ones) = values
            .iter()
            .try_fold(0u64, |sum, &value| sum.checked_add(value.as_()))
        else {
            return Ok(None);
        };
        let Some(bits) = bit_len(len, num_ones) else {
            return Ok(None);
        };
        if len
            .checked_mul(ptype.bit_width())
            .is_none_or(|plain| bits > plain)
        {
            return Ok(None);
        }

        // Each value is at most `num_ones`, below `bits`, which fits a `usize`, so the conversion
        // is exact.
        let mut buffer = BitBufferMut::new_set(bits);
        let mut end = 0;
        for &value in values {
            end += AsPrimitive::<usize>::as_(value);
            buffer.unset(end);
            end += 1;
        }
        Unary::try_new(buffer.freeze(), ptype, len).map(Some)
    })
}

/// Decode every value with one walk over the unset bits.
pub(super) fn unary_decode(array: &UnaryArray) -> PrimitiveArray {
    let len = array.len();

    // `validate` keeps every value within the ptype, so the loop has nothing to check.
    match_each_unsigned_integer_ptype!(array.dtype().as_ptype(), |T| {
        let mut values = BufferMut::<T>::with_capacity(len);
        array.for_each_value(|value| values.push(AsPrimitive::<T>::as_(value)));
        PrimitiveArray::new(values, Validity::NonNullable)
    })
}
