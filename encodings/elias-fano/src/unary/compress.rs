// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::Primitive;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::array::bit_len;
use super::vtable::Unary;
use super::vtable::UnaryArray;

/// Encode `array` as a [`Unary`] array.
///
/// Returns `None` when the values cannot, or should not, be stored this way:
/// - the dtype is not a non-nullable unsigned integer, as [`Unary`] has no validity yet;
/// - the values sum past `u64::MAX`;
/// - the bits, `len` plus that sum, would not fit a `usize`;
/// - the encoding would take more bits than the plain array's `len × bit width`, which happens
///   once the values average more than `bit width - 1`. This keeps large values from growing past
///   their input, and bounds what the encoder allocates.
pub fn unary_encode(
    array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<UnaryArray>> {
    let &DType::Primitive(ptype, NonNullable) = array.dtype() else {
        return Ok(None);
    };
    if !ptype.is_unsigned_int() {
        return Ok(None);
    }
    let len = array.len();
    // `sum` is null when the values overflow a `u64`.
    let Some(num_ones) = sum(array.array(), ctx)?
        .as_primitive()
        .try_typed_value::<u64>()?
    else {
        return Ok(None);
    };
    let Some(bits) = bit_len(len, num_ones) else {
        return Ok(None);
    };
    // A plain size past `usize::MAX` is larger than `bits`, which fits one.
    if len
        .checked_mul(ptype.bit_width())
        .is_some_and(|plain| bits > plain)
    {
        return Ok(None);
    }

    match_each_unsigned_integer_ptype!(ptype, |T| {
        let mut buffer = BitBufferMut::new_set(bits);
        let mut end = 0;
        for &value in array.as_slice::<T>() {
            end += AsPrimitive::<usize>::as_(value);
            // `num_ones` can be a cached statistic, so a wrong one must not run past the bits.
            vortex_ensure!(
                end < bits,
                "UnaryArray input sums past its sum statistic {num_ones}"
            );
            buffer.unset(end);
            end += 1;
        }
        Unary::try_new(buffer.freeze(), ptype, len).map(Some)
    })
}
