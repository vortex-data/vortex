// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::WidenArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::vtable::EliasFano;
use super::vtable::EliasFanoArray;

/// Encode `array` as an [`EliasFano`] array, leaving both children to the compressor.
///
/// The counts in `upper` are held in a [`Widen`](vortex_array::arrays::Widen) over the narrowest
/// type that holds them, unless only a `u64` does. `Widen` has no wire form yet, so the array can
/// be written only once the compressor has encoded `upper`: the EF scheme in `vortex-btrblocks`
/// forces it into [`Unary`](crate::Unary), and bit-packs `lower`.
///
/// Returns `None` when the dtype is not a non-nullable unsigned integer, or when the values are
/// not sorted.
pub fn elias_fano_encode(
    array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<EliasFanoArray>> {
    let &DType::Primitive(ptype, NonNullable) = array.dtype() else {
        return Ok(None);
    };
    if !ptype.is_unsigned_int() {
        return Ok(None);
    }

    match_each_unsigned_integer_ptype!(ptype, |T| {
        let values = array.as_slice::<T>();
        // Checked on the values, because a wrong `IsSorted` statistic would corrupt the output.
        if !values.is_sorted() {
            return Ok(None);
        }
        let max: u64 = values.last().map_or(0, |&max| max.as_());
        // Once `max` reaches the length, this gives between `len` and `2 × len` buckets.
        let lower_bits = (max / u64::try_from(values.len().max(1))?)
            .checked_ilog2()
            .unwrap_or(0);
        let mask = (1u64 << lower_bits) - 1;

        let mut upper = BufferMut::<u64>::zeroed(usize::try_from(max >> lower_bits)? + 1);
        let mut lower = BufferMut::<T>::with_capacity(values.len());
        for &value in values {
            let value: u64 = value.as_();
            upper[AsPrimitive::<usize>::as_(value >> lower_bits)] += 1;
            lower.push((value & mask).as_());
        }
        EliasFano::try_new(
            WidenArray::encode(PrimitiveArray::new(upper, Validity::NonNullable), ctx)?,
            PrimitiveArray::new(lower, Validity::NonNullable).into_array(),
            u8::try_from(lower_bits)?,
        )
        .map(Some)
    })
}
