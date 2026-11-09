// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Predicates decided by the bit width alone.
//!
//! Packed values lie in `[0, 2^W - 1]`, anything larger is a patch. A monotone predicate that
//! agrees at both ends of that range only needs the patches fixed up.

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;

/// `[0, 2^W - 1]` as `T`, or `None` when `W` is the full width of `T`.
pub(super) fn packed_range<T: NativePType>(bit_width: u8) -> Option<(T, T)> {
    if usize::from(bit_width) >= size_of::<T>() * 8 {
        return None;
    }
    Some((T::zero(), T::from((1u64 << bit_width) - 1)?))
}

/// `value` everywhere except `predicate(patch)` at patched positions. Without patches or nulls
/// it is a `ConstantArray`.
pub(super) fn constant_result<T: NativePType>(
    array: ArrayView<'_, BitPacked>,
    value: bool,
    nullability: Nullability,
    predicate: &dyn Fn(T) -> bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    let allocator = ctx.allocator().clone();
    let validity = array.validity()?.union_nullability(nullability);
    let Some(patches) = array.patches() else {
        if validity.definitely_no_nulls() {
            let scalar = Scalar::bool(value, validity.nullability());
            return Ok(ConstantArray::new(scalar, len).into_array());
        }
        return Ok(
            BoolArray::new(BitBuffer::full_in(value, len, allocator), validity).into_array(),
        );
    };

    let fill = if value { u8::MAX } else { 0 };
    let bytes = ByteBufferMut::full_in(fill, len.div_ceil(8), allocator);
    let mut bits = BitBufferMut::from_buffer(bytes, 0, len);

    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let offset = patches.offset();
    match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
        for (&index, &patch) in indices.as_slice::<I>().iter().zip(values.as_slice::<T>()) {
            let index: usize = index.as_();
            bits.set_to(index - offset, predicate(patch));
        }
    });
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
}
