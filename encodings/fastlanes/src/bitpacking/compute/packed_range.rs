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

/// `value` everywhere except `predicate(patch)` at patched positions.
pub(super) fn constant_bits<T: NativePType>(
    array: ArrayView<'_, BitPacked>,
    value: bool,
    predicate: &dyn Fn(T) -> bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BitBuffer> {
    let len = array.len();
    let allocator = ctx.allocator().clone();
    let Some(patches) = array.patches() else {
        return Ok(BitBuffer::full_in(value, len, allocator));
    };

    let fill = if value { u8::MAX } else { 0 };
    let bytes = ByteBufferMut::full_in(fill, len.div_ceil(8), allocator);
    let mut bits = BitBufferMut::from_buffer(bytes, 0, len);

    let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
    let offset = patches.offset();
    match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
        set_patch_bits(
            &mut bits,
            indices.as_slice::<I>(),
            values.as_slice::<T>(),
            offset,
            predicate,
        )
    });
    Ok(bits.freeze())
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
    let validity = array.validity()?.union_nullability(nullability);
    if array.patches().is_none() && validity.definitely_no_nulls() {
        let scalar = Scalar::bool(value, validity.nullability());
        return Ok(ConstantArray::new(scalar, array.len()).into_array());
    }
    let bits = constant_bits(array, value, predicate, ctx)?;
    Ok(BoolArray::new(bits, validity).into_array())
}

/// Set the bit of each patch, at its index less `patch_offset`, to `predicate` of its value.
pub(super) fn set_patch_bits<T: Copy, I: AsPrimitive<usize>>(
    bits: &mut BitBufferMut,
    indices: &[I],
    values: &[T],
    patch_offset: usize,
    predicate: &dyn Fn(T) -> bool,
) {
    for (&index, &patch) in indices.iter().zip(values) {
        let index: usize = index.as_();
        bits.set_to(index - patch_offset, predicate(patch));
    }
}
