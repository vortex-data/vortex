// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::match_each_unsigned_integer_ptype;

/// Decodes a sorted, non-null array of unsigned positions, such as patch indices or run ends,
/// into a `u64` buffer that can be binary searched directly.
///
/// A repeated probe decodes such a child once instead of reading it a value at a time on every
/// search. A one-off search should not call this: it costs a full decode.
pub fn decode_sorted_positions(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<u64>> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_unsigned_integer_ptype!(primitive.ptype(), |T| {
        // Every unsigned position type widens to `u64` without loss.
        Ok(primitive
            .as_slice::<T>()
            .iter()
            .map(|&position| AsPrimitive::<u64>::as_(position))
            .collect::<Buffer<u64>>())
    })
}
