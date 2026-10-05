// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::FixedWidthTakeValue;
use super::take_values;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::match_each_unsigned_integer_ptype;

// Avoid duplicating the indices dispatch in every record-width arm of `take`.
#[inline(never)]
pub(super) fn take_records<V: FixedWidthArray, T: FixedWidthTakeValue>(
    array: ArrayView<'_, V>,
    indices: &PrimitiveArray,
    allocator: &BufferAllocatorRef,
) -> VortexResult<ByteBuffer> {
    let values = V::values::<T>(array);
    vortex_ensure!(
        values.len() == array.len(),
        "Fixed-width values buffer length does not match record count"
    );
    let taken = match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
        take_values(values.as_slice(), indices.as_slice::<I>(), allocator)
    });
    Ok(taken.into_byte_buffer().aligned(values.alignment()))
}
