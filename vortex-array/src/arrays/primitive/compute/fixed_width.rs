// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::validity::Validity;

impl FixedWidthArray for Primitive {
    fn byte_width(array: ArrayView<'_, Self>) -> usize {
        array.ptype().byte_width()
    }

    fn values<T: Copy>(array: ArrayView<'_, Self>) -> Buffer<T> {
        let values = array.buffer_handle().to_host_sync();
        let alignment = values.alignment();
        Buffer::from_byte_buffer_aligned(values, alignment)
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        _len: usize,
        validity: Validity,
    ) -> VortexResult<PrimitiveArray> {
        Ok(PrimitiveArray::from_byte_buffer(
            values,
            array.ptype(),
            validity,
        ))
    }
}
