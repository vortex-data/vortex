// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer {
        array.buffer_handle().to_host_sync()
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
