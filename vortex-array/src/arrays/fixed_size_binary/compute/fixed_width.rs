// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::arrays::FixedSizeBinary;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::buffer::BufferHandle;
use crate::validity::Validity;

impl FixedWidthArray for FixedSizeBinary {
    fn byte_width(array: ArrayView<'_, Self>) -> usize {
        array.byte_width() as usize
    }

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer {
        FixedSizeBinaryArrayExt::values(&array)
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        len: usize,
        validity: Validity,
    ) -> VortexResult<FixedSizeBinaryArray> {
        FixedSizeBinaryArray::try_new_handle(
            BufferHandle::new_host(values),
            array.byte_width(),
            len,
            validity,
        )
    }
}
