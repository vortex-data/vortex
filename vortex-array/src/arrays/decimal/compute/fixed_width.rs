// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::buffer::BufferHandle;
use crate::validity::Validity;

impl FixedWidthArray for Decimal {
    fn byte_width(array: ArrayView<'_, Self>) -> usize {
        array.values_type().byte_width()
    }

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer {
        array.buffer_handle().to_host_sync()
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        _len: usize,
        validity: Validity,
    ) -> VortexResult<DecimalArray> {
        DecimalArray::try_new_handle(
            BufferHandle::new_host(values),
            array.values_type(),
            array.decimal_dtype(),
            validity,
        )
    }
}
