// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::OperationsVTable;
use vortex_array::{ArrayRef, IntoArray};
use vortex_buffer::BufferString;
use vortex_dtype::DType;
use vortex_scalar::Scalar;

use crate::fsst_view::{FSSTViewArray, FSSTViewVTable};

impl OperationsVTable<FSSTViewVTable> for FSSTViewVTable {
    fn slice(array: &FSSTViewArray, start: usize, stop: usize) -> ArrayRef {
        FSSTViewArray {
            views: array.views.slice(start..stop),
            fsst_buffer: array.fsst_buffer.clone(),
            compressor: array.compressor.clone(),
            compressed_offsets: array.compressed_offsets.clone(),
            uncompressed_offsets: array.uncompressed_offsets.clone(),
            validity: array.validity.clone(),
        }
        .into_array()
    }

    fn scalar_at(array: &FSSTViewArray, index: usize) -> Scalar {
        let bytes = array.bytes_at(index);
        match array.dtype() {
            DType::Utf8(n) => Scalar::utf8(unsafe { BufferString::new_unchecked(bytes) }, *n),
            DType::Binary(n) => Scalar::binary(bytes, *n),
            _ => unreachable!("FSSTViewArray can only be utf8/binary, checked at construction"),
        }
    }
}
