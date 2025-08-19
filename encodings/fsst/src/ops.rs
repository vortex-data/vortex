// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::{VarBinVTable, varbin_scalar};
use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::{FSSTArray, FSSTVTable};

impl OperationsVTable<FSSTVTable> for FSSTVTable {
    fn slice(array: &FSSTArray, start: usize, stop: usize) -> ArrayRef {
        // SAFETY: slicing the `codes` leaves the symbol table intact
        unsafe {
            FSSTArray::new_unchecked(
                array.dtype().clone(),
                array.symbols().clone(),
                array.symbol_lengths().clone(),
                array
                    .codes()
                    .slice(start, stop)
                    .as_::<VarBinVTable>()
                    .clone(),
                array.uncompressed_lengths().slice(start, stop),
            )
            .into_array()
        }
    }

    fn scalar_at(array: &FSSTArray, index: usize) -> Scalar {
        let compressed = array.codes().scalar_at(index);
        let binary_datum = compressed.as_binary().value().vortex_expect("non-null");

        let decoded_buffer =
            ByteBuffer::from(array.decompressor().decompress(binary_datum.as_slice()));
        varbin_scalar(decoded_buffer, array.dtype())
    }
}
