use vortex_array::arrays::{VarBinVTable, varbin_scalar};
use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayExt, ArrayRef, IntoArray};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::Scalar;

use crate::{FSSTArray, FSSTVTable};

impl OperationsVTable<FSSTVTable> for FSSTVTable {
    fn slice(array: &FSSTArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        // Slicing an FSST array leaves the symbol table unmodified,
        // only slicing the `codes` array.
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols().clone(),
            array.symbol_lengths().clone(),
            array
                .codes()
                .slice(start, stop)?
                .as_::<VarBinVTable>()
                .clone(),
            array.uncompressed_lengths().slice(start, stop)?,
        )?
        .into_array())
    }

    fn scalar_at(array: &FSSTArray, index: usize) -> VortexResult<Scalar> {
        let compressed = array.codes().scalar_at(index)?;
        let binary_datum = compressed
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("expected null to already be handled"))?;

        let decoded_buffer =
            ByteBuffer::from(array.decompressor().decompress(binary_datum.as_slice()));
        Ok(varbin_scalar(decoded_buffer, array.dtype()))
    }
}
