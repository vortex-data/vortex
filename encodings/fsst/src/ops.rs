use vortex_array::arrays::{VarBinArray, varbin_scalar};
use vortex_array::{Array, ArrayExt, ArrayOperationsImpl, ArrayRef};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::Scalar;

use crate::FSSTArray;

impl ArrayOperationsImpl for FSSTArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        // Slicing an FSST array leaves the symbol table unmodified,
        // only slicing the `codes` array.
        Ok(FSSTArray::try_new(
            self.dtype().clone(),
            self.symbols().clone(),
            self.symbol_lengths().clone(),
            self.codes()
                .slice(start, stop)?
                .as_::<VarBinArray>()
                .clone(),
            self.uncompressed_lengths().slice(start, stop)?,
        )?
        .into_array())
    }

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let compressed = self.codes().scalar_at(index)?;
        let binary_datum = compressed
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("expected null to already be handled"))?;

        let decoded_buffer =
            ByteBuffer::from(self.decompressor().decompress(binary_datum.as_slice()));
        Ok(varbin_scalar(decoded_buffer, self.dtype()))
    }
}
