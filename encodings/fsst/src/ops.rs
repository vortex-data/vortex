use vortex_array::arrays::VarBinArray;
use vortex_array::{Array, ArrayExt, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;

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
}
