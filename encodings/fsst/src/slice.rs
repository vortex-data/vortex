// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceReduce;
use vortex_array::arrays::VarBinVTable;
use vortex_error::VortexResult;

use crate::FSSTArray;
use crate::FSSTVTable;

impl SliceReduce for FSSTVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: slicing the `codes` leaves the symbol table intact
        Ok(Some(
            unsafe {
                FSSTArray::new_unchecked(
                    array.dtype().clone(),
                    array.symbols().clone(),
                    array.symbol_lengths().clone(),
                    array
                        .codes()
                        .slice(range.clone())?
                        .as_::<VarBinVTable>()
                        .clone(),
                    array.uncompressed_lengths().slice(range)?,
                )
            }
            .into_array(),
        ))
    }
}
