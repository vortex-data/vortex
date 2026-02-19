// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceReduce;
use vortex_array::arrays::VarBinVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

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
                    VarBinVTable::_slice(array.codes().as_::<VarBinVTable>(), range.clone())?
                        .try_into::<VarBinVTable>()
                        .map_err(|_| vortex_err!("cannot fail conversion"))?,
                    array.uncompressed_lengths().slice(range)?,
                )
            }
            .into_array(),
        ))
    }
}
