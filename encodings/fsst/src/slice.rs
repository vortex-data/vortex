// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::FSST;
use crate::FSSTData;

impl SliceReduce for FSST {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: slicing the `codes` leaves the symbol table intact
        Ok(Some(
            unsafe {
                FSSTData::new_unchecked(
                    array.dtype().clone(),
                    array.symbols().clone(),
                    array.symbol_lengths().clone(),
                    array
                        .codes()
                        .slice(range.clone())?
                        .try_into::<VarBin>()
                        .map_err(|_| vortex_err!("cannot fail conversion"))?,
                    array.uncompressed_lengths().slice(range)?,
                )
            }
            .into_array(),
        ))
    }
}
