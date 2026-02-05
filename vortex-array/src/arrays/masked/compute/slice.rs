// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::arrays::SliceReduce;
use crate::stats::ArrayStats;

impl SliceReduce for MaskedVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let child = array.child.slice(range.clone())?;
        let validity = array.validity.slice(range)?;

        Ok(Some(
            MaskedArray {
                child,
                validity,
                dtype: array.dtype.clone(),
                stats: ArrayStats::default(),
            }
            .into_array(),
        ))
    }
}
