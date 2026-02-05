// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::ScalarFnVTable;
use crate::arrays::SliceReduce;

impl SliceReduce for ScalarFnVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let children: Vec<_> = array
            .children()
            .iter()
            .map(|c| c.slice(range.clone()))
            .collect::<VortexResult<_>>()?;

        Ok(Some(
            ScalarFnArray {
                scalar_fn: array.scalar_fn.clone(),
                dtype: array.dtype.clone(),
                len: range.len(),
                children,
                stats: Default::default(),
            }
            .into_array(),
        ))
    }
}
