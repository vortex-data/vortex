// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_error::VortexResult;

use crate::ALPArray;
use crate::ALPVTable;

impl SliceKernel for ALPVTable {
    fn slice(
        array: &Self::Array,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let sliced_alp = ALPArray::new(
            array.encoded().slice(range.clone())?,
            array.exponents(),
            array
                .patches()
                .map(|p| p.slice(range))
                .transpose()?
                .flatten(),
        )
        .into_array();
        Ok(Some(sliced_alp))
    }
}
