// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::slice::SliceKernel;
use crate::optimizer::ArrayOptimizer;

impl SliceKernel for Patched {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let inner = array.inner().slice(range.clone())?.optimize_ctx(ctx)?;
        crate::arrays::patched::compute::select_rows(
            array,
            inner,
            array.patches().slice(range, ctx)?,
            ctx,
        )
    }
}
