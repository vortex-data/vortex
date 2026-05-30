// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata-only `filter`/`take` that go straight from an [`FSSTArray`] to an [`FSSTViewArray`].
//!
//! These are the "first hop" of the view pipeline. They never touch the compressed byte heap:
//! the [`FSSTArray`] is reinterpreted as an [`FSSTViewArray`] (sharing symbols + codes bytes,
//! deriving `sizes` from the consecutive offsets) and then the selection is applied to the small
//! `offsets`/`sizes`/`lengths`/`validity` arrays only.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterKernel;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::array::FSSTView;
use super::array::FSSTViewArray;
use super::array::fsstview_from_fsst;
use crate::FSSTArray;

/// Filter an [`FSSTArray`], producing an [`FSSTViewArray`] without touching the codes.
pub fn fsst_filter_to_view(
    array: &FSSTArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTViewArray> {
    let view = fsstview_from_fsst(array, ctx)?;
    let filtered: ArrayRef = <FSSTView as FilterKernel>::filter(view.as_view(), mask, ctx)?
        .vortex_expect("FSSTView filter always returns Some");
    filtered
        .try_downcast::<FSSTView>()
        .map_err(|_| vortex_err!("FSSTView filter must return an FSSTView"))
}

/// Take from an [`FSSTArray`], producing an [`FSSTViewArray`] without touching the codes.
pub fn fsst_take_to_view(
    array: &FSSTArray,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FSSTViewArray> {
    let view = fsstview_from_fsst(array, ctx)?;
    let taken: ArrayRef = <FSSTView as TakeExecute>::take(view.as_view(), indices, ctx)?
        .vortex_expect("FSSTView take always returns Some");
    taken
        .try_downcast::<FSSTView>()
        .map_err(|_| vortex_err!("FSSTView take must return an FSSTView"))
}
