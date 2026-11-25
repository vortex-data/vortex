// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::primitive::compare_primitive;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::DurationArray;
use crate::arrays::duration::DurationArraySlotsExt;
use crate::scalar_fn::builtin::operators::CompareOperator;

pub(super) fn compare_duration(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let lhs = lhs.clone().execute::<DurationArray>(ctx)?;
    let rhs = rhs.clone().execute::<DurationArray>(ctx)?;
    compare_primitive(lhs.values(), rhs.values(), op, ctx)
}
