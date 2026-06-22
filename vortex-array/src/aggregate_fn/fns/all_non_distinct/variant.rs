// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use crate::aggregate_fn::fns::all_non_distinct::all_non_distinct;
use crate::ExecutionCtx;
use crate::arrays::VariantArray;
use crate::arrays::variant::VariantArrayExt;

pub(super) fn check_variant_identical(
    lhs: &VariantArray,
    rhs: &VariantArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let lhs_core_storage = lhs.core_storage();
    let rhs_core_storage = rhs.core_storage();

    all_non_distinct(lhs_core_storage, rhs_core_storage, ctx)
}
