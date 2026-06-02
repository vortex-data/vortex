// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray as _;
use vortex_array::aggregate_fn::fns::max::max;
use vortex_array::aggregate_fn::fns::min::min;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

/// Compute min on the canonical form of the array to get a consistent baseline.
pub fn min_canonical_array(
    canonical: Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    min_array(&canonical.into_array(), ctx)
}

/// Compute max on the canonical form of the array to get a consistent baseline.
pub fn max_canonical_array(
    canonical: Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    max_array(&canonical.into_array(), ctx)
}

/// Compute min independently.
pub fn min_array(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Scalar>> {
    min(array, ctx)
}

/// Compute max independently.
pub fn max_array(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Scalar>> {
    max(array, ctx)
}
