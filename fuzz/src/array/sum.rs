// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray as _;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

/// Compute sum on the canonical form of the array to get a consistent baseline.
pub fn sum_canonical_array(canonical: Canonical, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    // TODO(joe): replace with baseline not using canonical
    sum(&canonical.into_array(), ctx)
}
