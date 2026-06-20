// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray as _;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_error::VortexResult;

/// Compute min_max on the canonical form of the array to get a consistent baseline.
pub fn min_max_canonical_array(
    canonical: Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<MinMaxResult>> {
    min_max(
        &canonical.into_array(),
        ctx,
        NumericalAggregateOpts::default(),
    )
}
