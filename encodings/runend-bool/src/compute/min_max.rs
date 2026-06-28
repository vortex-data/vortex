// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray as _;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::make_minmax_dtype;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEndBool;

/// RunEndBool-specific min/max kernel.
///
/// Min/max for a boolean array depends only on the decoded values, so we canonicalize to a
/// `BoolArray` and delegate. This still benefits from the optimized bool min/max path.
#[derive(Debug)]
pub(crate) struct RunEndBoolMinMaxKernel;

impl DynAggregateKernel for RunEndBoolMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<MinMax>() {
            return Ok(None);
        }

        if batch.as_opt::<RunEndBool>().is_none() {
            return Ok(None);
        }

        let canonical = batch.clone().execute::<Canonical>(ctx)?.into_array();
        let struct_dtype = make_minmax_dtype(batch.dtype());
        match min_max(&canonical, ctx)? {
            Some(result) => Ok(Some(Scalar::struct_(
                struct_dtype,
                vec![result.min, result.max],
            ))),
            None => Ok(Some(Scalar::null(struct_dtype))),
        }
    }
}
