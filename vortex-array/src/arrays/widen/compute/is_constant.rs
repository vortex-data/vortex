// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::scalar::Scalar;

/// Computes [`IsConstant`] of a [`Widen`] from its values.
///
/// Decoding is injective and keeps the validity, so the `Widen` is constant exactly when its
/// values are. [`IsConstant::make_partial`] reads the first value through the `Widen`, so the
/// partial holds the decoded value.
#[derive(Debug)]
pub(crate) struct WidenIsConstantKernel;

impl DynAggregateKernel for WidenIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(widen) = batch.as_opt::<Widen>() else {
            return Ok(None);
        };

        // NB: The free `is_constant` reports an empty array as not constant, but an empty batch
        // has the empty partial, which `make_partial` builds only for a constant verdict.
        let is_constant = batch.is_empty() || is_constant(widen.values(), ctx)?;

        Ok(Some(IsConstant::make_partial(batch, is_constant, ctx)?))
    }
}
