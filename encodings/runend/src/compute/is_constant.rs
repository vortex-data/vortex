// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::aggregate_fn::fns::is_constant::make_is_constant_partial_dtype;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEnd;

/// RunEnd-specific is_constant kernel.
///
/// If the values array of a run-end array is constant, the entire array is constant.
#[derive(Debug)]
pub(crate) struct RunEndIsConstantKernel;

impl DynAggregateKernel for RunEndIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        let result = is_constant(array.values(), ctx)?;
        let partial_dtype = make_is_constant_partial_dtype(batch.dtype());

        if result {
            let first_value = if batch.is_empty() {
                return Ok(Some(Scalar::null(partial_dtype)));
            } else {
                batch.scalar_at(0)?.into_nullable()
            };
            Ok(Some(Scalar::struct_(
                partial_dtype,
                vec![Scalar::bool(true, Nullability::NonNullable), first_value],
            )))
        } else {
            Ok(Some(Scalar::struct_(
                partial_dtype,
                vec![
                    Scalar::bool(false, Nullability::NonNullable),
                    Scalar::null(batch.dtype().as_nullable()),
                ],
            )))
        }
    }
}
