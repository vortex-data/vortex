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

use crate::DecimalByteParts;

/// DecimalByteParts-specific is_constant kernel.
///
/// Delegates to checking if the MSP (most significant part) is constant.
#[derive(Debug)]
pub(crate) struct DecimalBytePartsIsConstantKernel;

impl DynAggregateKernel for DecimalBytePartsIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<DecimalByteParts>() else {
            return Ok(None);
        };

        let result = is_constant(array.msp(), ctx)?;
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
