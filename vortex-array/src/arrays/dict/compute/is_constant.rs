// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::is_constant::IsConstant;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Dict;
use crate::scalar::Scalar;

/// Dict-specific is_constant kernel.
///
/// If codes are constant, the whole array is constant.
/// Otherwise, check the values array.
#[derive(Debug)]
pub(crate) struct DictIsConstantKernel;

impl DynAggregateKernel for DictIsConstantKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<IsConstant>() {
            return Ok(None);
        }

        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };

        let result = if is_constant(dict.codes(), ctx)? {
            true
        } else {
            is_constant(dict.values(), ctx)?
        };

        // Return in the partial dtype format: struct {is_constant, value}
        // We use the first scalar as the representative value.
        let partial_dtype =
            crate::aggregate_fn::fns::is_constant::make_is_constant_partial_dtype(batch.dtype());
        if result {
            let first_value = if batch.is_empty() {
                return Ok(Some(Scalar::null(partial_dtype)));
            } else {
                batch.scalar_at(0)?.into_nullable()
            };
            Ok(Some(Scalar::struct_(
                partial_dtype,
                vec![
                    Scalar::bool(true, crate::dtype::Nullability::NonNullable),
                    first_value,
                ],
            )))
        } else {
            Ok(Some(Scalar::struct_(
                partial_dtype,
                vec![
                    Scalar::bool(false, crate::dtype::Nullability::NonNullable),
                    Scalar::null(batch.dtype().as_nullable()),
                ],
            )))
        }
    }
}
