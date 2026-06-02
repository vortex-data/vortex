// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::max::max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::min::min;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;

/// RunEnd-specific min/max kernel.
///
/// Run-end encoded arrays store each unique run value once, so extrema can be computed directly
/// on the values array without decoding the repeated runs.
#[derive(Debug)]
pub(crate) struct RunEndExtremaKernel;

impl DynAggregateKernel for RunEndExtremaKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let is_min = aggregate_fn.is::<Min>();
        let is_max = aggregate_fn.is::<Max>();
        if !is_min && !is_max {
            return Ok(None);
        }

        let Some(run_end) = batch.as_opt::<RunEnd>() else {
            return Ok(None);
        };

        let result = if is_min {
            min(run_end.values(), ctx)?
        } else {
            max(run_end.values(), ctx)?
        };

        Ok(Some(to_partial_scalar(result, batch.dtype())?))
    }
}

fn to_partial_scalar(value: Option<Scalar>, dtype: &DType) -> VortexResult<Scalar> {
    let partial_dtype = dtype.as_nullable();
    match value {
        Some(value) => value.cast(&partial_dtype),
        None => Ok(Scalar::null(partial_dtype)),
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFn;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::RunEnd;
    use crate::compute::extrema::RunEndExtremaKernel;

    fn kernel_extrema(array: &ArrayRef, is_min: bool) -> VortexResult<Option<i32>> {
        let aggregate_fn = if is_min {
            AggregateFn::new(Min, EmptyOptions).erased()
        } else {
            AggregateFn::new(Max, EmptyOptions).erased()
        };
        let scalar = RunEndExtremaKernel
            .aggregate(
                &aggregate_fn,
                array,
                &mut LEGACY_SESSION.create_execution_ctx(),
            )?
            .expect("run-end extrema kernel should handle run-end arrays");

        Option::<i32>::try_from(&scalar)
    }

    #[test]
    fn run_end_extrema_kernel_uses_run_values() -> VortexResult<()> {
        let array = RunEnd::encode(
            buffer![5i32, 5, -3, -3, 12, 12, 12].into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?
        .into_array();

        assert_eq!(kernel_extrema(&array, true)?, Some(-3));
        assert_eq!(kernel_extrema(&array, false)?, Some(12));
        Ok(())
    }
}
