// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::max::max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min::min;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Dict;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::dtype::DType;
use crate::scalar::Scalar;

/// Dict-specific min/max kernel.
///
/// When all dictionary values are referenced, extrema can be computed directly on the values
/// array. Otherwise, unreferenced values are filtered out first so they do not affect the result.
#[derive(Debug)]
pub(crate) struct DictExtremaKernel;

impl DynAggregateKernel for DictExtremaKernel {
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

        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };

        let result = if dict.has_all_values_referenced() {
            compute_extremum(is_min, dict.values(), ctx)?
        } else {
            let referenced_mask = dict.compute_referenced_values_mask(true)?;
            let mask = Mask::from(referenced_mask);
            let filtered_values = dict.values().filter(mask)?;
            compute_extremum(is_min, &filtered_values, ctx)?
        };

        Ok(Some(to_partial_scalar(result, batch.dtype())?))
    }
}

fn compute_extremum(
    is_min: bool,
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    if is_min {
        min(array, ctx)
    } else {
        max(array, ctx)
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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFn;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::max::Max;
    use crate::aggregate_fn::fns::min::Min;
    use crate::aggregate_fn::kernels::DynAggregateKernel;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::dict::compute::extrema::DictExtremaKernel;
    use crate::builders::dict::dict_encode;
    use crate::session::ArraySession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn dict_covering() -> DictArray {
        DictArray::try_new(
            buffer![0u32, 1, 2, 3, 0, 1].into_array(),
            buffer![10i32, 20, 30, 40].into_array(),
        )
        .expect("valid test dictionary")
    }

    fn dict_non_covering_duplicates() -> DictArray {
        DictArray::try_new(
            buffer![1u32, 1, 1, 3, 3].into_array(),
            buffer![1i32, 2, 3, 4, 5].into_array(),
        )
        .expect("valid test dictionary")
    }

    fn dict_nullable_values() -> DictArray {
        dict_encode(
            &PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), Some(1), None])
                .into_array(),
            &mut SESSION.create_execution_ctx(),
        )
        .expect("valid nullable-value dictionary")
    }

    fn dict_empty() -> DictArray {
        DictArray::try_new(
            PrimitiveArray::from_iter(Vec::<u32>::new()).into_array(),
            buffer![10i32, 20, 30].into_array(),
        )
        .expect("valid empty dictionary")
    }

    fn kernel_extrema(array: &ArrayRef, is_min: bool) -> VortexResult<Option<i32>> {
        let aggregate_fn = if is_min {
            AggregateFn::new(Min, EmptyOptions).erased()
        } else {
            AggregateFn::new(Max, EmptyOptions).erased()
        };
        let scalar = DictExtremaKernel
            .aggregate(&aggregate_fn, array, &mut SESSION.create_execution_ctx())?
            .expect("dict extrema kernel should handle dict arrays");

        Option::<i32>::try_from(&scalar)
    }

    #[rstest]
    #[case::covering(dict_covering(), Some(10), Some(40))]
    #[case::non_covering_duplicates(dict_non_covering_duplicates(), Some(2), Some(4))]
    #[case::nullable_values(dict_nullable_values(), Some(1), Some(2))]
    #[case::empty(dict_empty(), None, None)]
    fn dict_extrema_kernel(
        #[case] dict: DictArray,
        #[case] expected_min: Option<i32>,
        #[case] expected_max: Option<i32>,
    ) -> VortexResult<()> {
        let array = dict.into_array();

        assert_eq!(kernel_extrema(&array, true)?, expected_min);
        assert_eq!(kernel_extrema(&array, false)?, expected_max);
        Ok(())
    }
}
