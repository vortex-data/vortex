// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::make_minmax_dtype;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::Dict;
use crate::scalar::Scalar;

/// Dict-specific min/max kernel.
///
/// When all dictionary values are referenced, min/max can be computed directly on the values
/// array. Otherwise, unreferenced values are filtered out first.
#[derive(Debug)]
pub(crate) struct DictMinMaxKernel;

impl DynAggregateKernel for DictMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<MinMax>() {
            return Ok(None);
        }

        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };

        let struct_dtype = make_minmax_dtype(batch.dtype());

        let result = if dict.has_all_values_referenced() {
            // All values are referenced, compute min/max directly on the values array.
            min_max(dict.values(), ctx)?
        } else {
            // Filter to only referenced values, then compute min/max.
            let referenced_mask = dict.compute_referenced_values_mask(true)?;
            let mask = Mask::from(referenced_mask);
            let filtered_values = dict.values().filter(mask)?;
            min_max(&filtered_values, ctx)?
        };

        match result {
            Some(r) => Ok(Some(Scalar::struct_(struct_dtype, vec![r.min, r.max]))),
            None => Ok(Some(Scalar::null(struct_dtype))),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::fns::min_max::min_max;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::builders::dict::dict_encode;

    fn assert_min_max(array: &ArrayRef, expected: Option<(i32, i32)>) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        match (min_max(array, &mut ctx)?, expected) {
            (Some(result), Some((expected_min, expected_max))) => {
                assert_eq!(i32::try_from(&result.min)?, expected_min);
                assert_eq!(i32::try_from(&result.max)?, expected_max);
            }
            (None, None) => {}
            (got, expected) => panic!(
                "min_max mismatch: expected {expected:?}, got {:?}",
                got.as_ref().map(|r| (
                    i32::try_from(&r.min.clone()).ok(),
                    i32::try_from(&r.max.clone()).ok()
                ))
            ),
        }
        Ok(())
    }

    #[rstest]
    #[case::covering(
        DictArray::try_new(
            buffer![0u32, 1, 2, 3, 0, 1].into_array(),
            buffer![10i32, 20, 30, 40].into_array(),
        ).unwrap(),
        (10, 40)
    )]
    #[case::non_covering_duplicates(
        DictArray::try_new(
            buffer![1u32, 1, 1, 3, 3].into_array(),
            buffer![1i32, 2, 3, 4, 5].into_array(),
        ).unwrap(),
        (2, 4)
    )]
    #[case::non_covering_gaps(
        DictArray::try_new(
            buffer![0u32, 2, 4].into_array(),
            buffer![1i32, 2, 3, 4, 5].into_array(),
        ).unwrap(),
        (1, 5)
    )]
    #[case::single(dict_encode(&buffer![42i32].into_array()).unwrap(), (42, 42))]
    #[case::nullable_codes(
        DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u32), None, Some(1), Some(2)]).into_array(),
            buffer![10i32, 20, 30].into_array(),
        ).unwrap(),
        (10, 30)
    )]
    #[case::nullable_values(
        dict_encode(
            &PrimitiveArray::from_option_iter([Some(1i32), None, Some(2), Some(1), None]).into_array()
        ).unwrap(),
        (1, 2)
    )]
    fn test_min_max(#[case] dict: DictArray, #[case] expected: (i32, i32)) -> VortexResult<()> {
        assert_min_max(&dict.into_array(), Some(expected))
    }

    #[test]
    fn test_sliced_dict() -> VortexResult<()> {
        let reference = PrimitiveArray::from_iter([1, 5, 10, 50, 100]);
        let dict = dict_encode(&reference.into_array())?;
        let sliced = dict.slice(1..3)?;
        assert_min_max(&sliced, Some((5, 10)))
    }

    #[rstest]
    #[case::empty(
        DictArray::try_new(
            PrimitiveArray::from_iter(Vec::<u32>::new()).into_array(),
            buffer![10i32, 20, 30].into_array(),
        ).unwrap()
    )]
    #[case::all_null_codes(
        DictArray::try_new(
            PrimitiveArray::from_option_iter([Option::<u32>::None, None, None]).into_array(),
            buffer![10i32, 20, 30].into_array(),
        ).unwrap()
    )]
    fn test_min_max_none(#[case] dict: DictArray) -> VortexResult<()> {
        assert_min_max(&dict.into_array(), None)
    }
}
