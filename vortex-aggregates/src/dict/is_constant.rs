// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::is_constant::IsConstant;
use crate::is_constant::is_constant;

/// Dict-specific is_constant kernel.
///
/// If codes are constant, the whole array is constant.
/// When all dictionary values are referenced, is_constant can be computed directly on the values
/// array. Otherwise, unreferenced values are filtered out first.
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

        if batch.is_empty() {
            return Ok(Some(IsConstant::make_partial(batch, false, ctx)?));
        }

        // If codes are constant, only one dictionary value is referenced (or all codes are null) -> constant.
        if is_constant(dict.codes(), ctx)? {
            return Ok(Some(IsConstant::make_partial(batch, true, ctx)?));
        }

        // Codes are non-constant and non-empty. If any code is null, there is at least one null
        // code and at least one valid code, so the dict is constant only if every referenced
        // dictionary value is also null.
        if !dict.codes().all_valid(ctx)? {
            let result = if !dict.values().dtype().is_nullable() {
                false
            } else if dict.has_all_values_referenced() {
                dict.values().all_invalid(ctx)?
            } else {
                let referenced_mask = dict.compute_referenced_values_mask(true, ctx)?;
                let mask = Mask::from(referenced_mask);
                let filtered_values = dict.values().filter(mask)?;
                filtered_values.all_invalid(ctx)?
            };
            return Ok(Some(IsConstant::make_partial(batch, result, ctx)?));
        }

        // All codes are valid; check the values array, filtering to only referenced values if needed.
        let result = if dict.has_all_values_referenced() {
            is_constant(dict.values(), ctx)?
        } else {
            let referenced_mask = dict.compute_referenced_values_mask(true, ctx)?;
            let mask = Mask::from(referenced_mask);
            let filtered_values = dict.values().filter(mask)?;
            is_constant(&filtered_values, ctx)?
        };

        Ok(Some(IsConstant::make_partial(batch, result, ctx)?))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::is_constant::IsConstant;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::non_null_value_and_null_code(vec![Some(0u32), None], vec![Some(42i32)], false, false)]
    #[case::non_null_value_and_null_code_all_ref(vec![Some(0u32), None], vec![Some(42i32)], true, false)]
    #[case::null_value_and_null_code(vec![Some(0u32), None], vec![None::<i32>], false, true)]
    #[case::null_value_and_null_code_all_ref(vec![Some(0u32), None], vec![None::<i32>], true, true)]
    #[case::all_null_codes(vec![None::<u32>, None], vec![Some(42i32)], false, true)]
    #[case::distinct_codes_same_value(vec![Some(0u32), Some(1)], vec![Some(42i32), Some(42)], true, true)]
    fn test_dict_is_constant_respects_null_codes(
        #[case] codes: Vec<Option<u32>>,
        #[case] values: Vec<Option<i32>>,
        #[case] all_values_referenced: bool,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::try_new(
            PrimitiveArray::from_option_iter(codes).into_array(),
            PrimitiveArray::from_option_iter(values).into_array(),
        )?;
        let dict = if all_values_referenced {
            // SAFETY: test cases with `all_values_referenced = true` reference every value index.
            unsafe { dict.set_all_values_referenced(true) }
        } else {
            dict
        }
        .into_array();

        let mut acc = Accumulator::try_new(IsConstant, EmptyOptions, dict.dtype().clone())?;
        acc.accumulate(&dict, &mut ctx)?;
        assert_eq!(
            acc.finish()?,
            Scalar::bool(expected, Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn test_dict_is_constant_non_nullable_values_with_null_code() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u32), None]).into_array(),
            buffer![42i32].into_array(),
        )?
        .into_array();

        let mut acc = Accumulator::try_new(IsConstant, EmptyOptions, dict.dtype().clone())?;
        acc.accumulate(&dict, &mut ctx)?;
        assert_eq!(acc.finish()?, Scalar::bool(false, Nullability::NonNullable));
        Ok(())
    }
}
