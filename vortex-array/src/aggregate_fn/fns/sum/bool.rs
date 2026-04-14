// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;

use super::SumState;
use super::checked_add_u64;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;

pub(super) fn accumulate_bool(
    inner: &mut SumState,
    b: &BoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let SumState::Unsigned(acc) = inner else {
        vortex_panic!("expected unsigned sum state for bool input");
    };

    let mask = b.as_ref().validity()?.to_mask(b.as_ref().len(), ctx)?;
    let true_count = match mask.bit_buffer() {
        AllOr::None => return Ok(false),
        AllOr::All => b.to_bit_buffer().true_count() as u64,
        AllOr::Some(validity) => b.to_bit_buffer().bitand(validity).true_count() as u64,
    };

    Ok(checked_add_u64(acc, true_count))
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::arrays::BoolArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::executor::VortexSessionExecute;

    #[test]
    fn sum_bool_all_true() -> VortexResult<()> {
        let arr: BoolArray = [true, true, true].into_iter().collect();
        let result = sum(
            &arr.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_mixed() -> VortexResult<()> {
        let arr: BoolArray = [true, false, true, false, true].into_iter().collect();
        let result = sum(
            &arr.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_all_false() -> VortexResult<()> {
        let arr: BoolArray = [false, false, false].into_iter().collect();
        let result = sum(
            &arr.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_with_nulls() -> VortexResult<()> {
        let arr = BoolArray::from_iter([Some(true), None, Some(true), Some(false)]);
        let result = sum(
            &arr.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(2));
        Ok(())
    }

    #[test]
    fn sum_bool_all_null() -> VortexResult<()> {
        let arr = BoolArray::from_iter([None::<bool>, None, None]);
        let result = sum(
            &arr.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_empty_produces_zero() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype)?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_finish_resets_state() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype)?;

        let batch1: BoolArray = [true, true, false].into_iter().collect();
        acc.accumulate(&batch1.into_array(), &mut ctx)?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().typed_value::<u64>(), Some(2));

        let batch2: BoolArray = [false, true].into_iter().collect();
        acc.accumulate(&batch2.into_array(), &mut ctx)?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().typed_value::<u64>(), Some(1));
        Ok(())
    }

    #[test]
    fn sum_bool_return_dtype() -> VortexResult<()> {
        let dtype = Sum
            .return_dtype(&EmptyOptions, &DType::Bool(Nullability::NonNullable))
            .unwrap();
        assert_eq!(dtype, DType::Primitive(PType::U64, Nullability::Nullable));
        Ok(())
    }

    #[test]
    fn sum_boolean_from_iter() -> VortexResult<()> {
        let arr = BoolArray::from_iter([true, false, false, true]).into_array();
        let result = sum(&arr, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(result.as_primitive().as_::<i32>(), Some(2));
        Ok(())
    }
}
