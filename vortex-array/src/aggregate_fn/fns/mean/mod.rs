// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::combined::BinaryCombined;
use crate::aggregate_fn::combined::Combined;
use crate::aggregate_fn::combined::CombinedOptions;
use crate::aggregate_fn::combined::PairOptions;
use crate::aggregate_fn::fns::count::Count;
use crate::aggregate_fn::fns::sum::Sum;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;

/// Compute the arithmetic mean of an array.
///
/// See [`Mean`] for details.
pub fn mean(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(
        Mean::combined(),
        PairOptions(
            NumericalAggregateOpts::default(),
            NumericalAggregateOpts::default(),
        ),
        array.dtype().clone(),
    )?;
    acc.accumulate(array, ctx)?;
    acc.finish()
}

/// Compute the arithmetic mean of an array.
///
/// Implemented as `Sum / Count` via [`BinaryCombined`].
///
/// Coercion / return type:
/// - Booleans and primitive numeric types are coerced to `f64` and the result
///   is a nullable `f64`.
/// - Decimals are kept as decimals but not implemented currently
#[derive(Clone, Debug)]
pub struct Mean;

impl Mean {
    pub fn combined() -> Combined<Self> {
        Combined(Mean)
    }
}

impl BinaryCombined for Mean {
    type Left = Sum;
    type Right = Count;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.mean")
    }

    fn left(&self) -> Sum {
        Sum
    }

    fn right(&self) -> Count {
        Count
    }

    fn left_name(&self) -> &'static str {
        "sum"
    }

    fn right_name(&self) -> &'static str {
        "count"
    }

    fn return_dtype(&self, input_dtype: &DType) -> Option<DType> {
        Some(mean_output_dtype(input_dtype)?.with_nullability(Nullability::Nullable))
    }

    fn finalize(&self, sum: ArrayRef, count: ArrayRef) -> VortexResult<ArrayRef> {
        let target = match sum.dtype() {
            DType::Decimal(..) => sum.dtype().with_nullability(Nullability::Nullable),
            _ => DType::Primitive(PType::F64, Nullability::Nullable),
        };
        let sum_cast = sum.cast(target.clone())?;
        let count_cast = count.cast(target.clone())?;
        let mean = sum_cast.binary(count_cast.clone(), Operator::Div)?;
        // Nulls are skipped during accumulation, so an all-null group has a count of zero and
        // the division produces 0/0 = NaN. The mean of an empty group is null (as in SQL), so
        // mask out zero-count entries. This matches `finalize_scalar`.
        let non_empty = count_cast
            .binary(
                ConstantArray::new(Scalar::zero_value(&target), count.len()).into_array(),
                Operator::NotEq,
            )?
            // A null count means a null group; keep it masked out.
            .fill_null(false)?;
        mean.mask(non_empty)
    }

    fn finalize_scalar(&self, left_scalar: Scalar, right_scalar: Scalar) -> VortexResult<Scalar> {
        if let DType::Decimal(..) = left_scalar.dtype() {
            vortex_bail!("mean::finalize_scalar not yet implemented for decimal inputs");
        }

        let target = DType::Primitive(PType::F64, Nullability::Nullable);
        let sum_cast = left_scalar.cast(&target)?;
        let count_cast = right_scalar.cast(&target)?;

        let sum = sum_cast.as_primitive().typed_value::<f64>();
        let count = count_cast.as_primitive().typed_value::<f64>();
        let value = match (sum, count) {
            // A `None` sum means the sum overflowed; a count of zero means an all-null (empty)
            // input. Both are null, matching the array `finalize` path.
            (None, _) | (_, None) | (_, Some(0.0)) => return Ok(Scalar::null(target)),
            (Some(s), Some(c)) => s / c,
        };
        Ok(Scalar::primitive(value, Nullability::Nullable))
    }

    fn serialize(&self, _options: &CombinedOptions<Self>) -> VortexResult<Option<Vec<u8>>> {
        unimplemented!("mean is not yet serializable");
    }

    fn coerce_args(
        &self,
        _options: &PairOptions<
            <Sum as AggregateFnVTable>::Options,
            <Count as AggregateFnVTable>::Options,
        >,
        input_dtype: &DType,
    ) -> VortexResult<DType> {
        // Advisory hint for query planners: where possible, cast input to the
        // type we're going to compute the mean in.
        Ok(coerced_input_dtype(input_dtype).unwrap_or_else(|| input_dtype.clone()))
    }
}

/// Hint for callers: what to cast the input to before accumulation.
///
/// - Bool stays as bool — `Sum` has a native bool path and bool → f64 isn't
///   currently a direct cast in vortex.
/// - Primitive numerics → `f64` so the sum and finalize work without overflow.
fn coerced_input_dtype(input_dtype: &DType) -> Option<DType> {
    match input_dtype {
        DType::Bool(_) => Some(input_dtype.clone()),
        DType::Primitive(_, n) => Some(DType::Primitive(PType::F64, *n)),
        DType::Decimal(..) => {
            unimplemented!("mean is not implemented for decimals yet")
        }
        _ => None,
    }
}

fn mean_output_dtype(input_dtype: &DType) -> Option<DType> {
    match input_dtype {
        DType::Bool(_) | DType::Primitive(..) => {
            Some(DType::Primitive(PType::F64, Nullability::Nullable))
        }
        DType::Decimal(..) => {
            unimplemented!("mean for decimals is not yet implemented");
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::validity::Validity;

    #[test]
    fn mean_all_valid() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0, 4.0, 5.0], Validity::NonNullable)
            .into_array();
        let mut ctx = array_session().create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_with_nulls() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(2.0f64), None, Some(4.0)]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_integers() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![10i32, 20, 30], Validity::NonNullable).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(20.0));
        Ok(())
    }

    #[test]
    fn mean_bool() -> VortexResult<()> {
        let array: BoolArray = [true, false, true, true].into_iter().collect();
        let mut ctx = array_session().create_execution_ctx();
        let result = mean(&array.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(0.75));
        Ok(())
    }

    #[test]
    fn mean_constant_non_null() -> VortexResult<()> {
        let array = ConstantArray::new(5.0f64, 4);
        let mut ctx = array_session().create_execution_ctx();
        let result = mean(&array.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(5.0));
        Ok(())
    }

    #[test]
    fn mean_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(1.0f64), None, Some(3.0)]);
        let chunk2 = PrimitiveArray::from_option_iter([Some(5.0f64), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = array_session().create_execution_ctx();
        let result = mean(&chunked.into_array(), &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    #[test]
    fn mean_skips_nans_by_default() -> VortexResult<()> {
        // NaNs are excluded from both the sum and the count.
        let array =
            PrimitiveArray::new(buffer![1.0f64, f64::NAN, 3.0], Validity::NonNullable).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(2.0));
        Ok(())
    }

    #[test]
    fn mean_with_nan_not_skipping() -> VortexResult<()> {
        let array =
            PrimitiveArray::new(buffer![1.0f64, f64::NAN, 3.0], Validity::NonNullable).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let keep_nans = NumericalAggregateOpts::include_nans();
        let mut acc = Accumulator::try_new(
            Mean::combined(),
            PairOptions(keep_nans, keep_nans),
            array.dtype().clone(),
        )?;
        acc.accumulate(&array, &mut ctx)?;
        let result = acc.finish()?;
        assert!(result.as_primitive().as_::<f64>().is_some_and(f64::is_nan));
        Ok(())
    }

    #[test]
    fn mean_all_null_returns_null() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter::<f64, _>([None, None, None]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let result = mean(&array, &mut ctx)?;
        assert_eq!(result.as_primitive().as_::<f64>(), None);
        Ok(())
    }

    #[test]
    fn mean_multi_batch() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(
            Mean::combined(),
            PairOptions(
                NumericalAggregateOpts::default(),
                NumericalAggregateOpts::default(),
            ),
            dtype,
        )?;

        let batch1 =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::new(buffer![4.0f64, 5.0], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(3.0));
        Ok(())
    }

    /// Groups exercised by both finalize paths, under the default skip-NaN options. NaNs and nulls
    /// are skipped during accumulation, so a group with no non-null, non-NaN values is empty and
    /// its mean is null.
    fn mean_cases() -> Vec<(Vec<Option<f64>>, Option<f64>)> {
        vec![
            (vec![Some(f64::NAN), Some(1.0), None], Some(1.0)),
            (vec![Some(f64::NAN), Some(1.0), Some(3.0)], Some(2.0)),
            (vec![None, None, Some(f64::NAN)], None),
            (vec![None, None, None], None),
            (vec![Some(1.0), Some(2.0), Some(3.0)], Some(2.0)),
        ]
    }

    #[test]
    fn mean_via_combined_partials() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        for (case, (group, expected)) in mean_cases().into_iter().enumerate() {
            let mut acc = Accumulator::try_new(
                Mean::combined(),
                PairOptions(
                    NumericalAggregateOpts::default(),
                    NumericalAggregateOpts::default(),
                ),
                DType::Primitive(PType::F64, Nullability::Nullable),
            )?;
            // Two batches per group so the result goes through partial combination and
            // `finalize_scalar`.
            let (head, tail) = group.split_at(2);
            let head = PrimitiveArray::from_option_iter(head.iter().copied()).into_array();
            let tail = PrimitiveArray::from_option_iter(tail.iter().copied()).into_array();
            acc.accumulate(&head, &mut ctx)?;
            acc.accumulate(&tail, &mut ctx)?;
            let result = acc.finish()?;
            assert_eq!(result.as_primitive().as_::<f64>(), expected, "case {case}");
        }
        Ok(())
    }

    #[test]
    fn mean_via_grouped_finalize() -> VortexResult<()> {
        let cases = mean_cases();
        let elements = PrimitiveArray::from_option_iter(
            cases.iter().flat_map(|(group, _)| group.iter().copied()),
        )
        .into_array();
        let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, cases.len())?;

        let mut acc = GroupedAccumulator::try_new(
            Mean::combined(),
            PairOptions(
                NumericalAggregateOpts::default(),
                NumericalAggregateOpts::default(),
            ),
            DType::Primitive(PType::F64, Nullability::Nullable),
        )?;
        let mut ctx = array_session().create_execution_ctx();
        acc.accumulate_list(&groups.into_array(), &mut ctx)?;
        let result = acc.finish()?;

        for (case, (_, expected)) in cases.into_iter().enumerate() {
            let actual = result.execute_scalar(case, &mut ctx)?;
            assert_eq!(actual.as_primitive().as_::<f64>(), expected, "case {case}");
        }
        Ok(())
    }
}
