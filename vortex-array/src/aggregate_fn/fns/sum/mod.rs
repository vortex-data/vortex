// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod constant;
mod decimal;
mod grouped;
mod primitive;
pub(crate) use grouped::PrimitiveGroupedSumEncodingKernel;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use self::bool::accumulate_bool;
use self::constant::multiply_constant;
use self::decimal::accumulate_decimal;
use self::primitive::accumulate_primitive;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::MAX_PRECISION;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

/// Return the sum of an array.
///
/// See [`Sum`] for details.
pub fn sum(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    // Short-circuit using cached array statistics.
    if let Precision::Exact(sum_scalar) = array.statistics().get(Stat::Sum) {
        return Ok(sum_scalar);
    }

    // Compute using Accumulator<Sum>.
    // TODO(ngates): we may want to wrap this three-step dance up into an extension crate maybe.
    let mut acc = Accumulator::try_new(
        Sum,
        NumericalAggregateOpts::default(),
        array.dtype().clone(),
    )?;
    acc.accumulate(array, ctx)?;
    let result = acc.finish()?;

    // Cache the computed sum as a statistic (only if non-null, i.e. no overflow).
    if let Some(val) = result.value().cloned() {
        array.statistics().set(Stat::Sum, Precision::Exact(val));
    }

    Ok(result)
}

/// Sum an array, starting from zero.
///
/// If the sum overflows, a null scalar will be returned.
/// If the array is all-invalid, the sum will be zero.
///
/// NaN handling for float inputs is controlled by [`NumericalAggregateOpts`]: with `skip_nans` (the
/// default) NaN values contribute nothing, otherwise any NaN value poisons the sum to NaN.
#[derive(Clone, Debug)]
pub struct Sum;

impl AggregateFnVTable for Sum {
    type Options = NumericalAggregateOpts;
    type Partial = SumPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.sum")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.serialize()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        NumericalAggregateOpts::deserialize(metadata)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        // When a sum overflows, we return a sum _value_ of null. Therefore, we all return dtypes
        // are nullable.
        use Nullability::Nullable;

        Some(match input_dtype {
            DType::Bool(_) => DType::Primitive(PType::U64, Nullable),
            DType::Primitive(ptype, _) => match ptype {
                PType::U8 | PType::U16 | PType::U32 | PType::U64 => {
                    DType::Primitive(PType::U64, Nullable)
                }
                PType::I8 | PType::I16 | PType::I32 | PType::I64 => {
                    DType::Primitive(PType::I64, Nullable)
                }
                PType::F16 | PType::F32 | PType::F64 => {
                    // Float sums cannot overflow, but all null floats still end up as null
                    DType::Primitive(PType::F64, Nullable)
                }
            },
            DType::Decimal(decimal_dtype, _) => {
                // Both Spark and DataFusion use this heuristic.
                // - https://github.com/apache/spark/blob/fcf636d9eb8d645c24be3db2d599aba2d7e2955a/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/Sum.scala#L66
                // - https://github.com/apache/datafusion/blob/4153adf2c0f6e317ef476febfdc834208bd46622/datafusion/functions-aggregate/src/sum.rs#L188
                let precision = u8::min(MAX_PRECISION, decimal_dtype.precision() + 10);
                DType::Decimal(
                    DecimalDType::new(precision, decimal_dtype.scale()),
                    Nullable,
                )
            }
            // Unsupported types
            _ => return None,
        })
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        let return_dtype = self
            .return_dtype(options, input_dtype)
            .ok_or_else(|| vortex_err!("Unsupported sum dtype: {}", input_dtype))?;
        let initial = make_zero_state(&return_dtype);

        Ok(SumPartial {
            return_dtype,
            current: Some(initial),
            skip_nans: options.skip_nans,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            // A null partial means the sub-accumulator saturated (overflow).
            partial.current = None;
            return Ok(());
        }
        let Some(ref mut inner) = partial.current else {
            return Ok(());
        };
        let saturated = match inner {
            SumState::Unsigned(acc) => {
                let val = other
                    .as_primitive()
                    .typed_value::<u64>()
                    .vortex_expect("checked non-null");
                checked_add_u64(acc, val)
            }
            SumState::Signed(acc) => {
                let val = other
                    .as_primitive()
                    .typed_value::<i64>()
                    .vortex_expect("checked non-null");
                checked_add_i64(acc, val)
            }
            SumState::Float(acc) => {
                let val = other
                    .as_primitive()
                    .typed_value::<f64>()
                    .vortex_expect("checked non-null");
                *acc += val;
                false
            }
            SumState::Decimal { value, dtype } => {
                let val = other
                    .as_decimal()
                    .decimal_value()
                    .vortex_expect("checked non-null");
                match value.checked_add(&val) {
                    Some(r) => {
                        *value = r;
                        !value.fits_in_precision(*dtype)
                    }
                    None => true,
                }
            }
        };
        if saturated {
            partial.current = None;
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(match &partial.current {
            None => Scalar::null(partial.return_dtype.as_nullable()),
            Some(SumState::Unsigned(v)) => Scalar::primitive(*v, Nullability::Nullable),
            Some(SumState::Signed(v)) => Scalar::primitive(*v, Nullability::Nullable),
            Some(SumState::Float(v)) => Scalar::primitive(*v, Nullability::Nullable),
            Some(SumState::Decimal { value, .. }) => {
                let decimal_dtype = *partial
                    .return_dtype
                    .as_decimal_opt()
                    .vortex_expect("return dtype must be decimal");
                Scalar::decimal(*value, decimal_dtype, Nullability::Nullable)
            }
        })
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.current = Some(make_zero_state(&partial.return_dtype));
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        match partial.current.as_ref() {
            None => true,
            Some(SumState::Float(v)) => v.is_nan(),
            Some(_) => false,
        }
    }

    fn try_accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        // NaN-aware shortcircuits only apply to NaN-including float sums; everything else takes
        // the default dispatch path.
        if partial.skip_nans || !matches!(partial.current, Some(SumState::Float(_))) {
            return Ok(false);
        }
        match batch.statistics().get_as::<u64>(Stat::NaNCount) {
            Precision::Exact(0) => {
                // NaN-free batch: the cached NaN-skipping sum (if any) equals the
                // NaN-including sum.
                if let Precision::Exact(sum) = batch.statistics().get(Stat::Sum) {
                    let sum = if sum.dtype() == &partial.return_dtype {
                        sum
                    } else {
                        sum.cast(&partial.return_dtype)?
                    };
                    self.combine_partials(partial, sum)?;
                    return Ok(true);
                }
                Ok(false)
            }
            Precision::Exact(_) => {
                // At least one NaN value: the sum is NaN without scanning the batch.
                if let Some(SumState::Float(acc)) = partial.current.as_mut() {
                    *acc = f64::NAN;
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Constants compute scalar * len and combine via combine_partials.
        if let Columnar::Constant(c) = batch {
            // NaN constants are treated as missing when skipping NaNs.
            if partial.skip_nans && c.scalar().as_primitive_opt().is_some_and(|p| p.is_nan()) {
                return Ok(());
            }
            if let Some(product) = multiply_constant(c.scalar(), c.len(), &partial.return_dtype)? {
                self.combine_partials(partial, product)?;
            }
            return Ok(());
        }

        let skip_nans = partial.skip_nans;
        let mut inner = match partial.current.take() {
            Some(inner) => inner,
            None => return Ok(()),
        };

        let result = match batch {
            Columnar::Canonical(c) => match c {
                Canonical::Primitive(p) => accumulate_primitive(&mut inner, p, ctx, skip_nans),
                Canonical::Bool(b) => accumulate_bool(&mut inner, b, ctx),
                Canonical::Decimal(d) => accumulate_decimal(&mut inner, d, ctx),
                _ => vortex_bail!("Unsupported canonical type for sum: {}", batch.dtype()),
            },
            Columnar::Constant(_) => unreachable!(),
        };

        match result {
            Ok(false) => partial.current = Some(inner),
            Ok(true) => {} // saturated: current stays None
            Err(e) => {
                partial.current = Some(inner);
                return Err(e);
            }
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

/// The group state for a sum aggregate, containing the accumulated value and configuration
/// needed for reset/result without external context.
pub struct SumPartial {
    return_dtype: DType,
    /// The current accumulated state, or `None` if saturated (checked overflow).
    current: Option<SumState>,
    /// Whether NaN values in float inputs are skipped.
    skip_nans: bool,
}

/// The accumulated sum value.
// TODO(ngates): instead of an enum, we should use a Box<dyn State> to avoid dispatcher over the
//  input type every time? Perhaps?
pub enum SumState {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Decimal {
        value: DecimalValue,
        dtype: DecimalDType,
    },
}

fn make_zero_state(return_dtype: &DType) -> SumState {
    match return_dtype {
        DType::Primitive(ptype, _) => match ptype {
            PType::U8 | PType::U16 | PType::U32 | PType::U64 => SumState::Unsigned(0),
            PType::I8 | PType::I16 | PType::I32 | PType::I64 => SumState::Signed(0),
            PType::F16 | PType::F32 | PType::F64 => SumState::Float(0.0),
        },
        DType::Decimal(decimal, _) => SumState::Decimal {
            value: DecimalValue::zero(decimal),
            dtype: *decimal,
        },
        _ => vortex_panic!("Unsupported sum type"),
    }
}

/// Checked add for u64, returning true if overflow occurred.
#[inline(always)]
fn checked_add_u64(acc: &mut u64, val: u64) -> bool {
    match acc.checked_add(val) {
        Some(r) => {
            *acc = r;
            false
        }
        None => true,
    }
}

/// Checked add for i64, returning true if overflow occurred.
#[inline(always)]
fn checked_add_i64(acc: &mut i64, val: i64) -> bool {
    match acc.checked_add(val) {
        Some(r) => {
            *acc = r;
            false
        }
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use num_traits::CheckedAdd;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProvider;
    use crate::scalar::DecimalValue;
    use crate::scalar::NumericOperator;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    /// Sum an array with an initial value (test-only helper).
    fn sum_with_accumulator(array: &ArrayRef, accumulator: &Scalar) -> VortexResult<Scalar> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        if accumulator.is_null() {
            return Ok(accumulator.clone());
        }
        if accumulator.is_zero() == Some(true) {
            return sum(array, &mut ctx);
        }

        let sum_dtype = Stat::Sum.dtype(array.dtype()).ok_or_else(|| {
            vortex_error::vortex_err!("Sum not supported for dtype: {}", array.dtype())
        })?;

        // For non-float types, try statistics short-circuit with accumulator.
        if !matches!(&sum_dtype, DType::Primitive(p, _) if p.is_float())
            && let Precision::Exact(sum_scalar) = array.statistics().get(Stat::Sum)
        {
            return add_scalars(&sum_dtype, &sum_scalar, accumulator);
        }

        // Compute array sum from zero (also caches stats).
        let array_sum = sum(array, &mut ctx)?;

        // Combine with the accumulator.
        add_scalars(&sum_dtype, &array_sum, accumulator)
    }

    /// Add two sum scalars with overflow checking.
    fn add_scalars(sum_dtype: &DType, lhs: &Scalar, rhs: &Scalar) -> VortexResult<Scalar> {
        if lhs.is_null() || rhs.is_null() {
            return Ok(Scalar::null(sum_dtype.as_nullable()));
        }

        Ok(match sum_dtype {
            DType::Primitive(ptype, _) if ptype.is_float() => {
                let lhs_val = f64::try_from(lhs)?;
                let rhs_val = f64::try_from(rhs)?;
                Scalar::primitive(lhs_val + rhs_val, Nullable)
            }
            DType::Primitive(..) => lhs
                .as_primitive()
                .checked_add(&rhs.as_primitive())
                .map(Scalar::from)
                .unwrap_or_else(|| Scalar::null(sum_dtype.as_nullable())),
            DType::Decimal(..) => lhs
                .as_decimal()
                .checked_binary_numeric(&rhs.as_decimal(), NumericOperator::Add)
                .map(Scalar::from)
                .unwrap_or_else(|| Scalar::null(sum_dtype.as_nullable())),
            _ => unreachable!("Sum will always be a decimal or a primitive dtype"),
        })
    }

    // Multi-batch and reset tests

    #[test]
    fn sum_multi_batch() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype)?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(48));
        Ok(())
    }

    #[test]
    fn sum_finish_resets_state() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, NumericalAggregateOpts::default(), dtype)?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().typed_value::<i64>(), Some(30));

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().typed_value::<i64>(), Some(18));
        Ok(())
    }

    // State merge tests (vtable-level)

    #[test]
    fn sum_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = Sum.empty_partial(&NumericalAggregateOpts::default(), &dtype)?;

        let scalar1 = Scalar::primitive(100i64, Nullable);
        Sum.combine_partials(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(50i64, Nullable);
        Sum.combine_partials(&mut state, scalar2)?;

        let result = Sum.to_scalar(&state)?;
        Sum.reset(&mut state);
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(150));
        Ok(())
    }

    // Stats caching test

    #[test]
    fn sum_stats() -> VortexResult<()> {
        let array = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_iter([1, 1, 1]).into_array(),
                PrimitiveArray::from_iter([2, 2, 2]).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .vortex_expect("operation should succeed in test");
        let array = array.into_array();
        // compute sum with accumulator to populate stats
        sum_with_accumulator(&array, &Scalar::primitive(2i64, Nullable))?;

        let sum_without_acc = sum(&array, &mut LEGACY_SESSION.create_execution_ctx())?;
        assert_eq!(sum_without_acc, Scalar::primitive(9i64, Nullable));
        Ok(())
    }

    // Constant float non-multiply test

    #[test]
    fn sum_constant_float_non_multiply() -> VortexResult<()> {
        let acc = -2048669276050936500000000000f64;
        let array = ConstantArray::new(6.1811675e16f64, 25);
        let result = sum_with_accumulator(&array.into_array(), &Scalar::primitive(acc, Nullable))
            .vortex_expect("operation should succeed in test");
        assert_eq!(
            f64::try_from(&result).vortex_expect("operation should succeed in test"),
            -2048669274505644600000000000f64
        );
        Ok(())
    }

    // Grouped sum tests

    fn run_grouped_sum(groups: &ArrayRef, elem_dtype: &DType) -> VortexResult<ArrayRef> {
        let mut acc = GroupedAccumulator::try_new(
            Sum,
            NumericalAggregateOpts::default(),
            elem_dtype.clone(),
        )?;
        acc.accumulate_list(groups, &mut LEGACY_SESSION.create_execution_ctx())?;
        acc.finish()
    }

    #[test]
    fn grouped_sum_fixed_size_list() -> VortexResult<()> {
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
        let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(6i64), Some(15i64)]).into_array();
        assert_arrays_eq!(&result, &expected);
        Ok(())
    }

    #[test]
    fn grouped_sum_with_null_elements() -> VortexResult<()> {
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5), Some(6)])
                .into_array();
        let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(4i64), Some(11i64)]).into_array();
        assert_arrays_eq!(&result, &expected);
        Ok(())
    }

    #[test]
    fn grouped_sum_with_null_group() -> VortexResult<()> {
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9], Validity::NonNullable)
                .into_array();
        let validity = Validity::from_iter([true, false, true]);
        let groups = FixedSizeListArray::try_new(elements, 3, validity, 3)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected =
            PrimitiveArray::from_option_iter([Some(6i64), None, Some(24i64)]).into_array();
        assert_arrays_eq!(&result, &expected);
        Ok(())
    }

    #[test]
    fn grouped_sum_all_null_elements_in_group() -> VortexResult<()> {
        let elements =
            PrimitiveArray::from_option_iter([None::<i32>, None, Some(3), Some(4)]).into_array();
        let groups = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(0i64), Some(7i64)]).into_array();
        assert_arrays_eq!(&result, &expected);
        Ok(())
    }

    #[test]
    fn grouped_sum_bool() -> VortexResult<()> {
        let elements: BoolArray = [true, false, true, true, true, true].into_iter().collect();
        let groups =
            FixedSizeListArray::try_new(elements.into_array(), 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Bool(Nullability::NonNullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(2u64), Some(3u64)]).into_array();
        assert_arrays_eq!(&result, &expected);
        Ok(())
    }

    #[test]
    fn grouped_sum_finish_resets() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc =
            GroupedAccumulator::try_new(Sum, NumericalAggregateOpts::default(), elem_dtype)?;

        let elements1 =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let groups1 = FixedSizeListArray::try_new(elements1, 2, Validity::NonNullable, 2)?;
        acc.accumulate_list(&groups1.into_array(), &mut ctx)?;
        let result1 = acc.finish()?;

        let expected1 = PrimitiveArray::from_option_iter([Some(3i64), Some(7i64)]).into_array();
        assert_arrays_eq!(&result1, &expected1);

        let elements2 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        let groups2 = FixedSizeListArray::try_new(elements2, 2, Validity::NonNullable, 1)?;
        acc.accumulate_list(&groups2.into_array(), &mut ctx)?;
        let result2 = acc.finish()?;

        let expected2 = PrimitiveArray::from_option_iter([Some(30i64)]).into_array();
        assert_arrays_eq!(&result2, &expected2);
        Ok(())
    }

    #[test]
    fn grouped_sum_listview_out_of_order_offsets_with_null_group() -> VortexResult<()> {
        let elements =
            PrimitiveArray::new(buffer![100i32, 200, 300], Validity::NonNullable).into_array();
        let offsets = PrimitiveArray::new(buffer![2i32, 0, 1], Validity::NonNullable).into_array();
        let sizes = PrimitiveArray::new(buffer![1i32, 1, 1], Validity::NonNullable).into_array();
        let validity = Validity::from_iter([true, false, true]);
        let groups = ListViewArray::try_new(elements, offsets, sizes, validity)?.into_array();

        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = run_grouped_sum(&groups, &elem_dtype)?;

        // group 0 -> elements[2..3] = 300; group 1 -> null; group 2 -> elements[1..2] = 200.
        let expected =
            PrimitiveArray::from_option_iter([Some(300i64), None, Some(200i64)]).into_array();
        assert_arrays_eq!(&result, &expected);
        Ok(())
    }

    // Chunked array tests

    #[test]
    fn sum_chunked_floats_with_nulls() -> VortexResult<()> {
        let chunk1 =
            PrimitiveArray::from_option_iter(vec![Some(1.5f64), None, Some(3.2), Some(4.8)]);
        let chunk2 = PrimitiveArray::from_option_iter(vec![Some(2.1f64), Some(5.7), None]);
        let chunk3 = PrimitiveArray::from_option_iter(vec![None, Some(1.0f64), Some(2.5), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )?;

        let result = sum(
            &chunked.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(20.8));
        Ok(())
    }

    #[test]
    fn sum_chunked_floats_all_nulls_is_zero() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None, None]);
        let chunk2 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let result = sum(
            &chunked.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(result, Scalar::primitive(0f64, Nullable));
        Ok(())
    }

    #[test]
    fn sum_chunked_floats_empty_chunks() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter(vec![Some(10.5f64), Some(20.3)]);
        let chunk2 = ConstantArray::new(Scalar::primitive(0f64, Nullable), 0);
        let chunk3 = PrimitiveArray::from_option_iter(vec![Some(5.2f64)]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )?;

        let result = sum(
            &chunked.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(36.0));
        Ok(())
    }

    #[test]
    fn sum_chunked_int_almost_all_null() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter::<u32, _>(vec![Some(1)]);
        let chunk2 = PrimitiveArray::from_option_iter::<u32, _>(vec![None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;

        let result = sum(
            &chunked.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().as_::<u64>(), Some(1));
        Ok(())
    }

    #[test]
    fn sum_chunked_decimals() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let chunk1 = DecimalArray::new(
            buffer![100i32, 100i32, 100i32, 100i32, 100i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk2 = DecimalArray::new(
            buffer![200i32, 200i32, 200i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk3 = DecimalArray::new(buffer![300i32, 300i32], decimal_dtype, Validity::AllValid);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )?;

        let result = sum(
            &chunked.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(1700)))
        );
        Ok(())
    }

    #[test]
    fn sum_chunked_decimals_with_nulls() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let chunk1 = DecimalArray::new(
            buffer![100i32, 100i32, 100i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk2 = DecimalArray::new(
            buffer![0i32, 0i32],
            decimal_dtype,
            Validity::from_iter([false, false]),
        );
        let chunk3 = DecimalArray::new(buffer![200i32, 200i32], decimal_dtype, Validity::AllValid);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )?;

        let result = sum(
            &chunked.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(700)))
        );
        Ok(())
    }

    #[test]
    fn sum_chunked_decimals_large() -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(3, 0);
        let chunk1 = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I16(500),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            1,
        );
        let chunk2 = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I16(600),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            1,
        );
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;

        let result = sum(
            &chunked.into_array(),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(1100)))
        );
        assert_eq!(
            result.dtype(),
            &DType::Decimal(DecimalDType::new(13, 0), Nullable)
        );
        Ok(())
    }
}
