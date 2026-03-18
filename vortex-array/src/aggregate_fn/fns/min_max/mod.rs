// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod decimal;
mod extension;
mod primitive;
mod varbin;

use std::sync::LazyLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use self::bool::accumulate_bool;
use self::decimal::accumulate_decimal;
use self::extension::accumulate_extension;
use self::primitive::accumulate_primitive;
use self::varbin::accumulate_varbinview;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::partial_ord::partial_max;
use crate::partial_ord::partial_min;
use crate::scalar::Scalar;

static NAMES: LazyLock<FieldNames> = LazyLock::new(|| FieldNames::from(["min", "max"]));

/// The minimum and maximum non-null values of an array, or `None` if there are no non-null values.
///
/// The result scalars have the non-nullable version of the array dtype.
/// This will update the stats set of the array as a side effect.
pub fn min_max(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<MinMaxResult>> {
    // Short-circuit using cached array statistics.
    let cached_min = array
        .statistics()
        .get(Stat::Min)
        .and_then(Precision::as_exact);
    let cached_max = array
        .statistics()
        .get(Stat::Max)
        .and_then(Precision::as_exact);
    if let Some((min, max)) = cached_min.zip(cached_max) {
        let non_nullable_dtype = array.dtype().as_nonnullable();
        return Ok(Some(MinMaxResult {
            min: min.cast(&non_nullable_dtype)?,
            max: max.cast(&non_nullable_dtype)?,
        }));
    }

    // Short-circuit for empty arrays or all-null arrays.
    if array.is_empty() || array.valid_count()? == 0 {
        return Ok(None);
    }

    // Short-circuit for unsupported dtypes.
    if MinMax.return_dtype(&EmptyOptions, array.dtype()).is_none() {
        return Ok(None);
    }

    // Compute using Accumulator<MinMax>.
    let mut acc = Accumulator::try_new(MinMax, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result_scalar = acc.finish()?;
    let result = MinMaxResult::from_scalar(result_scalar)?;

    // Cache the computed min/max as statistics.
    if let Some(ref r) = result {
        if let Some(min_value) = r.min.value() {
            array
                .statistics()
                .set(Stat::Min, Precision::Exact(min_value.clone()));
        }
        if let Some(max_value) = r.max.value() {
            array
                .statistics()
                .set(Stat::Max, Precision::Exact(max_value.clone()));
        }
    }

    Ok(result)
}

/// The minimum and maximum non-null values of an array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinMaxResult {
    pub min: Scalar,
    pub max: Scalar,
}

impl MinMaxResult {
    /// Extract a `MinMaxResult` from a struct scalar with `{min, max}` fields.
    pub fn from_scalar(scalar: Scalar) -> VortexResult<Option<Self>> {
        if scalar.is_null() {
            Ok(None)
        } else {
            let min = scalar
                .as_struct()
                .field_by_idx(0)
                .vortex_expect("missing min field");
            let max = scalar
                .as_struct()
                .field_by_idx(1)
                .vortex_expect("missing max field");
            Ok(Some(MinMaxResult { min, max }))
        }
    }
}

/// Compute the min and max of an array.
///
/// Returns a nullable struct scalar `{min: T, max: T}` where `T` is the non-nullable input dtype.
/// The struct is null when the array is empty or all-null.
#[derive(Clone, Debug)]
pub struct MinMax;

/// Partial accumulator state for min/max.
pub struct MinMaxPartial {
    min: Option<Scalar>,
    max: Option<Scalar>,
    element_dtype: DType,
}

impl MinMaxPartial {
    /// Merge a local `MinMaxResult` into this partial state.
    fn merge(&mut self, local: Option<MinMaxResult>) {
        let Some(MinMaxResult { min, max }) = local else {
            return;
        };

        self.min = Some(match self.min.take() {
            Some(current) => partial_min(min, current).vortex_expect("incomparable min scalars"),
            None => min,
        });

        self.max = Some(match self.max.take() {
            Some(current) => partial_max(max, current).vortex_expect("incomparable max scalars"),
            None => max,
        });
    }
}

/// Creates the struct dtype `{min: T, max: T}` (nullable) used for min/max aggregate results.
pub fn make_minmax_dtype(element_dtype: &DType) -> DType {
    DType::Struct(
        StructFields::new(
            NAMES.clone(),
            vec![
                element_dtype.as_nonnullable(),
                element_dtype.as_nonnullable(),
            ],
        ),
        Nullability::Nullable,
    )
}

impl AggregateFnVTable for MinMax {
    type Options = EmptyOptions;
    type Partial = MinMaxPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.min_max")
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        match input_dtype {
            DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(..)
            | DType::Binary(..)
            | DType::Extension(..) => Some(make_minmax_dtype(input_dtype)),
            _ => None,
        }
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(MinMaxPartial {
            min: None,
            max: None,
            element_dtype: input_dtype.clone(),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        let local = MinMaxResult::from_scalar(other)?;
        partial.merge(local);
        Ok(())
    }

    fn flush(&self, partial: &mut Self::Partial) -> VortexResult<Scalar> {
        let dtype = make_minmax_dtype(&partial.element_dtype);
        let result = match (partial.min.take(), partial.max.take()) {
            (Some(min), Some(max)) => Scalar::struct_(dtype, vec![min, max]),
            _ => Scalar::null(dtype),
        };
        Ok(result)
    }

    #[inline]
    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        false
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match batch {
            Columnar::Constant(c) => {
                let scalar = c.scalar();
                if scalar.is_null() {
                    return Ok(());
                }
                // Skip NaN float constants
                if scalar.as_primitive_opt().is_some_and(|p| p.is_nan()) {
                    return Ok(());
                }
                let non_nullable_dtype = scalar.dtype().as_nonnullable();
                let cast = scalar.cast(&non_nullable_dtype)?;
                partial.merge(Some(MinMaxResult {
                    min: cast.clone(),
                    max: cast,
                }));
                Ok(())
            }
            Columnar::Canonical(c) => match c {
                Canonical::Primitive(p) => accumulate_primitive(partial, p),
                Canonical::Bool(b) => accumulate_bool(partial, b),
                Canonical::VarBinView(v) => accumulate_varbinview(partial, v),
                Canonical::Decimal(d) => accumulate_decimal(partial, d),
                Canonical::Extension(e) => accumulate_extension(partial, e, ctx),
                Canonical::Null(_) => Ok(()),
                Canonical::Struct(_) | Canonical::List(_) | Canonical::FixedSizeList(_) => {
                    vortex_bail!("Unsupported canonical type for min_max: {}", batch.dtype())
                }
            },
        }
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: Scalar) -> VortexResult<Scalar> {
        Ok(partial)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::IntoArray as _;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::aggregate_fn::fns::min_max::MinMaxResult;
    use crate::aggregate_fn::fns::min_max::min_max;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::NullArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    #[test]
    fn test_prim_min_max() -> VortexResult<()> {
        let p = PrimitiveArray::new(buffer![1, 2, 3], Validity::NonNullable).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(
            min_max(&p, &mut ctx)?,
            Some(MinMaxResult {
                min: 1.into(),
                max: 3.into()
            })
        );
        Ok(())
    }

    #[test]
    fn test_bool_min_max() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let all_true = BoolArray::new(
            BitBuffer::from([true, true, true].as_slice()),
            Validity::NonNullable,
        )
        .into_array();
        assert_eq!(
            min_max(&all_true, &mut ctx)?,
            Some(MinMaxResult {
                min: true.into(),
                max: true.into()
            })
        );

        let all_false = BoolArray::new(
            BitBuffer::from([false, false, false].as_slice()),
            Validity::NonNullable,
        )
        .into_array();
        assert_eq!(
            min_max(&all_false, &mut ctx)?,
            Some(MinMaxResult {
                min: false.into(),
                max: false.into()
            })
        );

        let mixed = BoolArray::new(
            BitBuffer::from([false, true, false].as_slice()),
            Validity::NonNullable,
        )
        .into_array();
        assert_eq!(
            min_max(&mixed, &mut ctx)?,
            Some(MinMaxResult {
                min: false.into(),
                max: true.into()
            })
        );
        Ok(())
    }

    #[test]
    fn test_null_array() -> VortexResult<()> {
        let p = NullArray::new(1).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(min_max(&p, &mut ctx)?, None);
        Ok(())
    }

    #[test]
    fn test_prim_nan() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![f32::NAN, -f32::NAN, -1.0, 1.0],
            Validity::NonNullable,
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = min_max(&array.into_array(), &mut ctx)?.vortex_expect("should have result");
        assert_eq!(f32::try_from(&result.min)?, -1.0);
        assert_eq!(f32::try_from(&result.max)?, 1.0);
        Ok(())
    }

    #[test]
    fn test_prim_inf() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![f32::INFINITY, f32::NEG_INFINITY, -1.0, 1.0],
            Validity::NonNullable,
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = min_max(&array.into_array(), &mut ctx)?.vortex_expect("should have result");
        assert_eq!(f32::try_from(&result.min)?, f32::NEG_INFINITY);
        assert_eq!(f32::try_from(&result.max)?, f32::INFINITY);
        Ok(())
    }

    #[test]
    fn test_multi_batch() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(MinMax, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20, 5], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;

        let batch2 = PrimitiveArray::new(buffer![3i32, 25], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;

        let result = MinMaxResult::from_scalar(acc.finish()?)?.vortex_expect("should have result");
        assert_eq!(result.min, Scalar::from(3i32));
        assert_eq!(result.max, Scalar::from(25i32));
        Ok(())
    }

    #[test]
    fn test_finish_resets_state() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(MinMax, EmptyOptions, dtype)?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        let result1 = MinMaxResult::from_scalar(acc.finish()?)?.vortex_expect("should have result");
        assert_eq!(result1.min, Scalar::from(10i32));
        assert_eq!(result1.max, Scalar::from(20i32));

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        let result2 = MinMaxResult::from_scalar(acc.finish()?)?.vortex_expect("should have result");
        assert_eq!(result2.min, Scalar::from(3i32));
        assert_eq!(result2.max, Scalar::from(9i32));
        Ok(())
    }

    #[test]
    fn test_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = MinMax.empty_partial(&EmptyOptions, &dtype)?;

        let struct_dtype = crate::aggregate_fn::fns::min_max::make_minmax_dtype(&dtype);
        let scalar1 = Scalar::struct_(
            struct_dtype.clone(),
            vec![Scalar::from(5i32), Scalar::from(15i32)],
        );
        MinMax.combine_partials(&mut state, scalar1)?;

        let scalar2 = Scalar::struct_(struct_dtype, vec![Scalar::from(2i32), Scalar::from(10i32)]);
        MinMax.combine_partials(&mut state, scalar2)?;

        let result = MinMaxResult::from_scalar(MinMax.flush(&mut state)?)?
            .vortex_expect("should have result");
        assert_eq!(result.min, Scalar::from(2i32));
        assert_eq!(result.max, Scalar::from(15i32));
        Ok(())
    }

    #[test]
    fn test_constant_nan() -> VortexResult<()> {
        let scalar = Scalar::primitive(f16::NAN, Nullability::NonNullable);
        let array = ConstantArray::new(scalar, 2).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(min_max(&array, &mut ctx)?, None);
        Ok(())
    }

    #[test]
    fn test_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(5i32), None, Some(1)]);
        let chunk2 = PrimitiveArray::from_option_iter([Some(10i32), Some(3), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = min_max(&chunked.into_array(), &mut ctx)?.vortex_expect("should have result");
        assert_eq!(result.min, Scalar::from(1i32));
        assert_eq!(result.max, Scalar::from(10i32));
        Ok(())
    }

    #[test]
    fn test_all_null() -> VortexResult<()> {
        let p = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(min_max(&p.into_array(), &mut ctx)?, None);
        Ok(())
    }

    #[test]
    fn test_varbin() -> VortexResult<()> {
        let array = VarBinArray::from_iter(
            vec![
                Some("hello world"),
                None,
                Some("hello world this is a long string"),
                None,
            ],
            DType::Utf8(Nullability::Nullable),
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = min_max(&array.into_array(), &mut ctx)?.vortex_expect("should have result");
        assert_eq!(
            result.min,
            Scalar::utf8("hello world", Nullability::NonNullable)
        );
        assert_eq!(
            result.max,
            Scalar::utf8(
                "hello world this is a long string",
                Nullability::NonNullable
            )
        );
        Ok(())
    }

    #[test]
    fn test_decimal() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            buffer![100i32, 2000i32, 200i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([true, false, true]),
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let result = min_max(&decimal.into_array(), &mut ctx)?.vortex_expect("should have result");

        let non_nullable_dtype = DType::Decimal(DecimalDType::new(4, 2), Nullability::NonNullable);
        let expected_min = Scalar::try_new(
            non_nullable_dtype.clone(),
            Some(ScalarValue::from(DecimalValue::from(100i32))),
        )?;
        let expected_max = Scalar::try_new(
            non_nullable_dtype,
            Some(ScalarValue::from(DecimalValue::from(200i32))),
        )?;
        assert_eq!(result.min, expected_min);
        assert_eq!(result.max, expected_max);
        Ok(())
    }

    use crate::dtype::half::f16;

    #[test]
    fn test_bool_with_nulls() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        let result = min_max(
            &BoolArray::from_iter(vec![Some(true), Some(true), None, None]).into_array(),
            &mut ctx,
        )?;
        assert_eq!(
            result,
            Some(MinMaxResult {
                min: Scalar::bool(true, Nullability::NonNullable),
                max: Scalar::bool(true, Nullability::NonNullable),
            })
        );

        let result = min_max(
            &BoolArray::from_iter(vec![None, Some(true), Some(true)]).into_array(),
            &mut ctx,
        )?;
        assert_eq!(
            result,
            Some(MinMaxResult {
                min: Scalar::bool(true, Nullability::NonNullable),
                max: Scalar::bool(true, Nullability::NonNullable),
            })
        );

        let result = min_max(
            &BoolArray::from_iter(vec![None, Some(true), Some(true), None]).into_array(),
            &mut ctx,
        )?;
        assert_eq!(
            result,
            Some(MinMaxResult {
                min: Scalar::bool(true, Nullability::NonNullable),
                max: Scalar::bool(true, Nullability::NonNullable),
            })
        );

        let result = min_max(
            &BoolArray::from_iter(vec![Some(false), Some(false), None, None]).into_array(),
            &mut ctx,
        )?;
        assert_eq!(
            result,
            Some(MinMaxResult {
                min: Scalar::bool(false, Nullability::NonNullable),
                max: Scalar::bool(false, Nullability::NonNullable),
            })
        );
        Ok(())
    }

    #[test]
    fn test_varbin_all_nulls() -> VortexResult<()> {
        let array = VarBinArray::from_iter(
            vec![Option::<&str>::None, None, None],
            DType::Utf8(Nullability::Nullable),
        );
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert_eq!(min_max(&array.into_array(), &mut ctx)?, None);
        Ok(())
    }
}
