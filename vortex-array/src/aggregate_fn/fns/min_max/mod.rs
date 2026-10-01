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
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use self::bool::accumulate_bool;
use self::decimal::accumulate_decimal;
use self::extension::accumulate_extension;
use self::primitive::accumulate_primitive;
use self::varbin::accumulate_varbinview;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::half::f16;
use crate::expr::stats::Precision;
use crate::partial_ord::partial_max;
use crate::partial_ord::partial_min;
use crate::scalar::Scalar;

static NAMES: LazyLock<FieldNames> = LazyLock::new(|| FieldNames::from(["min", "max"]));

/// The minimum and maximum non-null values of an array, or `None` if there are no non-null values.
///
/// With `skip_nans`, NaN values are ignored. Otherwise, any NaN poisons both extrema to NaN.
/// An exact cached NaN count can avoid scanning for NaNs. The result scalars have the
/// non-nullable version of the array dtype, and computed extrema are cached under both functions.
pub fn min_max(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
    options: NumericalAggregateOpts,
) -> VortexResult<Option<MinMaxResult>> {
    let mut cached_options = options;
    if !options.skip_nans && array.dtype().is_float() {
        match ctx.aggregate_result_as::<u64>(array, &NanCount.bind(EmptyOptions))? {
            Precision::Exact(0) => cached_options = NumericalAggregateOpts::skip_nans(),
            Precision::Exact(_) => {
                let result = Some(nan_minmax_result(array.dtype()));
                cache_min_max_result(array, options, result.as_ref(), ctx)?;
                return Ok(result);
            }
            _ => {}
        }
    }

    let mut cached_extrema = ctx
        .aggregate_result(array, &Min.bind(cached_options))
        .as_exact()
        .zip(
            ctx.aggregate_result(array, &Max.bind(cached_options))
                .as_exact(),
        );

    // These logical types cannot contain NaNs, so either option computes the same extrema.
    // Extensions keep their own options because their storage type can contain NaNs.
    let nan_options_are_equivalent = match array.dtype() {
        DType::Primitive(ptype, _) => !ptype.is_float(),
        DType::Bool(_) | DType::Decimal(..) | DType::Utf8(_) | DType::Binary(_) => true,
        _ => false,
    };
    if cached_extrema.is_none() && nan_options_are_equivalent {
        let other_options = if cached_options.skip_nans {
            NumericalAggregateOpts::include_nans()
        } else {
            NumericalAggregateOpts::skip_nans()
        };
        cached_extrema = ctx
            .aggregate_result(array, &Min.bind(other_options))
            .as_exact()
            .zip(
                ctx.aggregate_result(array, &Max.bind(other_options))
                    .as_exact(),
            );
    }

    if let Some((min, max)) = cached_extrema {
        if min.is_null() || max.is_null() {
            return Ok(None);
        }

        let non_nullable_dtype = array.dtype().as_nonnullable();
        return Ok(Some(MinMaxResult {
            min: min.cast(&non_nullable_dtype)?,
            max: max.cast(&non_nullable_dtype)?,
        }));
    }

    if array.is_empty() || array.valid_count(ctx)? == 0 {
        return Ok(None);
    }

    if !supports_min_max(array.dtype()) {
        return Ok(None);
    }

    let result =
        MinMaxResult::from_scalar(ctx.compute_aggregate_result(array, &MinMax.bind(options))?)?;
    cache_min_max_result(array, options, result.as_ref(), ctx)?;

    Ok(result)
}

fn cache_min_max_result(
    array: &ArrayRef,
    options: NumericalAggregateOpts,
    result: Option<&MinMaxResult>,
    ctx: &ExecutionCtx,
) -> VortexResult<()> {
    let dtype = array.dtype().as_nullable();
    let (min, max) = match result {
        Some(result) => (result.min.cast(&dtype)?, result.max.cast(&dtype)?),
        None => (Scalar::null(dtype.clone()), Scalar::null(dtype)),
    };
    ctx.insert_aggregate_result(array, Min.bind(options), Precision::Exact(min))?;
    ctx.insert_aggregate_result(array, Max.bind(options), Precision::Exact(max))?;

    Ok(())
}

/// A `{min: NaN, max: NaN}` result for a poisoned NaN-including min/max over `dtype`.
fn nan_minmax_result(dtype: &DType) -> MinMaxResult {
    let nan = nan_scalar(dtype);
    MinMaxResult {
        min: nan.clone(),
        max: nan,
    }
}

/// A non-nullable NaN scalar of the float `dtype`.
pub(crate) fn nan_scalar(dtype: &DType) -> Scalar {
    match dtype.as_ptype() {
        PType::F16 => Scalar::primitive(f16::NAN, Nullability::NonNullable),
        PType::F32 => Scalar::primitive(f32::NAN, Nullability::NonNullable),
        PType::F64 => Scalar::primitive(f64::NAN, Nullability::NonNullable),
        _ => vortex_panic!("NaN scalar requested for non-float dtype {dtype}"),
    }
}

/// Whether a scalar holds a primitive float NaN value.
pub(crate) fn scalar_is_nan(scalar: &Scalar) -> bool {
    if !scalar.dtype().is_float() {
        return false;
    }

    scalar.as_primitive_opt().is_some_and(|p| p.is_nan())
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
///
/// NaN handling for float inputs is controlled by [`NumericalAggregateOpts`]: with `skip_nans` (the
/// default) NaN values are ignored, otherwise any NaN value poisons both extrema to NaN.
#[derive(Clone, Debug)]
pub struct MinMax;

/// Partial accumulator state for min/max.
pub struct MinMaxPartial {
    min: Option<Scalar>,
    max: Option<Scalar>,
}

impl MinMaxPartial {
    /// Merge a local `MinMaxResult` into this partial state.
    fn merge(
        &mut self,
        args: AggregateArgs<'_, NumericalAggregateOpts>,
        local: Option<MinMaxResult>,
    ) {
        let Some(MinMaxResult { min, max }) = local else {
            return;
        };

        // NaN scalars are incomparable under `partial_min`/`partial_max`, so they are handled
        // explicitly: a NaN extremum poisons the partial state when NaNs participate, and is
        // dropped when they are skipped.
        if scalar_is_nan(&min) || scalar_is_nan(&max) || self.is_poisoned() {
            if !args.options.skip_nans {
                self.poison(args);
            }
            return;
        }

        self.min = Some(match self.min.take() {
            Some(current) => partial_min(min, current).vortex_expect("incomparable min scalars"),
            None => min,
        });

        self.max = Some(match self.max.take() {
            Some(current) => partial_max(max, current).vortex_expect("incomparable max scalars"),
            None => max,
        });
    }

    /// Poison the partial state to `{min: NaN, max: NaN}`.
    fn poison(&mut self, args: AggregateArgs<'_, NumericalAggregateOpts>) {
        let nan = nan_scalar(args.dtype);
        self.min = Some(nan.clone());
        self.max = Some(nan);
    }

    /// Whether the partial state is poisoned to NaN.
    fn is_poisoned(&self) -> bool {
        self.min.as_ref().is_some_and(scalar_is_nan)
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

fn minmax_supported_dtype(input_dtype: &DType) -> bool {
    match input_dtype {
        DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(..)
        | DType::Binary(..)
        | DType::Extension(..) => true,
        DType::List(element_dtype, _) => minmax_supported_dtype(element_dtype),
        DType::FixedSizeList(element_dtype, ..) => minmax_supported_dtype(element_dtype),
        _ => false,
    }
}

/// Returns whether [`min_max`] can compute extrema for this logical dtype.
///
/// Lists have a result dtype for expression lowering but do not have comparison kernels.
/// Extensions can use a registered kernel or delegate to their storage array.
pub fn supports_min_max(input_dtype: &DType) -> bool {
    matches!(
        input_dtype,
        DType::Bool(_)
            | DType::Primitive(..)
            | DType::Decimal(..)
            | DType::Utf8(..)
            | DType::Binary(..)
            | DType::Extension(..)
    )
}

impl AggregateFnVTable for MinMax {
    type Options = NumericalAggregateOpts;
    type Partial = MinMaxPartial;

    fn is_representation_invariant(&self, _options: &Self::Options) -> bool {
        true
    }

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.min_max");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        minmax_supported_dtype(input_dtype).then(|| make_minmax_dtype(input_dtype))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(MinMaxPartial {
            min: None,
            max: None,
        })
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        let mut partial = MinMaxPartial {
            min: None,
            max: None,
        };
        // `merge` normalizes the parsed extrema: nulls stay empty and NaNs poison or drop.
        partial.merge(args, MinMaxResult::from_scalar(scalar)?);
        Ok(partial)
    }

    fn partial_from_result(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        result: Scalar,
    ) -> VortexResult<Option<Self::Partial>> {
        self.partial_from_scalar(args, result).map(Some)
    }

    fn merge_partials(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        mut first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        if let (Some(min), Some(max)) = (second.min, second.max) {
            first.merge(args, Some(MinMaxResult { min, max }));
        }
        Ok(first)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        let dtype = args.partial_dtype.clone();
        Ok(match (&partial.min, &partial.max) {
            (Some(min), Some(max)) => Scalar::struct_(dtype, vec![min.clone(), max.clone()]),
            _ => Scalar::null(dtype),
        })
    }

    #[inline]
    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        // A poisoned NaN-including min/max is fully determined.
        partial.is_poisoned()
    }

    fn try_accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        // NaN-aware shortcircuits only apply to NaN-including float min/max; everything else
        // takes the default dispatch path.
        if args.options.skip_nans || !args.dtype.is_float() {
            return Ok(false);
        }
        match ctx.aggregate_result_as::<u64>(batch, &NanCount.bind(EmptyOptions))? {
            Precision::Exact(0) => {
                // NaN-free batch: the cached NaN-skipping extrema (if any) are valid.
                let cached_min = ctx
                    .aggregate_result(batch, &Min.bind(NumericalAggregateOpts::skip_nans()))
                    .as_exact();
                let cached_max = ctx
                    .aggregate_result(batch, &Max.bind(NumericalAggregateOpts::skip_nans()))
                    .as_exact();
                if let Some((min, max)) = cached_min.zip(cached_max) {
                    // Null extrema represent a batch with no participating values.
                    if min.is_null() || max.is_null() {
                        return Ok(true);
                    }

                    // Cached float stats carry the (possibly nullable) array dtype; `to_scalar`
                    // builds a struct with non-nullable fields, so normalise here.
                    let non_nullable_dtype = args.dtype.as_nonnullable();
                    partial.merge(
                        args,
                        Some(MinMaxResult {
                            min: min.cast(&non_nullable_dtype)?,
                            max: max.cast(&non_nullable_dtype)?,
                        }),
                    );
                    return Ok(true);
                }
                Ok(false)
            }
            Precision::Exact(_) => {
                // At least one NaN value poisons both extrema without scanning the batch.
                partial.poison(args);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
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
                // NaN float constants are skipped or poison the extrema, per the options.
                if scalar_is_nan(scalar) {
                    if !args.options.skip_nans {
                        partial.poison(args);
                    }
                    return Ok(());
                }
                let non_nullable_dtype = scalar.dtype().as_nonnullable();
                let cast = scalar.cast(&non_nullable_dtype)?;
                partial.merge(
                    args,
                    Some(MinMaxResult {
                        min: cast.clone(),
                        max: cast,
                    }),
                );
                Ok(())
            }
            Columnar::Canonical(c) => match c {
                Canonical::Primitive(p) => accumulate_primitive(args, partial, p, ctx),
                Canonical::Bool(b) => accumulate_bool(args, partial, b, ctx),
                Canonical::VarBinView(v) => accumulate_varbinview(args, partial, v, ctx),
                Canonical::Decimal(d) => accumulate_decimal(args, partial, d, ctx),
                Canonical::Extension(e) => accumulate_extension(args, partial, e, ctx),
                Canonical::Null(_) => Ok(()),
                Canonical::Union(_) => {
                    vortex_bail!("TODO(connor)[Union]: implement min_max for Union arrays")
                }
                Canonical::Struct(_)
                | Canonical::List(_)
                | Canonical::Map(_)
                | Canonical::FixedSizeList(_)
                | Canonical::Variant(_) => {
                    vortex_bail!("Unsupported canonical type for min_max: {}", batch.dtype())
                }
            },
        }
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(args, partial)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray as _;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateDTypes;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::max::Max;
    use crate::aggregate_fn::fns::min::Min;
    use crate::aggregate_fn::fns::min_max::MinMax;
    use crate::aggregate_fn::fns::min_max::MinMaxPartial;
    use crate::aggregate_fn::fns::min_max::MinMaxResult;
    use crate::aggregate_fn::fns::min_max::make_minmax_dtype;
    use crate::aggregate_fn::fns::min_max::min_max;
    use crate::aggregate_fn::fns::nan_count::NanCount;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListArray;
    use crate::arrays::NullArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::stats::Precision;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    #[test]
    fn test_prim_min_max() -> VortexResult<()> {
        let p = PrimitiveArray::new(buffer![1, 2, 3], Validity::NonNullable).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            min_max(&p, &mut ctx, NumericalAggregateOpts::default())?,
            Some(MinMaxResult {
                min: 1.into(),
                max: 3.into()
            })
        );
        Ok(())
    }

    #[test]
    fn test_prim_min_max_multiple_null_runs() -> VortexResult<()> {
        // Several disjoint valid runs separated by nulls exercise the per-run fold; the extrema
        // (min 1, max 9) fall in different runs.
        let p = PrimitiveArray::from_option_iter([
            Some(5i32),
            Some(3),
            None,
            None,
            Some(9),
            None,
            Some(1),
            Some(7),
        ])
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            min_max(&p, &mut ctx, NumericalAggregateOpts::default())?,
            Some(MinMaxResult {
                min: 1.into(),
                max: 9.into()
            })
        );
        Ok(())
    }

    #[test]
    fn test_bool_min_max() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let all_true = BoolArray::new(
            BitBuffer::from([true, true, true].as_slice()),
            Validity::NonNullable,
        )
        .into_array();
        assert_eq!(
            min_max(&all_true, &mut ctx, NumericalAggregateOpts::default())?,
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
            min_max(&all_false, &mut ctx, NumericalAggregateOpts::default())?,
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
            min_max(&mixed, &mut ctx, NumericalAggregateOpts::default())?,
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
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            min_max(&p, &mut ctx, NumericalAggregateOpts::default())?,
            None
        );
        Ok(())
    }

    #[test]
    fn test_prim_nan() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![f32::NAN, -f32::NAN, -1.0, 1.0],
            Validity::NonNullable,
        );
        let mut ctx = SESSION.create_execution_ctx();
        let result = min_max(
            &array.into_array(),
            &mut ctx,
            NumericalAggregateOpts::default(),
        )?
        .vortex_expect("should have result");
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
        let mut ctx = SESSION.create_execution_ctx();
        let result = min_max(
            &array.into_array(),
            &mut ctx,
            NumericalAggregateOpts::default(),
        )?
        .vortex_expect("should have result");
        assert_eq!(f32::try_from(&result.min)?, f32::NEG_INFINITY);
        assert_eq!(f32::try_from(&result.max)?, f32::INFINITY);
        Ok(())
    }

    #[test]
    fn test_multi_batch() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(MinMax, NumericalAggregateOpts::default(), dtype)?;

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
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(MinMax, NumericalAggregateOpts::default(), dtype)?;

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
        let options = NumericalAggregateOpts::default();
        let owned = AggregateDTypes::try_new(&MinMax, &options, dtype)?;
        let args = owned.args(&options);
        let partial_of = |min: i32, max: i32| MinMaxPartial {
            min: Some(Scalar::from(min)),
            max: Some(Scalar::from(max)),
        };

        let state = MinMax.merge_partials(args, partial_of(5, 15), partial_of(2, 10))?;

        let result = MinMaxResult::from_scalar(MinMax.to_scalar(args, &state)?)?
            .vortex_expect("should have result");
        assert_eq!(result.min, Scalar::from(2i32));
        assert_eq!(result.max, Scalar::from(15i32));
        Ok(())
    }

    #[test]
    fn test_constant_nan() -> VortexResult<()> {
        let scalar = Scalar::primitive(f16::NAN, Nullability::NonNullable);
        let array = ConstantArray::new(scalar, 2).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            min_max(&array, &mut ctx, NumericalAggregateOpts::default())?,
            None
        );
        Ok(())
    }

    const KEEP_NANS: NumericalAggregateOpts = NumericalAggregateOpts::include_nans();

    fn assert_poisoned(result: Option<MinMaxResult>) -> VortexResult<()> {
        let result = result.vortex_expect("should have result");
        assert!(f64::try_from(&result.min.cast(&result.min.dtype().as_nullable())?)?.is_nan());
        assert!(f64::try_from(&result.max.cast(&result.max.dtype().as_nullable())?)?.is_nan());
        Ok(())
    }

    #[test]
    fn test_prim_nan_not_skipping() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![f32::NAN, -f32::NAN, -1.0, 1.0],
            Validity::NonNullable,
        )
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_poisoned(min_max(&array, &mut ctx, KEEP_NANS)?)
    }

    #[test]
    fn test_prim_no_nan_not_skipping() -> VortexResult<()> {
        let array =
            PrimitiveArray::new(buffer![3.0f32, -1.0, 1.0], Validity::NonNullable).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = min_max(&array, &mut ctx, KEEP_NANS)?.vortex_expect("should have result");
        assert_eq!(f32::try_from(&result.min)?, -1.0);
        assert_eq!(f32::try_from(&result.max)?, 3.0);
        Ok(())
    }

    #[test]
    fn test_constant_nan_not_skipping() -> VortexResult<()> {
        let scalar = Scalar::primitive(f64::NAN, Nullability::NonNullable);
        let array = ConstantArray::new(scalar, 2).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        assert_poisoned(min_max(&array, &mut ctx, KEEP_NANS)?)
    }

    #[test]
    fn test_not_skipping_shortcircuits_on_exact_nan_count_stat() -> VortexResult<()> {
        // The array has no NaNs; a planted exact NaNCount stat proves the poisoning came from
        // the stat rather than a scan.
        let array =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        array.aggregations().insert_result(
            NanCount.bind(EmptyOptions),
            Precision::Exact(Scalar::from(2u64)),
        )?;
        let mut ctx = SESSION.create_execution_ctx();
        assert_poisoned(min_max(&array, &mut ctx, KEEP_NANS)?)
    }

    #[test]
    fn test_not_skipping_uses_cached_stats_when_nan_free() -> VortexResult<()> {
        // With an exact NaNCount of zero, the planted exact Min/Max stats are usable as-is.
        let array =
            PrimitiveArray::new(buffer![1.0f64, 2.0, 3.0], Validity::NonNullable).into_array();
        array.aggregations().insert_result(
            NanCount.bind(EmptyOptions),
            Precision::Exact(Scalar::from(0u64)),
        )?;
        array.aggregations().insert_result(
            Min.bind(NumericalAggregateOpts::skip_nans()),
            Precision::Exact(Scalar::primitive(-10.0f64, Nullability::Nullable)),
        )?;
        array.aggregations().insert_result(
            Max.bind(NumericalAggregateOpts::skip_nans()),
            Precision::Exact(Scalar::primitive(10.0f64, Nullability::Nullable)),
        )?;
        let mut ctx = SESSION.create_execution_ctx();
        let result = min_max(&array, &mut ctx, KEEP_NANS)?.vortex_expect("should have result");
        assert_eq!(f64::try_from(&result.min)?, -10.0);
        assert_eq!(f64::try_from(&result.max)?, 10.0);
        Ok(())
    }

    #[test]
    fn test_accumulator_nan_including_nullable_cached_stats() -> VortexResult<()> {
        // A nullable float array's cached Min/Max stats are reconstructed as nullable scalars.
        // The NaN-including accumulator shortcircuit must normalise them to the non-nullable
        // struct field dtype before building the result scalar.
        let mut ctx = SESSION.create_execution_ctx();
        let array =
            PrimitiveArray::from_option_iter([Some(1.0f64), Some(2.0), Some(3.0)]).into_array();
        array.aggregations().insert_result(
            NanCount.bind(EmptyOptions),
            Precision::Exact(Scalar::from(0u64)),
        )?;
        array.aggregations().insert_result(
            Min.bind(NumericalAggregateOpts::skip_nans()),
            Precision::Exact(Scalar::primitive(1.0f64, Nullability::Nullable)),
        )?;
        array.aggregations().insert_result(
            Max.bind(NumericalAggregateOpts::skip_nans()),
            Precision::Exact(Scalar::primitive(3.0f64, Nullability::Nullable)),
        )?;

        let mut acc = Accumulator::try_new(MinMax, KEEP_NANS, array.dtype().clone())?;
        acc.accumulate(&array, &mut ctx)?;
        let result = MinMaxResult::from_scalar(acc.finish()?)?.vortex_expect("should have result");
        assert_eq!(f64::try_from(&result.min)?, 1.0);
        assert_eq!(f64::try_from(&result.max)?, 3.0);
        Ok(())
    }

    #[rstest]
    #[case::skipping_first(NumericalAggregateOpts::skip_nans())]
    #[case::including_first(NumericalAggregateOpts::include_nans())]
    fn non_float_extrema_reuse_equivalent_options(
        #[case] cached_options: NumericalAggregateOpts,
    ) -> VortexResult<()> {
        let array = buffer![1i32, 2, 3].into_array();
        let aggregations = array.aggregations();
        aggregations.insert_result(
            Min.bind(cached_options),
            Precision::Exact(Scalar::primitive(-10i32, Nullability::Nullable)),
        )?;
        aggregations.insert_result(
            Max.bind(cached_options),
            Precision::Exact(Scalar::primitive(10i32, Nullability::Nullable)),
        )?;
        let requested_options = if cached_options.skip_nans {
            NumericalAggregateOpts::include_nans()
        } else {
            NumericalAggregateOpts::skip_nans()
        };

        let mut ctx = SESSION.create_execution_ctx();
        let result = min_max(&array, &mut ctx, requested_options)?;
        assert_eq!(
            result,
            Some(MinMaxResult {
                min: Scalar::from(-10i32),
                max: Scalar::from(10i32),
            })
        );
        assert!(
            aggregations
                .get_result(&Min.bind(requested_options))
                .is_absent()
        );
        assert!(
            aggregations
                .get_result(&Max.bind(requested_options))
                .is_absent()
        );

        Ok(())
    }

    #[rstest]
    #[case::cached_first(true)]
    #[case::including_first(false)]
    fn cached_null_extrema_are_empty_partial(#[case] cached_first: bool) -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([None::<f64>, None]).into_array();
        let aggregations = array.aggregations();
        let mut ctx = SESSION.create_execution_ctx();
        let including = MinMax.bind(NumericalAggregateOpts::include_nans());

        if !cached_first {
            assert!(aggregations.compute_result(&including, &mut ctx)?.is_null());
        }

        aggregations.compute_result(&Min.bind(NumericalAggregateOpts::skip_nans()), &mut ctx)?;
        aggregations.compute_result(&Max.bind(NumericalAggregateOpts::skip_nans()), &mut ctx)?;
        aggregations.compute_result(&NanCount.bind(EmptyOptions), &mut ctx)?;

        assert!(aggregations.compute_result(&including, &mut ctx)?.is_null());
        Ok(())
    }

    #[test]
    fn test_multi_batch_nan_poisoning() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(MinMax, KEEP_NANS, dtype)?;

        let batch1 = PrimitiveArray::new(buffer![1.0f64, 2.0], Validity::NonNullable).into_array();
        acc.accumulate(&batch1, &mut ctx)?;
        assert!(!acc.is_saturated());

        let batch2 = PrimitiveArray::new(buffer![f64::NAN], Validity::NonNullable).into_array();
        acc.accumulate(&batch2, &mut ctx)?;
        assert!(acc.is_saturated());

        assert_poisoned(MinMaxResult::from_scalar(acc.finish()?)?)
    }

    #[test]
    fn test_chunked() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter([Some(5i32), None, Some(1)]);
        let chunk2 = PrimitiveArray::from_option_iter([Some(10i32), Some(3), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let mut ctx = SESSION.create_execution_ctx();
        let result = min_max(
            &chunked.into_array(),
            &mut ctx,
            NumericalAggregateOpts::default(),
        )?
        .vortex_expect("should have result");
        assert_eq!(result.min, Scalar::from(1i32));
        assert_eq!(result.max, Scalar::from(10i32));
        Ok(())
    }

    #[test]
    fn test_all_null() -> VortexResult<()> {
        let p = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]);
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            min_max(&p.into_array(), &mut ctx, NumericalAggregateOpts::default())?,
            None
        );
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
        let mut ctx = SESSION.create_execution_ctx();
        let result = min_max(
            &array.into_array(),
            &mut ctx,
            NumericalAggregateOpts::default(),
        )?
        .vortex_expect("should have result");
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
        let mut ctx = SESSION.create_execution_ctx();
        let result = min_max(
            &decimal.into_array(),
            &mut ctx,
            NumericalAggregateOpts::default(),
        )?
        .vortex_expect("should have result");

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

    #[test]
    fn list_and_fixed_size_list_return_dtype() {
        let element_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let list_dtype = DType::List(Arc::new(element_dtype.clone()), Nullability::Nullable);
        let fixed_size_list_dtype =
            DType::FixedSizeList(Arc::new(element_dtype), 1, Nullability::Nullable);

        assert_eq!(
            MinMax.return_dtype(&NumericalAggregateOpts::default(), &list_dtype),
            Some(make_minmax_dtype(&list_dtype))
        );
        assert_eq!(
            MinMax.return_dtype(&NumericalAggregateOpts::default(), &fixed_size_list_dtype),
            Some(make_minmax_dtype(&fixed_size_list_dtype))
        );
    }

    #[test]
    fn list_and_fixed_size_list_min_max_returns_none() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let list_array = ListArray::try_new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        assert_eq!(
            min_max(&list_array, &mut ctx, NumericalAggregateOpts::default())?,
            None
        );

        let fixed_size_list_array = FixedSizeListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            2,
            Validity::NonNullable,
            2,
        )?
        .into_array();
        assert_eq!(
            min_max(
                &fixed_size_list_array,
                &mut ctx,
                NumericalAggregateOpts::default()
            )?,
            None
        );

        Ok(())
    }

    use crate::dtype::half::f16;

    #[test]
    fn test_bool_with_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let result = min_max(
            &BoolArray::from_iter(vec![Some(true), Some(true), None, None]).into_array(),
            &mut ctx,
            NumericalAggregateOpts::default(),
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
            NumericalAggregateOpts::default(),
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
            NumericalAggregateOpts::default(),
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
            NumericalAggregateOpts::default(),
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

    /// Regression test for <https://github.com/vortex-data/vortex/issues/7074>.
    ///
    /// A chunked all-true bool array with an empty first chunk returned min=false because
    /// `accumulate_bool` on the empty chunk incorrectly merged min=false,max=false into the
    /// partial state.
    #[test]
    fn test_bool_chunked_with_empty_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let empty = BoolArray::new(BitBuffer::from([].as_slice()), Validity::NonNullable);
        let chunk1 = BoolArray::new(
            BitBuffer::from([true, true].as_slice()),
            Validity::NonNullable,
        );
        let chunk2 = BoolArray::new(
            BitBuffer::from([true, true, true].as_slice()),
            Validity::NonNullable,
        );
        let chunked = ChunkedArray::try_new(
            vec![empty.into_array(), chunk1.into_array(), chunk2.into_array()],
            DType::Bool(Nullability::NonNullable),
        )?;

        let result = min_max(
            &chunked.into_array(),
            &mut ctx,
            NumericalAggregateOpts::default(),
        )?;
        assert_eq!(
            result,
            Some(MinMaxResult {
                min: Scalar::bool(true, Nullability::NonNullable),
                max: Scalar::bool(true, Nullability::NonNullable),
            })
        );
        Ok(())
    }

    /// Regression test for <https://github.com/vortex-data/vortex/issues/8145>.
    ///
    /// A chunked array whose first chunk is an *empty* constant array, as produced by
    /// `fill_null` on an empty all-null chunk, returned `max = u32::MAX` because
    /// `ChunkedArrayAggregate` accumulated the empty chunk, folding its fill scalar into the
    /// running min/max. Empty chunks are now skipped during chunked aggregation.
    #[test]
    fn test_chunked_with_empty_constant_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let empty = ConstantArray::new(Scalar::primitive(u32::MAX, Nullability::NonNullable), 0)
            .into_array();
        let chunk1 = PrimitiveArray::new(buffer![7631471u32], Validity::NonNullable).into_array();
        let chunk2 = PrimitiveArray::new(buffer![0u32], Validity::NonNullable).into_array();
        let chunked = ChunkedArray::try_new(
            vec![empty, chunk1, chunk2],
            DType::Primitive(PType::U32, Nullability::NonNullable),
        )?;

        assert_eq!(
            min_max(
                &chunked.into_array(),
                &mut ctx,
                NumericalAggregateOpts::default()
            )?,
            Some(MinMaxResult {
                min: Scalar::primitive(0u32, Nullability::NonNullable),
                max: Scalar::primitive(7631471u32, Nullability::NonNullable),
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
        let mut ctx = SESSION.create_execution_ctx();
        assert_eq!(
            min_max(
                &array.into_array(),
                &mut ctx,
                NumericalAggregateOpts::default()
            )?,
            None
        );
        Ok(())
    }
}
