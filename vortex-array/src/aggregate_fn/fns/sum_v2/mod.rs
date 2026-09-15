// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SQL-style sums with explicit overflow and empty-input state.
//!
//! [`SumV2`] returns null when there are no valid values, while [`Sum`] returns zero. Tracking empty
//! input separately from overflow requires a different partial representation. A separate aggregate
//! preserves compatibility with the scalar partials stored by older Vortex files.

mod grouped;
pub(crate) use grouped::PrimitiveGroupedSumV2EncodingKernel;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ArrayView;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::sum::SumState;
use crate::aggregate_fn::fns::sum::accumulate_bool;
use crate::aggregate_fn::fns::sum::accumulate_decimal;
use crate::aggregate_fn::fns::sum::accumulate_primitive;
use crate::aggregate_fn::fns::sum::checked_add_sum_states;
use crate::aggregate_fn::fns::sum::make_zero_state;
use crate::aggregate_fn::fns::sum::multiply_constant;
use crate::arrays::Struct;
use crate::arrays::struct_::StructArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

const SUM_FIELD: &str = "sum";
const IS_OVERFLOW_FIELD: &str = "is_overflow";
const IS_EMPTY_FIELD: &str = "is_empty";

/// Return the SQL-style sum of an array.
///
/// Unlike [`sum`](crate::aggregate_fn::fns::sum::sum), this returns null when the array has no
/// valid values.
pub fn sum_v2(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(
        SumV2,
        NumericalAggregateOpts::default(),
        array.dtype().clone(),
    )?;
    acc.accumulate(array, ctx)?;
    acc.finish()
}

/// Sum an array, returning null when it has no valid values or if the sum overflows.
///
/// This aggregate intentionally has a distinct ID and partial representation from the legacy
/// [`Sum`]. Keeping `vortex.sum` unchanged preserves the scalar partials stored by older Vortex
/// files, while `SumV2` can use an explicit `{ sum, is_overflow, is_empty }` state.
///
/// NaN handling for float inputs is controlled by [`NumericalAggregateOpts`]. With `skip_nans`
/// (the default), NaN values contribute nothing but still make the input non-empty. Otherwise,
/// any NaN value poisons the sum to NaN.
#[derive(Clone, Copy, Debug)]
pub struct SumV2;

impl SumV2 {
    /// Build an encoding kernel's partial from a widened primitive sum.
    ///
    /// `sum` must have dtype `u64`, `i64`, or `f64`. A null sum records overflow and overrides
    /// `is_empty`. Set `is_empty` only when there were no valid inputs. Valid NaNs make the input
    /// non-empty even when skipped. The returned struct and its fields are non-null.
    pub fn partial_from_sum(sum: Scalar, is_empty: bool) -> VortexResult<Scalar> {
        vortex_ensure!(
            matches!(
                sum.dtype(),
                DType::Primitive(PType::U64 | PType::I64 | PType::F64, _)
            ),
            "Expected a widened primitive sum, got {}",
            sum.dtype(),
        );

        let sum_dtype = sum.dtype().as_nonnullable();
        let is_overflow = sum.is_null();
        let sum = if is_overflow {
            Scalar::zero_value(&sum_dtype)
        } else {
            sum.cast(&sum_dtype)?
        };

        Ok(Scalar::struct_(
            sum_v2_partial_dtype(sum_dtype),
            vec![
                sum,
                Scalar::bool(is_overflow, Nullability::NonNullable),
                Scalar::bool(is_empty && !is_overflow, Nullability::NonNullable),
            ],
        ))
    }
}

impl AggregateFnVTable for SumV2 {
    type Options = NumericalAggregateOpts;
    type Partial = SumV2Partial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.sum_v2");
        *ID
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

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        Sum.return_dtype(options, input_dtype)
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
            .map(sum_v2_partial_dtype)
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        Ok(SumV2Partial::empty(args.return_dtype))
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: Scalar,
    ) -> VortexResult<Self::Partial> {
        let mut partial = SumV2Partial::empty(args.return_dtype);
        let (sum, is_overflow, is_empty) = decode_partial_scalar(scalar)?;
        validate_sum_field_dtype(&sum, args.return_dtype)?;

        // Adding the parsed value to the zero state cannot overflow; treat a decimal value that
        // no longer fits its precision as an already-overflowed partial.
        let overflowed = checked_add_sum_state(&mut partial.sum, args.return_dtype, &sum)?;
        partial.is_overflow = is_overflow || overflowed;
        partial.is_empty = is_empty && !partial.is_overflow;
        Ok(partial)
    }

    fn merge_partials(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        mut acc: Self::Partial,
        partial: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        if acc.is_overflow {
            return Ok(acc);
        }
        if partial.is_overflow {
            // An overflowed state keeps its last valid sum, so an empty accumulator adopts the
            // incoming sum rather than reporting zero.
            if acc.is_empty {
                acc.sum = partial.sum;
            }
            acc.is_overflow = true;
            acc.is_empty = false;
            return Ok(acc);
        }
        if partial.is_empty {
            return Ok(acc);
        }
        acc.is_overflow = checked_add_sum_states(&mut acc.sum, args.return_dtype, &partial.sum)?;
        acc.is_empty = false;
        Ok(acc)
    }

    fn to_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::struct_(
            args.partial_dtype.clone(),
            vec![
                sum_state_scalar(partial, args.return_dtype, Nullability::NonNullable),
                Scalar::bool(partial.is_overflow, Nullability::NonNullable),
                Scalar::bool(partial.is_empty, Nullability::NonNullable),
            ],
        ))
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> bool {
        partial.is_overflow || matches!(&partial.sum, SumState::Float(value) if value.is_nan())
    }

    fn try_accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if args.options.skip_nans || !matches!(&partial.sum, SumState::Float(_)) {
            return Ok(false);
        }

        match batch.statistics().get_as::<u64>(Stat::NaNCount) {
            Precision::Exact(0) => Ok(false),
            Precision::Exact(_) => {
                let SumState::Float(sum) = &mut partial.sum else {
                    unreachable!("checked float sum state")
                };
                *sum = f64::NAN;
                partial.is_empty = false;
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
        if partial.is_overflow {
            return Ok(());
        }

        if let Columnar::Constant(constant) = batch {
            if !constant.scalar().is_null() && !constant.is_empty() {
                partial.is_empty = false;
            }
            if args.options.skip_nans
                && constant
                    .scalar()
                    .as_primitive_opt()
                    .is_some_and(|primitive| primitive.is_nan())
            {
                return Ok(());
            }
            if let Some(product) =
                multiply_constant(constant.scalar(), constant.len(), args.return_dtype)?
            {
                if product.is_null() {
                    partial.is_overflow = true;
                    partial.is_empty = false;
                } else {
                    partial.is_overflow =
                        checked_add_sum_state(&mut partial.sum, args.return_dtype, &product)?;
                    partial.is_empty = false;
                }
            }
            return Ok(());
        }

        let any_valid = partial.is_empty && has_valid_value(batch, ctx)?;
        let result = match batch {
            Columnar::Canonical(canonical) => match canonical {
                Canonical::Primitive(array) => {
                    accumulate_primitive(&mut partial.sum, array, ctx, args.options.skip_nans)
                }
                Canonical::Bool(array) => accumulate_bool(&mut partial.sum, array, ctx),
                Canonical::Decimal(array) => {
                    accumulate_decimal(&mut partial.sum, args.return_dtype, array, ctx)
                }
                _ => vortex_bail!("Unsupported canonical type for sum_v2: {}", batch.dtype()),
            },
            Columnar::Constant(_) => unreachable!(),
        };

        if any_valid {
            partial.is_empty = false;
        }
        if result? {
            partial.is_overflow = true;
            partial.is_empty = false;
        }
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        if let Some(partials) = partials.as_opt::<Struct>() {
            return finalize_struct(partials);
        }

        let sum = partials.get_item(SUM_FIELD)?;
        let is_invalid = partials
            .get_item(IS_OVERFLOW_FIELD)?
            .binary(partials.get_item(IS_EMPTY_FIELD)?, Operator::Or)?
            .fill_null(true)?;
        sum.mask(is_invalid.not()?)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        if partial.is_overflow || partial.is_empty {
            return Ok(Scalar::null(args.return_dtype.as_nullable()));
        }
        Ok(sum_state_scalar(
            partial,
            args.return_dtype,
            Nullability::Nullable,
        ))
    }
}

fn finalize_struct(partials: ArrayView<'_, Struct>) -> VortexResult<ArrayRef> {
    let sum = partials.unmasked_field_by_name(SUM_FIELD)?.clone();
    let is_overflow = partials.unmasked_field_by_name(IS_OVERFLOW_FIELD)?.clone();
    let is_empty = partials.unmasked_field_by_name(IS_EMPTY_FIELD)?.clone();

    let is_invalid = is_overflow
        .binary(is_empty, Operator::Or)?
        .fill_null(true)?;
    let mut is_valid = is_invalid.not()?;
    match partials.struct_validity() {
        Validity::NonNullable | Validity::AllValid => {}
        validity => {
            is_valid = is_valid.binary(validity.to_array(partials.len()), Operator::And)?;
        }
    }

    sum.mask(is_valid)
}

/// In-memory state for SumV2 accumulation.
pub struct SumV2Partial {
    sum: SumState,
    is_overflow: bool,
    is_empty: bool,
}

impl SumV2Partial {
    /// The state of a group with no accumulated values.
    fn empty(return_dtype: &DType) -> Self {
        Self {
            sum: make_zero_state(return_dtype),
            is_overflow: false,
            is_empty: true,
        }
    }
}

fn has_valid_value(batch: &Columnar, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    let (validity, len) = match batch {
        Columnar::Canonical(Canonical::Primitive(array)) => {
            (array.as_ref().validity()?, array.as_ref().len())
        }
        Columnar::Canonical(Canonical::Bool(array)) => {
            (array.as_ref().validity()?, array.as_ref().len())
        }
        Columnar::Canonical(Canonical::Decimal(array)) => {
            (array.as_ref().validity()?, array.as_ref().len())
        }
        Columnar::Canonical(_) => return Ok(false),
        Columnar::Constant(constant) => {
            return Ok(!constant.is_empty() && !constant.scalar().is_null());
        }
    };
    Ok(validity.execute_mask(len, ctx)?.true_count() > 0)
}

fn decode_partial_scalar(scalar: Scalar) -> VortexResult<(Scalar, bool, bool)> {
    vortex_ensure!(!scalar.is_null(), "SumV2 partial must not be null");

    let Some(fields) = scalar.as_struct_opt() else {
        vortex_bail!("SumV2 partial must be a struct, got {}", scalar.dtype());
    };
    let sum = fields
        .field(SUM_FIELD)
        .ok_or_else(|| vortex_err!("SumV2 partial is missing the sum field"))?;
    let is_overflow = bool::try_from(
        &fields
            .field(IS_OVERFLOW_FIELD)
            .ok_or_else(|| vortex_err!("SumV2 partial is missing the is_overflow field"))?,
    )?;
    let is_empty = bool::try_from(
        &fields
            .field(IS_EMPTY_FIELD)
            .ok_or_else(|| vortex_err!("SumV2 partial is missing the is_empty field"))?,
    )?;

    Ok((sum, is_overflow, is_empty))
}

fn validate_sum_field_dtype(sum: &Scalar, return_dtype: &DType) -> VortexResult<()> {
    vortex_ensure!(
        sum.dtype().nullability() == Nullability::NonNullable
            && sum.dtype().eq_ignore_nullability(return_dtype),
        "SumV2 partial value has dtype {}, expected {}",
        sum.dtype(),
        return_dtype.as_nonnullable(),
    );
    Ok(())
}

fn checked_add_sum_state(
    state: &mut SumState,
    return_dtype: &DType,
    other: &Scalar,
) -> VortexResult<bool> {
    Ok(match state {
        SumState::Unsigned(sum) => checked_add_u64(sum, u64::try_from(other)?),
        SumState::Signed(sum) => checked_add_i64(sum, i64::try_from(other)?),
        SumState::Float(sum) => {
            *sum += f64::try_from(other)?;
            false
        }
        SumState::Decimal(value) => {
            let dtype = return_dtype
                .as_decimal_opt()
                .vortex_expect("decimal sum result dtype");
            let other = DecimalValue::try_from(other)?;
            match value.checked_add(&other) {
                Some(result) if result.fits_in_precision(*dtype) => {
                    *value = result;
                    false
                }
                Some(_) | None => true,
            }
        }
    })
}

fn sum_v2_partial_dtype(sum_dtype: DType) -> DType {
    DType::Struct(sum_v2_partial_fields(sum_dtype), Nullability::Nullable)
}

fn sum_v2_partial_fields(sum_dtype: DType) -> StructFields {
    StructFields::new(
        FieldNames::from_iter([
            FieldName::from(SUM_FIELD),
            FieldName::from(IS_OVERFLOW_FIELD),
            FieldName::from(IS_EMPTY_FIELD),
        ]),
        vec![
            sum_dtype.as_nonnullable(),
            DType::Bool(Nullability::NonNullable),
            DType::Bool(Nullability::NonNullable),
        ],
    )
}

fn sum_state_scalar(
    partial: &SumV2Partial,
    return_dtype: &DType,
    nullability: Nullability,
) -> Scalar {
    match &partial.sum {
        SumState::Unsigned(value) => Scalar::primitive(*value, nullability),
        SumState::Signed(value) => Scalar::primitive(*value, nullability),
        SumState::Float(value) => Scalar::primitive(*value, nullability),
        SumState::Decimal(value) => Scalar::decimal(
            *value,
            *return_dtype
                .as_decimal_opt()
                .vortex_expect("decimal sum result dtype"),
            nullability,
        ),
    }
}

fn checked_add_u64(sum: &mut u64, value: u64) -> bool {
    match sum.checked_add(value) {
        Some(result) => {
            *sum = result;
            false
        }
        None => true,
    }
}

fn checked_add_i64(sum: &mut i64, value: i64) -> bool {
    match sum.checked_add(value) {
        Some(result) => {
            *sum = result;
            false
        }
        None => true,
    }
}

#[cfg(test)]
mod tests;
