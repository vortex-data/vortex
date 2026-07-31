// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod constant;
mod decimal;
mod grouped;
mod primitive;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

pub(crate) use grouped::PrimitiveGroupedSumEncodingKernel;
use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use self::bool::accumulate_bool;
use self::constant::multiply_constant;
use self::decimal::accumulate_decimal;
use self::primitive::accumulate_primitive;
use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::arrays::ConstantArray;
use crate::arrays::StructArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::MAX_PRECISION;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

const SUM_FIELD: &str = "sum";
const IS_OVERFLOW_FIELD: &str = "is_overflow";
const IS_EMPTY_FIELD: &str = "is_empty";

/// Return the sum of an array.
///
/// See [`Sum`] for details.
pub fn sum(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
    let mut acc = Accumulator::try_new(Sum, SumAggregateOpts::default(), array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result = acc.finish()?;

    if let Some(val) = result.value().cloned() {
        array.statistics().set(Stat::Sum, Precision::Exact(val));
    }

    Ok(result)
}

/// Sum an array, returning null when it has no valid values.
///
/// If the sum overflows, a null scalar is returned. Legacy scalar partials remain supported; their
/// zero identity preserves the historical zero-on-empty behavior when encountered.
#[derive(Clone, Copy, Debug)]
pub struct Sum;

/// Options for [`Sum`].
///
/// New sums use a struct partial that can distinguish an empty input from a zero sum. The
/// `struct_partial` field exists for deserializing aggregates written before that partial was
/// introduced; callers should normally construct these options with [`Default`],
/// [`SumAggregateOpts::skip_nans`], or [`SumAggregateOpts::include_nans`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SumAggregateOpts {
    /// Whether NaN values are skipped (treated as missing) during aggregation.
    pub skip_nans: bool,
    /// Whether partials use the `{ sum, is_overflow, is_empty }` struct representation.
    pub struct_partial: bool,
}

impl SumAggregateOpts {
    /// Options that skip NaN values and use the canonical struct partial.
    pub const fn skip_nans() -> Self {
        Self {
            skip_nans: true,
            struct_partial: true,
        }
    }

    /// Options that include NaN values and use the canonical struct partial.
    pub const fn include_nans() -> Self {
        Self {
            skip_nans: false,
            struct_partial: true,
        }
    }

    /// Serialize these options to protobuf-encoded metadata bytes.
    pub fn serialize(&self) -> Vec<u8> {
        pb::SumAggregateOpts {
            skip_nans: self.skip_nans,
            struct_partial: Some(self.struct_partial),
        }
        .encode_to_vec()
    }

    /// Deserialize these options from protobuf-encoded metadata bytes.
    ///
    /// Historical Sum options were serialized as [`NumericalAggregateOpts`]. They use the same
    /// wire representation for `skip_nans` and omit `struct_partial`, which selects the legacy
    /// scalar partial representation.
    pub fn deserialize(metadata: &[u8]) -> VortexResult<Self> {
        let options = pb::SumAggregateOpts::decode(metadata)?;
        Ok(Self {
            skip_nans: options.skip_nans,
            struct_partial: options.struct_partial.unwrap_or(false),
        })
    }
}

impl Default for SumAggregateOpts {
    fn default() -> Self {
        Self::skip_nans()
    }
}

impl From<NumericalAggregateOpts> for SumAggregateOpts {
    fn from(options: NumericalAggregateOpts) -> Self {
        Self {
            skip_nans: options.skip_nans,
            struct_partial: true,
        }
    }
}

impl Display for SumAggregateOpts {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // The partial representation is a storage-compatibility detail. Keeping it out of the
        // display preserves the stats-table field name across old and new Sum partials.
        if !self.skip_nans {
            write!(f, "skip_nans=false")?;
        }
        Ok(())
    }
}

// Both Spark and DataFusion use this heuristic.
// - https://github.com/apache/spark/blob/fcf636d9eb8d645c24be3db2d599aba2d7e2955a/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/Sum.scala#L66
// - https://github.com/apache/datafusion/blob/4153adf2c0f6e317ef476febfdc834208bd46622/datafusion/functions-aggregate/src/sum.rs#L188
pub(crate) fn sum_decimal_dtype(input: &DecimalDType) -> DecimalDType {
    DecimalDType::new(
        u8::min(MAX_PRECISION, input.precision() + 10),
        input.scale(),
    )
}

impl AggregateFnVTable for Sum {
    type Options = SumAggregateOpts;
    type Partial = SumPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.sum");
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
        SumAggregateOpts::deserialize(metadata)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        // When a sum overflows, we return a null sum value. Therefore, all return dtypes are
        // nullable.
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
                DType::Decimal(sum_decimal_dtype(decimal_dtype), Nullable)
            }
            // Unsupported types
            _ => return None,
        })
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        let return_dtype = self.return_dtype(options, input_dtype)?;
        if options.struct_partial {
            Some(sum_partial_dtype(return_dtype))
        } else {
            Some(return_dtype)
        }
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        let return_dtype = self
            .return_dtype(options, input_dtype)
            .ok_or_else(|| vortex_err!("Unsupported sum dtype: {}", input_dtype))?;
        let sum = make_zero_state(&return_dtype);
        Ok(SumPartial {
            return_dtype,
            sum,
            is_overflow: false,
            is_empty: true,
            skip_nans: options.skip_nans,
            struct_partial: options.struct_partial,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        let other = normalize_partial_scalar(other, &partial.return_dtype)?;
        let fields = other.as_struct();
        let other = fields
            .field(SUM_FIELD)
            .ok_or_else(|| vortex_err!("Sum partial is missing the `{SUM_FIELD}` field"))?;
        let other_is_overflow = fields
            .field(IS_OVERFLOW_FIELD)
            .and_then(|is_overflow| is_overflow.as_bool().value())
            .ok_or_else(|| vortex_err!("Sum partial has an invalid `{IS_OVERFLOW_FIELD}` field"))?;
        let other_is_empty = fields
            .field(IS_EMPTY_FIELD)
            .and_then(|is_empty| is_empty.as_bool().value())
            .ok_or_else(|| vortex_err!("Sum partial has an invalid `{IS_EMPTY_FIELD}` field"))?;

        partial.is_empty &= other_is_empty;
        if partial.is_overflow || other_is_overflow {
            partial.is_overflow = true;
            return Ok(());
        }
        if other_is_empty {
            return Ok(());
        }

        let saturated = match &mut partial.sum {
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
            partial.is_overflow = true;
        } else {
            partial.is_empty = false;
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        if !partial.struct_partial {
            return Ok(legacy_sum_value_scalar(partial));
        }

        Ok(Scalar::struct_(
            sum_partial_dtype(partial.return_dtype.clone()),
            vec![
                sum_state_scalar(partial),
                Scalar::bool(partial.is_overflow, Nullability::NonNullable),
                Scalar::bool(partial.is_empty, Nullability::NonNullable),
            ],
        ))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.sum = make_zero_state(&partial.return_dtype);
        partial.is_overflow = false;
        partial.is_empty = true;
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        partial.is_overflow || matches!(&partial.sum, SumState::Float(v) if v.is_nan())
    }

    fn try_accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        if partial.skip_nans {
            return try_accumulate_cached_sum(self, partial, batch);
        }

        // NaN-aware short-circuits only apply to NaN-including float sums.
        if !matches!(&partial.sum, SumState::Float(_)) {
            return Ok(false);
        }
        match batch.statistics().get_as::<u64>(Stat::NaNCount) {
            Precision::Exact(0) => {
                // NaN-free batch: the cached NaN-skipping sum (if any) equals the
                // NaN-including sum.
                try_accumulate_cached_sum(self, partial, batch)
            }
            Precision::Exact(_) => {
                // At least one NaN value: the sum is NaN without scanning the batch.
                let SumState::Float(acc) = &mut partial.sum else {
                    unreachable!("checked float sum state")
                };
                *acc = f64::NAN;
                partial.is_empty = false;
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
        if partial.is_overflow {
            return Ok(());
        }

        // Constants compute scalar * len and combine via combine_partials.
        if let Columnar::Constant(c) = batch {
            if !c.scalar().is_null() && !c.is_empty() {
                partial.is_empty = false;
            }
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
        let any_valid = if partial.is_empty {
            match batch {
                Columnar::Canonical(c) => match c {
                    Canonical::Primitive(p) => {
                        any_valid(p.as_ref().validity()?, p.as_ref().len(), ctx)?
                    }
                    Canonical::Bool(b) => any_valid(b.as_ref().validity()?, b.as_ref().len(), ctx)?,
                    Canonical::Decimal(d) => {
                        any_valid(d.as_ref().validity()?, d.as_ref().len(), ctx)?
                    }
                    _ => vortex_bail!("Unsupported canonical type for sum: {}", batch.dtype()),
                },
                Columnar::Constant(_) => unreachable!(),
            }
        } else {
            false
        };

        let result = match batch {
            Columnar::Canonical(c) => match c {
                Canonical::Primitive(p) => {
                    accumulate_primitive(&mut partial.sum, p, ctx, skip_nans)
                }
                Canonical::Bool(b) => accumulate_bool(&mut partial.sum, b, ctx),
                Canonical::Decimal(d) => accumulate_decimal(&mut partial.sum, d, ctx),
                _ => vortex_bail!("Unsupported canonical type for sum: {}", batch.dtype()),
            },
            Columnar::Constant(_) => unreachable!(),
        };

        match result {
            Ok(false) => {
                if any_valid {
                    partial.is_empty = false;
                }
            }
            Ok(true) => partial.is_overflow = true,
            Err(e) => return Err(e),
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        let partials = normalize_partial_array(partials)?;
        let sum = partials.get_item(SUM_FIELD)?;
        let is_invalid = partials
            .get_item(IS_OVERFLOW_FIELD)?
            .binary(partials.get_item(IS_EMPTY_FIELD)?, Operator::Or)?
            .fill_null(true)?;
        sum.mask(is_invalid.not()?)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        if partial.struct_partial {
            Ok(sum_value_scalar(partial))
        } else {
            Ok(legacy_sum_value_scalar(partial))
        }
    }
}

/// In-memory state for sum accumulation.
pub struct SumPartial {
    return_dtype: DType,
    /// The non-null running sum, initialized to zero.
    sum: SumState,
    /// Whether checked arithmetic overflowed.
    is_overflow: bool,
    /// Whether no valid value has been accumulated.
    is_empty: bool,
    /// Whether NaN values in float inputs are skipped.
    skip_nans: bool,
    /// Whether this accumulator emits the canonical struct partial.
    struct_partial: bool,
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

pub(crate) fn make_zero_state(return_dtype: &DType) -> SumState {
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

fn sum_partial_dtype(sum_dtype: DType) -> DType {
    DType::Struct(
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
        ),
        Nullability::Nullable,
    )
}

fn sum_state_scalar(partial: &SumPartial) -> Scalar {
    match &partial.sum {
        SumState::Unsigned(v) => Scalar::primitive(*v, Nullability::NonNullable),
        SumState::Signed(v) => Scalar::primitive(*v, Nullability::NonNullable),
        SumState::Float(v) => Scalar::primitive(*v, Nullability::NonNullable),
        SumState::Decimal { value, .. } => {
            let decimal_dtype = *partial
                .return_dtype
                .as_decimal_opt()
                .vortex_expect("return dtype must be decimal");
            Scalar::decimal(*value, decimal_dtype, Nullability::NonNullable)
        }
    }
}

fn sum_value_scalar(partial: &SumPartial) -> Scalar {
    if partial.is_overflow || partial.is_empty {
        return Scalar::null(partial.return_dtype.as_nullable());
    }

    nullable_sum_state_scalar(partial)
}

fn legacy_sum_value_scalar(partial: &SumPartial) -> Scalar {
    if partial.is_overflow {
        return Scalar::null(partial.return_dtype.as_nullable());
    }

    nullable_sum_state_scalar(partial)
}

fn nullable_sum_state_scalar(partial: &SumPartial) -> Scalar {
    match &partial.sum {
        SumState::Unsigned(v) => Scalar::primitive(*v, Nullability::Nullable),
        SumState::Signed(v) => Scalar::primitive(*v, Nullability::Nullable),
        SumState::Float(v) => Scalar::primitive(*v, Nullability::Nullable),
        SumState::Decimal { value, .. } => {
            let decimal_dtype = *partial
                .return_dtype
                .as_decimal_opt()
                .vortex_expect("return dtype must be decimal");
            Scalar::decimal(*value, decimal_dtype, Nullability::Nullable)
        }
    }
}

/// Normalize an array of scalar legacy Sum partials into the canonical struct partial shape.
///
/// Canonical partial arrays are returned unchanged. A legacy non-null scalar becomes a non-empty
/// partial, while a legacy null becomes an overflowed partial. Legacy Sum used zero for empty
/// inputs, so a scalar partial cannot represent `is_empty = true`.
pub fn normalize_partial_array(partials: ArrayRef) -> VortexResult<ArrayRef> {
    if matches!(partials.dtype(), DType::Struct(..)) {
        return Ok(partials);
    }

    let len = partials.len();
    let sum_dtype = partials.dtype().as_nonnullable();
    let is_overflow = partials.is_null()?;
    let sum = partials.fill_null(Scalar::zero_value(&sum_dtype))?;
    let is_empty = ConstantArray::new(false, len).into_array();

    Ok(StructArray::try_new(
        FieldNames::from_iter([
            FieldName::from(SUM_FIELD),
            FieldName::from(IS_OVERFLOW_FIELD),
            FieldName::from(IS_EMPTY_FIELD),
        ]),
        vec![sum, is_overflow, is_empty],
        len,
        Validity::AllValid,
    )?
    .into_array())
}

fn normalize_partial_scalar(partial: Scalar, return_dtype: &DType) -> VortexResult<Scalar> {
    let partial_dtype = sum_partial_dtype(return_dtype.clone());
    if matches!(partial.dtype(), DType::Struct(..)) {
        if partial.is_null() {
            return Ok(Scalar::struct_(
                partial_dtype,
                vec![
                    Scalar::zero_value(&return_dtype.as_nonnullable()),
                    Scalar::bool(true, Nullability::NonNullable),
                    Scalar::bool(false, Nullability::NonNullable),
                ],
            ));
        }
        return partial.cast(&partial_dtype);
    }

    if !partial.dtype().eq_ignore_nullability(return_dtype) {
        vortex_bail!(
            "Legacy Sum partial has dtype {}, expected {}",
            partial.dtype(),
            return_dtype
        );
    }

    let is_overflow = partial.is_null();
    let sum = if is_overflow {
        Scalar::zero_value(&return_dtype.as_nonnullable())
    } else {
        partial.cast(&return_dtype.as_nonnullable())?
    };
    Ok(Scalar::struct_(
        partial_dtype,
        vec![
            sum,
            Scalar::bool(is_overflow, Nullability::NonNullable),
            Scalar::bool(false, Nullability::NonNullable),
        ],
    ))
}

fn try_accumulate_cached_sum(
    vtable: &Sum,
    partial: &mut SumPartial,
    batch: &ArrayRef,
) -> VortexResult<bool> {
    let Precision::Exact(sum) = batch.statistics().get(Stat::Sum) else {
        return Ok(false);
    };

    let sum = if sum.dtype() == &partial.return_dtype {
        sum
    } else {
        sum.cast(&partial.return_dtype)?
    };
    vtable.combine_partials(partial, sum)?;
    Ok(true)
}

fn any_valid(validity: Validity, len: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    Ok(validity.execute_mask(len, ctx)?.true_count() > 0)
}

/// Checked add for u64, returning true if overflow occurred.
#[inline(always)]
pub(crate) fn checked_add_u64(acc: &mut u64, val: u64) -> bool {
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
pub(crate) fn checked_add_i64(acc: &mut i64, val: i64) -> bool {
    match acc.checked_add(val) {
        Some(r) => {
            *acc = r;
            false
        }
        None => true,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod arithmetic_tests {
    use num_traits::CheckedAdd;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::SumAggregateOpts;
    use crate::aggregate_fn::fns::sum::sum;
    use crate::array_session;
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
        let mut ctx = array_session().create_execution_ctx();
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
            // Add widens the result precision, so restate the sum in the accumulator's own
            // decimal type, treating a value that no longer fits as an overflow.
            DType::Decimal(decimal_dtype, _) => lhs
                .as_decimal()
                .checked_binary_numeric(&rhs.as_decimal(), NumericOperator::Add)?
                .and_then(|scalar| scalar.as_decimal().decimal_value())
                .filter(|value| value.fits_in_precision(*decimal_dtype))
                .map(|value| Scalar::decimal(value, *decimal_dtype, Nullable))
                .unwrap_or_else(|| Scalar::null(sum_dtype.as_nullable())),
            _ => unreachable!("Sum will always be a decimal or a primitive dtype"),
        })
    }

    // Multi-batch and reset tests

    #[test]
    fn sum_multi_batch() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, SumAggregateOpts::default(), dtype)?;

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
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, SumAggregateOpts::default(), dtype)?;

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
        let options = SumAggregateOpts::default();
        let mut state = Sum.empty_partial(&options, &dtype)?;

        let scalar1 = Scalar::primitive(100i64, Nullable);
        Sum.combine_partials(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(50i64, Nullable);
        Sum.combine_partials(&mut state, scalar2)?;

        let result = Sum.finalize_scalar(&state)?;
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

        let sum_without_acc = sum(&array, &mut array_session().create_execution_ctx())?;
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
        let mut acc =
            GroupedAccumulator::try_new(Sum, SumAggregateOpts::default(), elem_dtype.clone())?;
        acc.accumulate_list(groups, &mut array_session().create_execution_ctx())?;
        acc.finish()
    }

    #[test]
    fn grouped_sum_fixed_size_list() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
        let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(6i64), Some(15i64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_with_null_elements() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5), Some(6)])
                .into_array();
        let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(4i64), Some(11i64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_with_null_group() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9], Validity::NonNullable)
                .into_array();
        let validity = Validity::from_iter([true, false, true]);
        let groups = FixedSizeListArray::try_new(elements, 3, validity, 3)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected =
            PrimitiveArray::from_option_iter([Some(6i64), None, Some(24i64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_all_null_elements_in_group() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements =
            PrimitiveArray::from_option_iter([None::<i32>, None, Some(3), Some(4)]).into_array();
        let groups = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([None, Some(7i64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_bool() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elements: BoolArray = [true, false, true, true, true, true].into_iter().collect();
        let groups =
            FixedSizeListArray::try_new(elements.into_array(), 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Bool(Nullability::NonNullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(2u64), Some(3u64)]).into_array();
        assert_arrays_eq!(&result, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_finish_resets() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = GroupedAccumulator::try_new(Sum, SumAggregateOpts::default(), elem_dtype)?;

        let elements1 =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let groups1 = FixedSizeListArray::try_new(elements1, 2, Validity::NonNullable, 2)?;
        acc.accumulate_list(&groups1.into_array(), &mut ctx)?;
        let result1 = acc.finish()?;

        let expected1 = PrimitiveArray::from_option_iter([Some(3i64), Some(7i64)]).into_array();
        assert_arrays_eq!(&result1, &expected1, &mut ctx);

        let elements2 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        let groups2 = FixedSizeListArray::try_new(elements2, 2, Validity::NonNullable, 1)?;
        acc.accumulate_list(&groups2.into_array(), &mut ctx)?;
        let result2 = acc.finish()?;

        let expected2 = PrimitiveArray::from_option_iter([Some(30i64)]).into_array();
        assert_arrays_eq!(&result2, &expected2, &mut ctx);
        Ok(())
    }

    #[test]
    fn grouped_sum_listview_out_of_order_offsets_with_null_group() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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
        assert_arrays_eq!(&result, &expected, &mut ctx);
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
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(result.as_primitive().as_::<f64>(), Some(20.8));
        Ok(())
    }

    #[test]
    fn sum_chunked_floats_all_nulls_is_null() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None, None]);
        let chunk2 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype)?;
        let result = sum(
            &chunked.into_array(),
            &mut array_session().create_execution_ctx(),
        )?;
        assert_eq!(result, Scalar::null(DType::Primitive(PType::F64, Nullable)));
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
            &mut array_session().create_execution_ctx(),
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
            &mut array_session().create_execution_ctx(),
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
            &mut array_session().create_execution_ctx(),
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
            &mut array_session().create_execution_ctx(),
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
            &mut array_session().create_execution_ctx(),
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
