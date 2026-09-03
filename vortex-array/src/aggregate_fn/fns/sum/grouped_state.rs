// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use num_traits::CheckedAdd;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use super::checked_add_i64;
use super::checked_add_u64;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::GroupedState;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Monomorphic sums for one grouped sum accumulator.
///
/// The variant is chosen once from the aggregate's sum dtype, so accumulation never dispatches on
/// the state type per value. Decimal states use the narrowest native width that can hold the sum
/// dtype's precision.
pub(crate) enum SumGroupedValues {
    Unsigned(Vec<u64>),
    Signed(Vec<i64>),
    Float(Vec<f64>),
    Decimal8(Vec<i8>),
    Decimal16(Vec<i16>),
    Decimal32(Vec<i32>),
    Decimal64(Vec<i64>),
    Decimal128(Vec<i128>),
    Decimal256(Vec<i256>),
}

/// Invoke `$body` with the `Vec` of whichever decimal width `$values` holds.
macro_rules! match_each_decimal_state {
    ($values:expr, | $sums:ident | $body:expr, $otherwise:expr) => {{
        match $values {
            SumGroupedValues::Decimal8($sums) => $body,
            SumGroupedValues::Decimal16($sums) => $body,
            SumGroupedValues::Decimal32($sums) => $body,
            SumGroupedValues::Decimal64($sums) => $body,
            SumGroupedValues::Decimal128($sums) => $body,
            SumGroupedValues::Decimal256($sums) => $body,
            _ => $otherwise,
        }
    }};
}
pub(crate) use match_each_decimal_state;

impl SumGroupedValues {
    fn len(&self) -> usize {
        match self {
            Self::Unsigned(values) => values.len(),
            Self::Signed(values) => values.len(),
            Self::Float(values) => values.len(),
            other => match_each_decimal_state!(other, |values| values.len(), unreachable!()),
        }
    }

    fn resize(&mut self, len: usize) {
        match self {
            Self::Unsigned(values) => values.resize(len, 0),
            Self::Signed(values) => values.resize(len, 0),
            Self::Float(values) => values.resize(len, 0.0),
            other => match_each_decimal_state!(
                other,
                |values| values.resize(len, Default::default()),
                unreachable!()
            ),
        }
    }
}

/// Dense per-group sums and their overflow flags.
///
/// Shared by the [`Sum`](super::Sum) and [`SumV2`](crate::aggregate_fn::fns::sum_v2::SumV2)
/// grouped states, which differ only in how they present a saturated or untouched group.
pub(crate) struct DenseSums {
    values: SumGroupedValues,
    overflowed: Vec<bool>,
    decimal_dtype: Option<DecimalDType>,
}

impl DenseSums {
    /// Create dense state for sums of `sum_dtype`, the aggregate's (non-null) sum type.
    pub(crate) fn try_new(sum_dtype: &DType) -> VortexResult<Self> {
        let values = match sum_dtype {
            DType::Primitive(PType::U64, _) => SumGroupedValues::Unsigned(Vec::new()),
            DType::Primitive(PType::I64, _) => SumGroupedValues::Signed(Vec::new()),
            DType::Primitive(PType::F64, _) => SumGroupedValues::Float(Vec::new()),
            DType::Decimal(dtype, _) => match DecimalType::smallest_decimal_value_type(dtype) {
                DecimalType::I8 => SumGroupedValues::Decimal8(Vec::new()),
                DecimalType::I16 => SumGroupedValues::Decimal16(Vec::new()),
                DecimalType::I32 => SumGroupedValues::Decimal32(Vec::new()),
                DecimalType::I64 => SumGroupedValues::Decimal64(Vec::new()),
                DecimalType::I128 => SumGroupedValues::Decimal128(Vec::new()),
                DecimalType::I256 => SumGroupedValues::Decimal256(Vec::new()),
            },
            dtype => vortex_bail!("Unsupported grouped sum dtype: {dtype}"),
        };
        Ok(Self {
            values,
            overflowed: Vec::new(),
            decimal_dtype: sum_dtype.as_decimal_opt().copied(),
        })
    }

    /// The number of allocated group slots.
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// Grow the state to hold at least `num_groups` slots.
    pub(crate) fn ensure_groups(&mut self, num_groups: usize) {
        let len = num_groups.max(self.len());
        self.values.resize(len);
        self.overflowed.resize(len, false);
    }

    /// Split borrow of the sums and their overflow flags, for use by grouped sum kernels.
    pub(crate) fn parts_mut(&mut self) -> (&mut SumGroupedValues, &mut [bool]) {
        (&mut self.values, &mut self.overflowed)
    }

    /// The decimal dtype of the state, if this is a decimal sum.
    pub(crate) fn decimal_dtype(&self) -> Option<DecimalDType> {
        self.decimal_dtype
    }

    fn decimal_dtype_or_bail(&self) -> VortexResult<DecimalDType> {
        self.decimal_dtype
            .ok_or_else(|| vortex_err!("Expected a decimal grouped sum state"))
    }

    /// Whether one group has overflowed, or accumulated a NaN float sum.
    pub(crate) fn is_saturated(&self, group_id: usize) -> bool {
        if self.overflowed[group_id] {
            return true;
        }
        matches!(&self.values, SumGroupedValues::Float(values) if values[group_id].is_nan())
    }

    /// Whether one group has overflowed.
    pub(crate) fn is_overflowed(&self, group_id: usize) -> bool {
        self.overflowed.get(group_id).copied().unwrap_or(false)
    }

    /// Mark one group as overflowed.
    pub(crate) fn set_overflowed(&mut self, group_id: usize) {
        self.overflowed[group_id] = true;
    }

    /// Add one non-null sum value into a group, saturating it on overflow.
    pub(crate) fn add_scalar(&mut self, group_id: usize, value: &Scalar) -> VortexResult<()> {
        if self.overflowed[group_id] {
            return Ok(());
        }

        let decimal_dtype = self.decimal_dtype;
        let (values, overflowed) = self.parts_mut();
        match values {
            SumGroupedValues::Unsigned(values) => {
                let value = value
                    .as_primitive()
                    .typed_value::<u64>()
                    .vortex_expect("checked non-null");
                overflowed[group_id] = checked_add_u64(&mut values[group_id], value);
            }
            SumGroupedValues::Signed(values) => {
                let value = value
                    .as_primitive()
                    .typed_value::<i64>()
                    .vortex_expect("checked non-null");
                overflowed[group_id] = checked_add_i64(&mut values[group_id], value);
            }
            SumGroupedValues::Float(values) => {
                values[group_id] += value
                    .as_primitive()
                    .typed_value::<f64>()
                    .vortex_expect("checked non-null");
            }
            decimals => {
                let dtype =
                    decimal_dtype.ok_or_else(|| vortex_err!("Expected a decimal sum partial"))?;
                let value = value
                    .as_decimal()
                    .decimal_value()
                    .ok_or_else(|| vortex_err!("Expected a decimal sum partial"))?;
                match_each_decimal_state!(
                    decimals,
                    |sums| {
                        match value.cast() {
                            Some(value) => add_decimal(sums, overflowed, group_id, value, dtype),
                            // The partial does not fit the state width, so it cannot fit the sum
                            // dtype's precision either.
                            None => overflowed[group_id] = true,
                        }
                    },
                    unreachable!("checked decimal state above")
                );
            }
        }
        Ok(())
    }

    /// One group's current sum, ignoring whether the group overflowed.
    pub(crate) fn value_scalar(
        &self,
        group_id: usize,
        nullability: Nullability,
    ) -> VortexResult<Scalar> {
        Ok(match &self.values {
            SumGroupedValues::Unsigned(values) => {
                Scalar::primitive(values.get(group_id).copied().unwrap_or(0), nullability)
            }
            SumGroupedValues::Signed(values) => {
                Scalar::primitive(values.get(group_id).copied().unwrap_or(0), nullability)
            }
            SumGroupedValues::Float(values) => {
                Scalar::primitive(values.get(group_id).copied().unwrap_or(0.0), nullability)
            }
            decimals => {
                let dtype = self.decimal_dtype_or_bail()?;
                match_each_decimal_state!(
                    decimals,
                    |values| Scalar::decimal(
                        DecimalValue::from(values.get(group_id).copied().unwrap_or_default()),
                        dtype,
                        nullability,
                    ),
                    unreachable!("checked decimal state above")
                )
            }
        })
    }

    /// Take the accumulated overflow flags, leaving the state empty.
    pub(crate) fn take_overflowed(&mut self) -> Vec<bool> {
        std::mem::take(&mut self.overflowed)
    }

    /// Take the accumulated sums as an array with `validity`, leaving the state empty.
    pub(crate) fn take_values(&mut self, validity: Validity) -> VortexResult<ArrayRef> {
        let decimal_dtype = self.decimal_dtype;
        Ok(match &mut self.values {
            SumGroupedValues::Unsigned(values) => {
                PrimitiveArray::new(Buffer::from(std::mem::take(values)), validity).into_array()
            }
            SumGroupedValues::Signed(values) => {
                PrimitiveArray::new(Buffer::from(std::mem::take(values)), validity).into_array()
            }
            SumGroupedValues::Float(values) => {
                PrimitiveArray::new(Buffer::from(std::mem::take(values)), validity).into_array()
            }
            decimals => {
                let dtype =
                    decimal_dtype.ok_or_else(|| vortex_err!("Expected a decimal sum state"))?;
                match_each_decimal_state!(
                    decimals,
                    |values| DecimalArray::new(
                        Buffer::from(std::mem::take(values)),
                        dtype,
                        validity
                    )
                    .into_array(),
                    unreachable!("checked decimal state above")
                )
            }
        })
    }

    /// Fold an array of nullable sum partials into dense groups, where a null partial saturates
    /// its group.
    pub(crate) fn accumulate_nullable_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &[u32],
        validity: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let decimal_dtype = self.decimal_dtype;
        let (values, overflowed) = self.parts_mut();
        match values {
            SumGroupedValues::Unsigned(values) => {
                let partials = partials.clone().execute::<PrimitiveArray>(ctx)?;
                accumulate_partials_with(
                    values,
                    overflowed,
                    group_ids,
                    validity,
                    partials.as_slice::<u64>(),
                    checked_add_u64,
                );
            }
            SumGroupedValues::Signed(values) => {
                let partials = partials.clone().execute::<PrimitiveArray>(ctx)?;
                accumulate_partials_with(
                    values,
                    overflowed,
                    group_ids,
                    validity,
                    partials.as_slice::<i64>(),
                    checked_add_i64,
                );
            }
            SumGroupedValues::Float(values) => {
                let partials = partials.clone().execute::<PrimitiveArray>(ctx)?;
                let sums = partials.as_slice::<f64>();
                for_each_valid_partial(group_ids, validity, overflowed, |idx, group, _| {
                    values[group] += sums[idx];
                });
            }
            decimals => {
                let dtype =
                    decimal_dtype.ok_or_else(|| vortex_err!("Expected decimal sum partials"))?;
                let partials = partials.clone().execute::<DecimalArray>(ctx)?;
                match_each_decimal_state!(
                    decimals,
                    |sums| accumulate_decimal_partials(
                        sums, overflowed, group_ids, validity, &partials, dtype
                    ),
                    unreachable!("checked decimal state above")
                );
            }
        }
        Ok(())
    }
}

/// Dense grouped [`Sum`](super::Sum) state.
///
/// Each group holds one native sum plus an overflow flag. A saturated group flushes as a null
/// partial, matching the scalar `Sum` contract where null means "overflowed"; empty groups keep
/// their zero initial value.
pub(crate) struct SumGroupedState {
    sums: DenseSums,
    partial_dtype: DType,
}

impl SumGroupedState {
    pub(crate) fn try_new(partial_dtype: DType) -> VortexResult<Self> {
        Ok(Self {
            sums: DenseSums::try_new(&partial_dtype)?,
            partial_dtype,
        })
    }

    /// The dense sums, for use by the grouped sum kernel.
    pub(crate) fn sums_mut(&mut self) -> &mut DenseSums {
        &mut self.sums
    }
}

impl GroupedState for SumGroupedState {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.sums.len()
    }

    fn ensure_groups(&mut self, num_groups: usize) -> VortexResult<()> {
        self.sums.ensure_groups(num_groups);
        Ok(())
    }

    fn is_saturated(&self, group_id: usize) -> bool {
        self.sums.is_saturated(group_id)
    }

    fn combine_scalar(&mut self, group_id: usize, partial: Scalar) -> VortexResult<()> {
        // A null partial means the sub-accumulator saturated (overflow).
        if partial.is_null() {
            self.sums.set_overflowed(group_id);
            return Ok(());
        }
        self.sums.add_scalar(group_id, &partial)
    }

    fn partial_scalar(&self, group_id: usize) -> VortexResult<Scalar> {
        if self.sums.is_overflowed(group_id) {
            return Ok(Scalar::null(self.partial_dtype.clone()));
        }
        self.sums.value_scalar(group_id, Nullability::Nullable)
    }

    fn accumulate_partials(
        &mut self,
        partials: &ArrayRef,
        group_ids: &[u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Null partials are saturated groups, so they poison their target group rather than
        // contributing a value.
        let validity = partials.validity()?.execute_mask(partials.len(), ctx)?;
        self.sums
            .accumulate_nullable_partials(partials, group_ids, &validity, ctx)
    }

    fn flush_partials(&mut self) -> VortexResult<ArrayRef> {
        let overflowed = self.sums.take_overflowed();
        let validity = if overflowed.iter().any(|&overflowed| overflowed) {
            Validity::from_iter(overflowed.iter().map(|&overflowed| !overflowed))
        } else {
            Validity::AllValid
        };
        self.sums.take_values(validity)
    }
}

/// Invoke `f(row_idx, group_id)` for each non-null partial, saturating groups whose partial is
/// null.
fn for_each_valid_partial(
    group_ids: &[u32],
    validity: &Mask,
    overflowed: &mut [bool],
    mut f: impl FnMut(usize, usize, &mut [bool]),
) {
    if validity.all_true() {
        for (idx, &group_id) in group_ids.iter().enumerate() {
            let group = group_id as usize;
            if !overflowed[group] {
                f(idx, group, overflowed);
            }
        }
        return;
    }

    for (idx, (&group_id, valid)) in group_ids.iter().zip(validity.iter()).enumerate() {
        let group = group_id as usize;
        if !valid {
            overflowed[group] = true;
        } else if !overflowed[group] {
            f(idx, group, overflowed);
        }
    }
}

fn accumulate_partials_with<T: Copy>(
    values: &mut [T],
    overflowed: &mut [bool],
    group_ids: &[u32],
    validity: &Mask,
    partials: &[T],
    checked_add: fn(&mut T, T) -> bool,
) {
    for_each_valid_partial(group_ids, validity, overflowed, |idx, group, overflowed| {
        if checked_add(&mut values[group], partials[idx]) {
            overflowed[group] = true;
        }
    });
}

fn accumulate_decimal_partials<I>(
    values: &mut [I],
    overflowed: &mut [bool],
    group_ids: &[u32],
    validity: &Mask,
    partials: &DecimalArray,
    dtype: DecimalDType,
) where
    I: NativeDecimalType + CheckedAdd,
{
    match_each_decimal_value_type!(partials.values_type(), |T| {
        let partials = partials.buffer::<T>();
        for_each_valid_partial(group_ids, validity, overflowed, |idx, group, overflowed| {
            match <I as BigCast>::from(partials[idx]) {
                Some(value) => add_decimal(values, overflowed, group, value, dtype),
                None => overflowed[group] = true,
            }
        });
    });
}

/// Add `value` into one decimal group, saturating the group on overflow or precision loss.
pub(crate) fn add_decimal<T>(
    values: &mut [T],
    overflowed: &mut [bool],
    group_id: usize,
    value: T,
    dtype: DecimalDType,
) where
    T: NativeDecimalType + CheckedAdd,
{
    if overflowed[group_id] {
        return;
    }
    let Some(result) = values[group_id].checked_add(&value) else {
        overflowed[group_id] = true;
        return;
    };
    let precision = usize::from(dtype.precision());
    if T::MIN_BY_PRECISION[precision] <= result && result <= T::MAX_BY_PRECISION[precision] {
        values[group_id] = result;
    } else {
        overflowed[group_id] = true;
    }
}
