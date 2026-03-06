// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::BitAnd;

use itertools::Itertools;
use num_traits::ToPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::stats::Stat;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

#[derive(Clone, Debug)]
pub struct Sum;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SumOptions {
    checked: bool,
}

impl Display for SumOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.checked {
            write!(f, "checked")
        } else {
            write!(f, "unchecked")
        }
    }
}

impl AggregateFnVTable for Sum {
    type Options = SumOptions;
    type GroupState = SumGroupState;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.sum")
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        Stat::Sum
            .dtype(input_dtype)
            .ok_or_else(|| vortex_err!("Cannot sum {}", input_dtype))
    }

    fn state_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn state_new(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::GroupState> {
        let return_dtype = Stat::Sum
            .dtype(input_dtype)
            .ok_or_else(|| vortex_err!("Cannot sum {}", input_dtype))?;

        let initial = match &return_dtype {
            DType::Primitive(ptype, _) => match ptype {
                PType::U8 | PType::U16 | PType::U32 | PType::U64 => SumState::Unsigned(0),
                PType::I8 | PType::I16 | PType::I32 | PType::I64 => SumState::Signed(0),
                PType::F16 | PType::F32 | PType::F64 => SumState::Float(0.0),
            },
            DType::Decimal(decimal, _) => SumState::Decimal(DecimalValue::zero(decimal)),
            _ => vortex_panic!("Unsupported sum type"),
        };

        Ok(SumGroupState {
            checked: options.checked,
            return_dtype,
            current: Some(initial),
        })
    }

    fn state_merge(&self, state: &mut Self::GroupState, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }
        let checked = state.checked;
        let Some(ref mut inner) = state.current else {
            return Ok(());
        };
        let saturated = match inner {
            SumState::Unsigned(acc) => {
                let val = other
                    .as_primitive()
                    .typed_value::<u64>()
                    .ok_or_else(|| vortex_err!("Expected u64 scalar for unsigned sum merge"))?;
                add_u64(acc, val, checked)
            }
            SumState::Signed(acc) => {
                let val = other
                    .as_primitive()
                    .typed_value::<i64>()
                    .ok_or_else(|| vortex_err!("Expected i64 scalar for signed sum merge"))?;
                add_i64(acc, val, checked)
            }
            SumState::Float(acc) => {
                let val = other
                    .as_primitive()
                    .typed_value::<f64>()
                    .ok_or_else(|| vortex_err!("Expected f64 scalar for float sum merge"))?;
                *acc += val;
                false
            }
            SumState::Decimal(acc) => {
                let val = other
                    .as_decimal()
                    .decimal_value()
                    .ok_or_else(|| vortex_err!("Expected decimal scalar for decimal sum merge"))?;
                match acc.checked_add(&val) {
                    Some(r) => {
                        *acc = r;
                        false
                    }
                    None => true,
                }
            }
        };
        if saturated {
            state.current = None;
        }
        Ok(())
    }

    fn state_flush(&self, state: &mut Self::GroupState) -> VortexResult<Scalar> {
        let result = match &state.current {
            None => Scalar::null(state.return_dtype.as_nullable()),
            Some(SumState::Unsigned(v)) => Scalar::primitive(*v, Nullability::Nullable),
            Some(SumState::Signed(v)) => Scalar::primitive(*v, Nullability::Nullable),
            Some(SumState::Float(v)) => Scalar::primitive(*v, Nullability::Nullable),
            Some(SumState::Decimal(v)) => {
                let decimal_dtype = *state
                    .return_dtype
                    .as_decimal_opt()
                    .vortex_expect("return dtype must be decimal");
                Scalar::decimal(*v, decimal_dtype, Nullability::Nullable)
            }
        };

        // Reset the state
        state.current = Some(make_zero_state(&state.return_dtype));

        Ok(result)
    }

    fn state_is_saturated(&self, state: &Self::GroupState) -> bool {
        state.current.is_none()
    }

    fn state_accumulate(
        &self,
        state: &mut Self::GroupState,
        batch: &Canonical,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let checked = state.checked;
        let mut inner = match state.current.take() {
            Some(inner) => inner,
            None => return Ok(()),
        };

        let result = match batch {
            Canonical::Primitive(p) => accumulate_primitive(&mut inner, p, checked),
            Canonical::Bool(b) => accumulate_bool(&mut inner, b, checked),
            Canonical::Decimal(d) => accumulate_decimal(&mut inner, d),
            _ => vortex_bail!("Unsupported canonical type for sum: {}", batch.dtype()),
        };

        match result {
            Ok(false) => state.current = Some(inner),
            Ok(true) => {} // saturated: current stays None
            Err(e) => {
                state.current = Some(inner);
                return Err(e);
            }
        }
        Ok(())
    }

    fn finalize(&self, states: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(states)
    }

    fn finalize_scalar(&self, state: Scalar) -> VortexResult<Scalar> {
        Ok(state)
    }
}

/// The group state for a sum aggregate, containing the accumulated value and configuration
/// needed for reset/result without external context.
pub struct SumGroupState {
    checked: bool,
    return_dtype: DType,
    /// The current accumulated state, or `None` if saturated (checked overflow).
    current: Option<SumState>,
}

/// The accumulated sum value.
///
// TODO(ngates): instead of an enum, we should use a Box<dyn State> to avoid dispatcher over the
//  input type every time? Perhaps?
pub enum SumState {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Decimal(DecimalValue),
}

fn make_zero_state(return_dtype: &DType) -> SumState {
    match return_dtype {
        DType::Primitive(ptype, _) => match ptype {
            PType::U8 | PType::U16 | PType::U32 | PType::U64 => SumState::Unsigned(0),
            PType::I8 | PType::I16 | PType::I32 | PType::I64 => SumState::Signed(0),
            PType::F16 | PType::F32 | PType::F64 => SumState::Float(0.0),
        },
        DType::Decimal(decimal, _) => SumState::Decimal(DecimalValue::zero(decimal)),
        _ => vortex_panic!("Unsupported sum type"),
    }
}

/// Add `val` to `acc`, returning true if overflow occurred (checked mode) or wrapping (unchecked).
fn add_u64(acc: &mut u64, val: u64, checked: bool) -> bool {
    if checked {
        match acc.checked_add(val) {
            Some(r) => {
                *acc = r;
                false
            }
            None => true,
        }
    } else {
        *acc = acc.wrapping_add(val);
        false
    }
}

fn add_i64(acc: &mut i64, val: i64, checked: bool) -> bool {
    if checked {
        match acc.checked_add(val) {
            Some(r) => {
                *acc = r;
                false
            }
            None => true,
        }
    } else {
        *acc = acc.wrapping_add(val);
        false
    }
}

/// Accumulate a primitive array into the sum state.
/// Returns Ok(true) if saturated (overflow), Ok(false) if not.
fn accumulate_primitive(
    inner: &mut SumState,
    p: &PrimitiveArray,
    checked: bool,
) -> VortexResult<bool> {
    let mask = p.validity_mask()?;
    match mask.bit_buffer() {
        AllOr::None => Ok(false),
        AllOr::All => accumulate_primitive_all(inner, p, checked),
        AllOr::Some(validity) => accumulate_primitive_valid(inner, p, validity, checked),
    }
}

fn accumulate_primitive_all(
    inner: &mut SumState,
    p: &PrimitiveArray,
    checked: bool,
) -> VortexResult<bool> {
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| {
                for &v in p.as_slice::<T>() {
                    if add_u64(acc, v.to_u64().vortex_expect("unsigned to u64"), checked) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            signed: |_T| { vortex_panic!("unsigned sum state with signed input") },
            floating: |_T| { vortex_panic!("unsigned sum state with float input") }
        ),
        SumState::Signed(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("signed sum state with unsigned input") },
            signed: |T| {
                for &v in p.as_slice::<T>() {
                    if add_i64(acc, v.to_i64().vortex_expect("signed to i64"), checked) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            floating: |_T| { vortex_panic!("signed sum state with float input") }
        ),
        SumState::Float(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("float sum state with unsigned input") },
            signed: |_T| { vortex_panic!("float sum state with signed input") },
            floating: |T| {
                for &v in p.as_slice::<T>() {
                    *acc += ToPrimitive::to_f64(&v).vortex_expect("float to f64");
                }
                Ok(false)
            }
        ),
        SumState::Decimal(_) => vortex_panic!("decimal sum state with primitive input"),
    }
}

fn accumulate_primitive_valid(
    inner: &mut SumState,
    p: &PrimitiveArray,
    validity: &vortex_buffer::BitBuffer,
    checked: bool,
) -> VortexResult<bool> {
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| {
                for (&v, valid) in p.as_slice::<T>().iter().zip_eq(validity.iter()) {
                    if valid && add_u64(acc, v.to_u64().vortex_expect("unsigned to u64"), checked) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            signed: |_T| { vortex_panic!("unsigned sum state with signed input") },
            floating: |_T| { vortex_panic!("unsigned sum state with float input") }
        ),
        SumState::Signed(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("signed sum state with unsigned input") },
            signed: |T| {
                for (&v, valid) in p.as_slice::<T>().iter().zip_eq(validity.iter()) {
                    if valid && add_i64(acc, v.to_i64().vortex_expect("signed to i64"), checked) {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            floating: |_T| { vortex_panic!("signed sum state with float input") }
        ),
        SumState::Float(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |_T| { vortex_panic!("float sum state with unsigned input") },
            signed: |_T| { vortex_panic!("float sum state with signed input") },
            floating: |T| {
                for (&v, valid) in p.as_slice::<T>().iter().zip_eq(validity.iter()) {
                    if valid {
                        *acc += ToPrimitive::to_f64(&v).vortex_expect("float to f64");
                    }
                }
                Ok(false)
            }
        ),
        SumState::Decimal(_) => vortex_panic!("decimal sum state with primitive input"),
    }
}

/// Accumulate a boolean array into the sum state (counts true values as u64).
/// Returns Ok(true) if saturated (overflow), Ok(false) if not.
fn accumulate_bool(inner: &mut SumState, b: &BoolArray, checked: bool) -> VortexResult<bool> {
    let SumState::Unsigned(acc) = inner else {
        vortex_panic!("expected unsigned sum state for bool input");
    };

    let mask = b.validity_mask()?;
    let true_count = match mask.bit_buffer() {
        AllOr::None => return Ok(false),
        AllOr::All => b.to_bit_buffer().true_count() as u64,
        AllOr::Some(validity) => b.to_bit_buffer().bitand(validity).true_count() as u64,
    };

    Ok(add_u64(acc, true_count, checked))
}

/// Accumulate a decimal array into the sum state.
/// Returns Ok(true) if saturated (overflow), Ok(false) if not.
fn accumulate_decimal(inner: &mut SumState, d: &DecimalArray) -> VortexResult<bool> {
    let SumState::Decimal(acc) = inner else {
        vortex_panic!("expected decimal sum state for decimal input");
    };

    let mask = d.validity_mask()?;
    match mask.bit_buffer() {
        AllOr::None => Ok(false),
        AllOr::All => match_each_decimal_value_type!(d.values_type(), |T| {
            for &v in d.buffer::<T>().iter() {
                match acc.checked_add(&DecimalValue::from(v)) {
                    Some(r) => *acc = r,
                    None => return Ok(true),
                }
            }
            Ok(false)
        }),
        AllOr::Some(validity) => match_each_decimal_value_type!(d.values_type(), |T| {
            for (&v, valid) in d.buffer::<T>().iter().zip_eq(validity.iter()) {
                if valid {
                    match acc.checked_add(&DecimalValue::from(v)) {
                        Some(r) => *acc = r,
                        None => return Ok(true),
                    }
                }
            }
            Ok(false)
        }),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::DynGroupedAccumulator;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::fns::sum::Sum;
    use crate::aggregate_fn::fns::sum::SumOptions;
    use crate::arrays::BoolArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    fn session() -> VortexSession {
        VortexSession::empty()
    }

    fn checked_opts() -> SumOptions {
        SumOptions { checked: true }
    }

    fn unchecked_opts() -> SumOptions {
        SumOptions { checked: false }
    }

    fn run_sum(batch: &ArrayRef, options: &SumOptions) -> VortexResult<Scalar> {
        let mut acc = Accumulator::try_new(Sum, options.clone(), batch.dtype().clone(), session())?;
        acc.accumulate(batch)?;
        acc.finish()
    }

    // Primitive sum tests

    #[test]
    fn sum_i32() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(10));
        Ok(())
    }

    #[test]
    fn sum_u8() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![10u8, 20, 30], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(60));
        Ok(())
    }

    #[test]
    fn sum_f64() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.5f64, 2.5, 3.0], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(7.0));
        Ok(())
    }

    #[test]
    fn sum_with_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(6));
        Ok(())
    }

    #[test]
    fn sum_all_null() -> VortexResult<()> {
        // Arrow semantics: sum of all nulls is zero (identity element)
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    // Empty accumulator tests

    #[test]
    fn sum_empty_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_empty_f64_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    // Multi-batch and reset tests

    #[test]
    fn sum_multi_batch() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1)?;

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2)?;

        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(48));
        Ok(())
    }

    #[test]
    fn sum_finish_resets_state() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;

        let batch1 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        acc.accumulate(&batch1)?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().typed_value::<i64>(), Some(30));

        let batch2 = PrimitiveArray::new(buffer![3i32, 6, 9], Validity::NonNullable).into_array();
        acc.accumulate(&batch2)?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().typed_value::<i64>(), Some(18));
        Ok(())
    }

    // State merge tests (vtable-level)

    #[test]
    fn sum_state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut state = Sum.state_new(&checked_opts(), &dtype)?;

        let scalar1 = Scalar::primitive(100i64, Nullability::Nullable);
        Sum.state_merge(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(50i64, Nullability::Nullable);
        Sum.state_merge(&mut state, scalar2)?;

        let result = Sum.state_flush(&mut state)?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(150));
        Ok(())
    }

    // Overflow tests

    #[test]
    fn sum_checked_overflow() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &checked_opts())?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow_is_saturated() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;
        assert!(!acc.is_saturated());

        let batch =
            PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        acc.accumulate(&batch)?;
        assert!(acc.is_saturated());

        // finish resets state, clearing saturation
        drop(acc.finish()?);
        assert!(!acc.is_saturated());
        Ok(())
    }

    #[test]
    fn sum_unchecked_wrapping() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = run_sum(&arr, &unchecked_opts())?;
        assert_eq!(
            result.as_primitive().typed_value::<i64>(),
            Some(i64::MAX.wrapping_add(1))
        );
        Ok(())
    }

    // Boolean sum tests

    #[test]
    fn sum_bool_all_true() -> VortexResult<()> {
        let arr: BoolArray = [true, true, true].into_iter().collect();
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_mixed() -> VortexResult<()> {
        let arr: BoolArray = [true, false, true, false, true].into_iter().collect();
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_all_false() -> VortexResult<()> {
        let arr: BoolArray = [false, false, false].into_iter().collect();
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_with_nulls() -> VortexResult<()> {
        let arr = BoolArray::from_iter([Some(true), None, Some(true), Some(false)]);
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(2));
        Ok(())
    }

    #[test]
    fn sum_bool_all_null() -> VortexResult<()> {
        // Arrow semantics: sum of all nulls is zero (identity element)
        let arr = BoolArray::from_iter([None::<bool>, None, None]);
        let result = run_sum(&arr.into_array(), &checked_opts())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_empty_produces_zero() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_finish_resets_state() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, checked_opts(), dtype, session())?;

        let batch1: BoolArray = [true, true, false].into_iter().collect();
        acc.accumulate(&batch1.into_array())?;
        let result1 = acc.finish()?;
        assert_eq!(result1.as_primitive().typed_value::<u64>(), Some(2));

        let batch2: BoolArray = [false, true].into_iter().collect();
        acc.accumulate(&batch2.into_array())?;
        let result2 = acc.finish()?;
        assert_eq!(result2.as_primitive().typed_value::<u64>(), Some(1));
        Ok(())
    }

    #[test]
    fn sum_bool_return_dtype() -> VortexResult<()> {
        let dtype = Sum.return_dtype(&checked_opts(), &DType::Bool(Nullability::NonNullable))?;
        assert_eq!(dtype, DType::Primitive(PType::U64, Nullability::Nullable));
        Ok(())
    }

    // Grouped sum tests

    fn run_grouped_sum(groups: &ArrayRef, elem_dtype: &DType) -> VortexResult<ArrayRef> {
        let mut acc =
            GroupedAccumulator::try_new(Sum, checked_opts(), elem_dtype.clone(), session())?;
        acc.accumulate_list(groups)?;
        acc.finish()
    }

    #[test]
    fn grouped_sum_fixed_size_list() -> VortexResult<()> {
        // Groups: [[1,2,3], [4,5,6]] -> sums [6, 15]
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
        // Groups: [[Some(1), None, Some(3)], [None, Some(5), Some(6)]] -> sums [4, 11]
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5), Some(6)])
                .into_array();
        let groups = FixedSizeListArray::try_new(elements, 3, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(4i64), Some(11i64)]).into_array();
        assert_arrays_eq!(&result, &expected);
        Ok(())
    }

    #[test]
    fn grouped_sum_with_null_group() -> VortexResult<()> {
        // Groups: [[1,2,3], null, [7,8,9]] -> sums [6, null, 24]
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
        // Groups: [[None, None], [Some(3), Some(4)]] -> sums [0, 7] (Arrow semantics)
        let elements =
            PrimitiveArray::from_option_iter([None::<i32>, None, Some(3), Some(4)]).into_array();
        let groups = FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 2)?;

        let elem_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let result = run_grouped_sum(&groups.into_array(), &elem_dtype)?;

        let expected = PrimitiveArray::from_option_iter([Some(0i64), Some(7i64)]).into_array();
        assert_arrays_eq!(&result, &expected);
        Ok(())
    }

    #[test]
    fn grouped_sum_bool() -> VortexResult<()> {
        // Groups: [[true, false, true], [true, true, true]] -> sums [2, 3]
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
        let elem_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = GroupedAccumulator::try_new(Sum, checked_opts(), elem_dtype, session())?;

        // First batch: [[1, 2], [3, 4]]
        let elements1 =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let groups1 = FixedSizeListArray::try_new(elements1, 2, Validity::NonNullable, 2)?;
        acc.accumulate_list(&groups1.into_array())?;
        let result1 = acc.finish()?;

        let expected1 = PrimitiveArray::from_option_iter([Some(3i64), Some(7i64)]).into_array();
        assert_arrays_eq!(&result1, &expected1);

        // Second batch after reset: [[10, 20]]
        let elements2 = PrimitiveArray::new(buffer![10i32, 20], Validity::NonNullable).into_array();
        let groups2 = FixedSizeListArray::try_new(elements2, 2, Validity::NonNullable, 1)?;
        acc.accumulate_list(&groups2.into_array())?;
        let result2 = acc.finish()?;

        let expected2 = PrimitiveArray::from_option_iter([Some(30i64)]).into_array();
        assert_arrays_eq!(&result2, &expected2);
        Ok(())
    }
}
