// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
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

impl AggregateFnVTable for Sum {
    type Options = EmptyOptions;
    type Partial = SumPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new_ref("vortex.sum")
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        Stat::Sum
            .dtype(input_dtype)
            .ok_or_else(|| vortex_err!("Cannot sum {}", input_dtype))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> VortexResult<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        let return_dtype = Stat::Sum
            .dtype(input_dtype)
            .ok_or_else(|| vortex_err!("Cannot sum {}", input_dtype))?;

        let initial = make_zero_state(&return_dtype);

        Ok(SumPartial {
            return_dtype,
            current: Some(initial),
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
            SumState::Decimal(acc) => {
                let val = other
                    .as_decimal()
                    .decimal_value()
                    .vortex_expect("checked non-null");
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
            partial.current = None;
        }
        Ok(())
    }

    fn flush(&self, partial: &mut Self::Partial) -> VortexResult<Scalar> {
        let result = match &partial.current {
            None => Scalar::null(partial.return_dtype.as_nullable()),
            Some(SumState::Unsigned(v)) => Scalar::primitive(*v, Nullability::Nullable),
            Some(SumState::Signed(v)) => Scalar::primitive(*v, Nullability::Nullable),
            Some(SumState::Float(v)) => Scalar::primitive(*v, Nullability::Nullable),
            Some(SumState::Decimal(v)) => {
                let decimal_dtype = *partial
                    .return_dtype
                    .as_decimal_opt()
                    .vortex_expect("return dtype must be decimal");
                Scalar::decimal(*v, decimal_dtype, Nullability::Nullable)
            }
        };

        // Reset the state
        partial.current = Some(make_zero_state(&partial.return_dtype));

        Ok(result)
    }

    #[inline]
    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        partial.current.is_none()
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mut inner = match partial.current.take() {
            Some(inner) => inner,
            None => return Ok(()),
        };

        let result = match batch {
            Columnar::Canonical(c) => match c {
                Canonical::Primitive(p) => accumulate_primitive(&mut inner, p),
                Canonical::Bool(b) => accumulate_bool(&mut inner, b),
                Canonical::Decimal(d) => accumulate_decimal(&mut inner, d),
                _ => vortex_bail!("Unsupported canonical type for sum: {}", batch.dtype()),
            },
            Columnar::Constant(c) => accumulate_constant(&mut inner, c),
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

    fn finalize_scalar(&self, partial: Scalar) -> VortexResult<Scalar> {
        Ok(partial)
    }
}

/// The group state for a sum aggregate, containing the accumulated value and configuration
/// needed for reset/result without external context.
pub struct SumPartial {
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

fn accumulate_primitive(inner: &mut SumState, p: &PrimitiveArray) -> VortexResult<bool> {
    let mask = p.validity_mask()?;
    match mask.bit_buffer() {
        AllOr::None => Ok(false),
        AllOr::All => accumulate_primitive_all(inner, p),
        AllOr::Some(validity) => accumulate_primitive_valid(inner, p, validity),
    }
}

fn accumulate_primitive_all(inner: &mut SumState, p: &PrimitiveArray) -> VortexResult<bool> {
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| {
                for &v in p.as_slice::<T>() {
                    if checked_add_u64(acc, v.to_u64().vortex_expect("unsigned to u64")) {
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
                    if checked_add_i64(acc, v.to_i64().vortex_expect("signed to i64")) {
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
) -> VortexResult<bool> {
    match inner {
        SumState::Unsigned(acc) => match_each_native_ptype!(p.ptype(),
            unsigned: |T| {
                for (&v, valid) in p.as_slice::<T>().iter().zip_eq(validity.iter()) {
                    if valid && checked_add_u64(acc, v.to_u64().vortex_expect("unsigned to u64")) {
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
                    if valid && checked_add_i64(acc, v.to_i64().vortex_expect("signed to i64")) {
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

fn accumulate_bool(inner: &mut SumState, b: &BoolArray) -> VortexResult<bool> {
    let SumState::Unsigned(acc) = inner else {
        vortex_panic!("expected unsigned sum state for bool input");
    };

    let mask = b.validity_mask()?;
    let true_count = match mask.bit_buffer() {
        AllOr::None => return Ok(false),
        AllOr::All => b.to_bit_buffer().true_count() as u64,
        AllOr::Some(validity) => b.to_bit_buffer().bitand(validity).true_count() as u64,
    };

    Ok(checked_add_u64(acc, true_count))
}

/// Accumulate a constant array into the sum state.
/// Computes `scalar * len` and adds to the accumulator.
/// Returns Ok(true) if saturated (overflow), Ok(false) if not.
fn accumulate_constant(inner: &mut SumState, c: &ConstantArray) -> VortexResult<bool> {
    let scalar = c.scalar();
    if scalar.is_null() || c.is_empty() {
        return Ok(false);
    }
    let len = c.len();

    match scalar.dtype() {
        DType::Bool(_) => {
            let SumState::Unsigned(acc) = inner else {
                vortex_panic!("expected unsigned sum state for bool input");
            };
            let val = scalar
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("Expected non-null bool scalar for sum"))?;
            if val {
                Ok(checked_add_u64(acc, len as u64))
            } else {
                Ok(false)
            }
        }
        DType::Primitive(..) => {
            let pvalue = scalar
                .as_primitive()
                .pvalue()
                .ok_or_else(|| vortex_err!("Expected non-null primitive scalar for sum"))?;
            match inner {
                SumState::Unsigned(acc) => {
                    let val = pvalue.cast::<u64>()?;
                    match val.checked_mul(len as u64) {
                        Some(product) => Ok(checked_add_u64(acc, product)),
                        None => Ok(true),
                    }
                }
                SumState::Signed(acc) => {
                    let val = pvalue.cast::<i64>()?;
                    match i64::try_from(len).ok().and_then(|l| val.checked_mul(l)) {
                        Some(product) => Ok(checked_add_i64(acc, product)),
                        None => Ok(true),
                    }
                }
                SumState::Float(acc) => {
                    let val = pvalue.cast::<f64>()?;
                    *acc += val * len as f64;
                    Ok(false)
                }
                SumState::Decimal(_) => {
                    vortex_panic!("decimal sum state with primitive input")
                }
            }
        }
        DType::Decimal(..) => {
            let SumState::Decimal(acc) = inner else {
                vortex_panic!("expected decimal sum state for decimal input");
            };
            let val = scalar
                .as_decimal()
                .decimal_value()
                .ok_or_else(|| vortex_err!("Expected non-null decimal scalar for sum"))?;
            let len_decimal = DecimalValue::from(len as i128);
            match val.checked_mul(&len_decimal) {
                Some(product) => match acc.checked_add(&product) {
                    Some(r) => {
                        *acc = r;
                        Ok(false)
                    }
                    None => Ok(true),
                },
                None => Ok(true),
            }
        }
        _ => vortex_bail!("Unsupported constant type for sum: {}", scalar.dtype()),
    }
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
    use crate::aggregate_fn::EmptyOptions;
    use crate::aggregate_fn::GroupedAccumulator;
    use crate::aggregate_fn::fns::sum::Sum;
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

    fn run_sum(batch: &ArrayRef) -> VortexResult<Scalar> {
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, batch.dtype().clone(), session())?;
        acc.accumulate(batch)?;
        acc.finish()
    }

    // Primitive sum tests

    #[test]
    fn sum_i32() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let result = run_sum(&arr)?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(10));
        Ok(())
    }

    #[test]
    fn sum_u8() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![10u8, 20, 30], Validity::NonNullable).into_array();
        let result = run_sum(&arr)?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(60));
        Ok(())
    }

    #[test]
    fn sum_f64() -> VortexResult<()> {
        let arr =
            PrimitiveArray::new(buffer![1.5f64, 2.5, 3.0], Validity::NonNullable).into_array();
        let result = run_sum(&arr)?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(7.0));
        Ok(())
    }

    #[test]
    fn sum_with_nulls() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(2i32), None, Some(4)]).into_array();
        let result = run_sum(&arr)?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(6));
        Ok(())
    }

    #[test]
    fn sum_all_null() -> VortexResult<()> {
        // Arrow semantics: sum of all nulls is zero (identity element)
        let arr = PrimitiveArray::from_option_iter([None::<i32>, None, None]).into_array();
        let result = run_sum(&arr)?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    // Empty accumulator tests

    #[test]
    fn sum_empty_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype, session())?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_empty_f64_produces_zero() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype, session())?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<f64>(), Some(0.0));
        Ok(())
    }

    // Multi-batch and reset tests

    #[test]
    fn sum_multi_batch() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype, session())?;

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
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype, session())?;

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
        let mut state = Sum.empty_partial(&EmptyOptions, &dtype)?;

        let scalar1 = Scalar::primitive(100i64, Nullability::Nullable);
        Sum.combine_partials(&mut state, scalar1)?;

        let scalar2 = Scalar::primitive(50i64, Nullability::Nullable);
        Sum.combine_partials(&mut state, scalar2)?;

        let result = Sum.flush(&mut state)?;
        assert_eq!(result.as_primitive().typed_value::<i64>(), Some(150));
        Ok(())
    }

    // Overflow tests

    #[test]
    fn sum_checked_overflow() -> VortexResult<()> {
        let arr = PrimitiveArray::new(buffer![i64::MAX, 1i64], Validity::NonNullable).into_array();
        let result = run_sum(&arr)?;
        assert!(result.is_null());
        Ok(())
    }

    #[test]
    fn sum_checked_overflow_is_saturated() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype, session())?;
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

    // Boolean sum tests

    #[test]
    fn sum_bool_all_true() -> VortexResult<()> {
        let arr: BoolArray = [true, true, true].into_iter().collect();
        let result = run_sum(&arr.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_mixed() -> VortexResult<()> {
        let arr: BoolArray = [true, false, true, false, true].into_iter().collect();
        let result = run_sum(&arr.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(3));
        Ok(())
    }

    #[test]
    fn sum_bool_all_false() -> VortexResult<()> {
        let arr: BoolArray = [false, false, false].into_iter().collect();
        let result = run_sum(&arr.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_with_nulls() -> VortexResult<()> {
        let arr = BoolArray::from_iter([Some(true), None, Some(true), Some(false)]);
        let result = run_sum(&arr.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(2));
        Ok(())
    }

    #[test]
    fn sum_bool_all_null() -> VortexResult<()> {
        // Arrow semantics: sum of all nulls is zero (identity element)
        let arr = BoolArray::from_iter([None::<bool>, None, None]);
        let result = run_sum(&arr.into_array())?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_empty_produces_zero() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype, session())?;
        let result = acc.finish()?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(0));
        Ok(())
    }

    #[test]
    fn sum_bool_finish_resets_state() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let mut acc = Accumulator::try_new(Sum, EmptyOptions, dtype, session())?;

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
        let dtype = Sum.return_dtype(&EmptyOptions, &DType::Bool(Nullability::NonNullable))?;
        assert_eq!(dtype, DType::Primitive(PType::U64, Nullability::Nullable));
        Ok(())
    }

    // Grouped sum tests

    fn run_grouped_sum(groups: &ArrayRef, elem_dtype: &DType) -> VortexResult<ArrayRef> {
        let mut acc =
            GroupedAccumulator::try_new(Sum, EmptyOptions, elem_dtype.clone(), session())?;
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
        let mut acc = GroupedAccumulator::try_new(Sum, EmptyOptions, elem_dtype, session())?;

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
