// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arithmetic implementations for PrimitiveScalar enum.

use vortex_dtype::half::f16;
use vortex_error::vortex_panic;
use vortex_vector::PrimitiveDatum;
use vortex_vector::match_each_float_pscalar_pair;
use vortex_vector::match_each_integer_pscalar_pair;
use vortex_vector::primitive::PScalar;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveScalar;
use vortex_vector::primitive::PrimitiveVector;

use crate::arithmetic::Arithmetic;
use crate::arithmetic::CheckedArithmetic;

impl<Op> CheckedArithmetic<Op> for &PrimitiveScalar
where
    for<'a> &'a PScalar<i8>: CheckedArithmetic<Op, &'a PScalar<i8>, Output = PScalar<i8>>,
    for<'a> &'a PScalar<i16>: CheckedArithmetic<Op, &'a PScalar<i16>, Output = PScalar<i16>>,
    for<'a> &'a PScalar<i32>: CheckedArithmetic<Op, &'a PScalar<i32>, Output = PScalar<i32>>,
    for<'a> &'a PScalar<i64>: CheckedArithmetic<Op, &'a PScalar<i64>, Output = PScalar<i64>>,
    for<'a> &'a PScalar<u8>: CheckedArithmetic<Op, &'a PScalar<u8>, Output = PScalar<u8>>,
    for<'a> &'a PScalar<u16>: CheckedArithmetic<Op, &'a PScalar<u16>, Output = PScalar<u16>>,
    for<'a> &'a PScalar<u32>: CheckedArithmetic<Op, &'a PScalar<u32>, Output = PScalar<u32>>,
    for<'a> &'a PScalar<u64>: CheckedArithmetic<Op, &'a PScalar<u64>, Output = PScalar<u64>>,
{
    type Output = PrimitiveScalar;

    fn checked_eval(self, rhs: &PrimitiveScalar) -> Option<Self::Output> {
        match_each_integer_pscalar_pair!(
            (&self, &rhs),
            |l, r| { CheckedArithmetic::<Op, _>::checked_eval(l, r).map(Into::into) },
            { vortex_panic!("cannot compare primitive scalar of different types") }
        )
    }
}

impl<Op> Arithmetic<Op, &PrimitiveScalar> for &PrimitiveScalar
where
    for<'a> &'a PScalar<f16>: Arithmetic<Op, &'a PScalar<f16>, Output = PScalar<f16>>,
    for<'a> &'a PScalar<f32>: Arithmetic<Op, &'a PScalar<f32>, Output = PScalar<f32>>,
    for<'a> &'a PScalar<f64>: Arithmetic<Op, &'a PScalar<f64>, Output = PScalar<f64>>,
{
    type Output = PrimitiveScalar;

    fn eval(self, rhs: &PrimitiveScalar) -> Self::Output {
        match_each_float_pscalar_pair!(
            (self, rhs),
            |l, r| { Arithmetic::<Op, _>::eval(l, r).into() },
            {
                vortex_panic!(
                    "Cannot perform arithmetic on PrimitiveScalars of different types: {:?} and {:?}",
                    self,
                    rhs
                )
            }
        )
    }
}

/// Scalar on LHS, owned Vector on RHS - modifies vector in place.
/// Returns a scalar if the input scalar is null.
impl<Op> Arithmetic<Op, PrimitiveVector> for &PrimitiveScalar
where
    for<'a> &'a f16: Arithmetic<Op, PVector<f16>, Output = PVector<f16>>,
    for<'a> &'a f32: Arithmetic<Op, PVector<f32>, Output = PVector<f32>>,
    for<'a> &'a f64: Arithmetic<Op, PVector<f64>, Output = PVector<f64>>,
{
    type Output = PrimitiveDatum;

    fn eval(self, rhs: PrimitiveVector) -> Self::Output {
        match (self, rhs) {
            (PrimitiveScalar::F16(s), PrimitiveVector::F16(v)) => match s.value() {
                Some(scalar_val) => {
                    PrimitiveDatum::Vector(Arithmetic::<Op, _>::eval(&scalar_val, v).into())
                }
                None => PrimitiveDatum::Scalar(s.clone().into()),
            },
            (PrimitiveScalar::F32(s), PrimitiveVector::F32(v)) => match s.value() {
                Some(scalar_val) => {
                    PrimitiveDatum::Vector(Arithmetic::<Op, _>::eval(&scalar_val, v).into())
                }
                None => PrimitiveDatum::Scalar(s.clone().into()),
            },
            (PrimitiveScalar::F64(s), PrimitiveVector::F64(v)) => match s.value() {
                Some(scalar_val) => {
                    PrimitiveDatum::Vector(Arithmetic::<Op, _>::eval(&scalar_val, v).into())
                }
                None => PrimitiveDatum::Scalar(s.clone().into()),
            },
            (s, v) => vortex_panic!(
                "Cannot perform arithmetic between scalar {:?} and vector {:?}",
                s,
                v
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_vector::primitive::PScalar;

    use super::*;
    use crate::arithmetic::Add;

    #[test]
    fn test_checked_add_i32() {
        let left: PrimitiveScalar = PScalar::new(Some(5i32)).into();
        let right: PrimitiveScalar = PScalar::new(Some(3i32)).into();

        let result = CheckedArithmetic::<Add, _>::checked_eval(&left, &right).unwrap();
        if let PrimitiveScalar::I32(v) = result {
            assert_eq!(v.value(), Some(8));
        } else {
            panic!("Expected I32 result");
        }
    }

    #[test]
    fn test_checked_add_overflow() {
        let left: PrimitiveScalar = PScalar::new(Some(i32::MAX)).into();
        let right: PrimitiveScalar = PScalar::new(Some(1i32)).into();

        let result = CheckedArithmetic::<Add, _>::checked_eval(&left, &right);
        assert!(result.is_none());
    }

    #[test]
    fn test_float_add() {
        let left: PrimitiveScalar = PScalar::new(Some(1.5f64)).into();
        let right: PrimitiveScalar = PScalar::new(Some(2.5f64)).into();

        let result = Arithmetic::<Add, _>::eval(&left, &right);
        if let PrimitiveScalar::F64(v) = result {
            assert_eq!(v.value(), Some(4.0));
        } else {
            panic!("Expected F64 result");
        }
    }
}
