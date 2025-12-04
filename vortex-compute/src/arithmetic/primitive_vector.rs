// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arithmetic implementations for PrimitiveVector enum.

use vortex_dtype::half::f16;
use vortex_error::vortex_panic;
use vortex_vector::PrimitiveDatum;
use vortex_vector::match_each_float_pvector_pair;
use vortex_vector::match_each_integer_pvector_pair;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveScalar;
use vortex_vector::primitive::PrimitiveVector;

use crate::arithmetic::Arithmetic;
use crate::arithmetic::CheckedArithmetic;

impl<Op> CheckedArithmetic<Op, &PrimitiveVector> for PrimitiveVector
where
    for<'a> PVector<i8>: CheckedArithmetic<Op, &'a PVector<i8>, Output = PVector<i8>>,
    for<'a> PVector<i16>: CheckedArithmetic<Op, &'a PVector<i16>, Output = PVector<i16>>,
    for<'a> PVector<i32>: CheckedArithmetic<Op, &'a PVector<i32>, Output = PVector<i32>>,
    for<'a> PVector<i64>: CheckedArithmetic<Op, &'a PVector<i64>, Output = PVector<i64>>,
    for<'a> PVector<u8>: CheckedArithmetic<Op, &'a PVector<u8>, Output = PVector<u8>>,
    for<'a> PVector<u16>: CheckedArithmetic<Op, &'a PVector<u16>, Output = PVector<u16>>,
    for<'a> PVector<u32>: CheckedArithmetic<Op, &'a PVector<u32>, Output = PVector<u32>>,
    for<'a> PVector<u64>: CheckedArithmetic<Op, &'a PVector<u64>, Output = PVector<u64>>,
{
    type Output = PrimitiveVector;

    fn checked_eval(self, rhs: &PrimitiveVector) -> Option<Self::Output> {
        match_each_integer_pvector_pair!(
            (self, &rhs),
            |l, r| { CheckedArithmetic::<Op, _>::checked_eval(l, r).map(Into::into) },
            { vortex_panic!("dont use checked arithmetic for floats") }
        )
    }
}

impl<Op> Arithmetic<Op, &PrimitiveVector> for PrimitiveVector
where
    for<'a> PVector<f16>: Arithmetic<Op, &'a PVector<f16>, Output = PVector<f16>>,
    for<'a> PVector<f32>: Arithmetic<Op, &'a PVector<f32>, Output = PVector<f32>>,
    for<'a> PVector<f64>: Arithmetic<Op, &'a PVector<f64>, Output = PVector<f64>>,
{
    type Output = PrimitiveVector;

    fn eval(self, rhs: &PrimitiveVector) -> Self::Output {
        match_each_float_pvector_pair!(
            (self, rhs),
            |l, r| { Arithmetic::<Op, _>::eval(l, r).into() },
            |l, r| {
                vortex_panic!(
                    "Cannot perform arithmetic on PrimitiveVectors of different types: {:?} and {:?}",
                    l,
                    r
                )
            }
        )
    }
}

/// Vector on LHS, Scalar on RHS - modifies vector in place.
/// Returns a scalar if the input scalar is null.
impl<Op> Arithmetic<Op, &PrimitiveScalar> for PrimitiveVector
where
    for<'a> PVector<f16>: Arithmetic<Op, &'a f16, Output = PVector<f16>>,
    for<'a> PVector<f32>: Arithmetic<Op, &'a f32, Output = PVector<f32>>,
    for<'a> PVector<f64>: Arithmetic<Op, &'a f64, Output = PVector<f64>>,
{
    type Output = PrimitiveDatum;

    fn eval(self, rhs: &PrimitiveScalar) -> Self::Output {
        match (self, rhs) {
            (PrimitiveVector::F16(v), PrimitiveScalar::F16(s)) => match s.value() {
                Some(scalar_val) => {
                    PrimitiveDatum::Vector(Arithmetic::<Op, _>::eval(v, &scalar_val).into())
                }
                None => PrimitiveDatum::Scalar(s.clone().into()),
            },
            (PrimitiveVector::F32(v), PrimitiveScalar::F32(s)) => match s.value() {
                Some(scalar_val) => {
                    PrimitiveDatum::Vector(Arithmetic::<Op, _>::eval(v, &scalar_val).into())
                }
                None => PrimitiveDatum::Scalar(s.clone().into()),
            },
            (PrimitiveVector::F64(v), PrimitiveScalar::F64(s)) => match s.value() {
                Some(scalar_val) => {
                    PrimitiveDatum::Vector(Arithmetic::<Op, _>::eval(v, &scalar_val).into())
                }
                None => PrimitiveDatum::Scalar(s.clone().into()),
            },
            (v, s) => vortex_panic!(
                "Cannot perform arithmetic between vector {:?} and scalar {:?}",
                v,
                s
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_vector::VectorMutOps;
    use vortex_vector::VectorOps;
    use vortex_vector::primitive::PVectorMut;

    use super::*;
    use crate::arithmetic::Add;

    #[test]
    fn test_checked_add_i32() {
        let left: PrimitiveVector = PVectorMut::from_iter([1i32, 2, 3].map(Some))
            .freeze()
            .into();
        let right: PrimitiveVector = PVectorMut::from_iter([10i32, 20, 30].map(Some))
            .freeze()
            .into();

        let result = CheckedArithmetic::<Add, _>::checked_eval(left, &right).unwrap();
        if let PrimitiveVector::I32(v) = result {
            assert_eq!(v.scalar_at(0).value(), Some(11));
            assert_eq!(v.scalar_at(1).value(), Some(22));
            assert_eq!(v.scalar_at(2).value(), Some(33));
        } else {
            panic!("Expected I32 result");
        }
    }

    #[test]
    fn test_float_add() {
        let left: PrimitiveVector = PVectorMut::from_iter([1.0f64, 2.0, 3.0].map(Some))
            .freeze()
            .into();
        let right: PrimitiveVector = PVectorMut::from_iter([0.5f64, 0.5, 0.5].map(Some))
            .freeze()
            .into();

        let result = Arithmetic::<Add, _>::eval(left, &right);
        if let PrimitiveVector::F64(v) = result {
            assert_eq!(v.scalar_at(0).value(), Some(1.5));
            assert_eq!(v.scalar_at(1).value(), Some(2.5));
            assert_eq!(v.scalar_at(2).value(), Some(3.5));
        } else {
            panic!("Expected F64 result");
        }
    }
}
