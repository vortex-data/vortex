// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_vector::PrimitiveDatum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::primitive::PrimitiveScalar;
use vortex_vector::primitive::PrimitiveVector;

use crate::arithmetic::Arithmetic;
use crate::arithmetic::CheckedArithmetic;

impl<Op> CheckedArithmetic<Op> for PrimitiveDatum
where
    for<'a> &'a PrimitiveScalar: CheckedArithmetic<Op, Output = PrimitiveScalar>,
    for<'a> PrimitiveVector: CheckedArithmetic<Op, &'a PrimitiveVector, Output = PrimitiveVector>,
{
    type Output = PrimitiveDatum;

    fn checked_eval(self, rhs: PrimitiveDatum) -> Option<Self::Output> {
        match (self, rhs) {
            (PrimitiveDatum::Scalar(sc1), PrimitiveDatum::Scalar(sc2)) => {
                (&sc1).checked_eval(&sc2).map(PrimitiveDatum::Scalar)
            }
            (PrimitiveDatum::Vector(vec1), PrimitiveDatum::Vector(vec2)) => {
                vec1.checked_eval(&vec2).map(PrimitiveDatum::Vector)
            }
            (PrimitiveDatum::Vector(vec1), PrimitiveDatum::Scalar(sc2)) => {
                let len = vec1.len();
                vec1.checked_eval(&sc2.repeat(len).freeze().into_primitive())
                    .map(PrimitiveDatum::Vector)
            }
            (PrimitiveDatum::Scalar(sc1), PrimitiveDatum::Vector(vec2)) => {
                let len = vec2.len();
                sc1.repeat(len)
                    .freeze()
                    .into_primitive()
                    .checked_eval(&vec2)
                    .map(PrimitiveDatum::Vector)
            }
        }
    }
}

impl<Op> Arithmetic<Op> for PrimitiveDatum
where
    for<'a> &'a PrimitiveScalar: Arithmetic<Op, &'a PrimitiveScalar, Output = PrimitiveScalar>,
    for<'a> &'a PrimitiveScalar: Arithmetic<Op, PrimitiveVector, Output = PrimitiveDatum>,
    for<'a> PrimitiveVector: Arithmetic<Op, &'a PrimitiveVector, Output = PrimitiveVector>,
    for<'a> PrimitiveVector: Arithmetic<Op, &'a PrimitiveScalar, Output = PrimitiveDatum>,
{
    type Output = PrimitiveDatum;

    fn eval(self, rhs: PrimitiveDatum) -> Self::Output {
        match (self, rhs) {
            (PrimitiveDatum::Scalar(sc1), PrimitiveDatum::Scalar(sc2)) => {
                PrimitiveDatum::Scalar((&sc1).eval(&sc2))
            }
            (PrimitiveDatum::Vector(vec1), PrimitiveDatum::Vector(vec2)) => {
                PrimitiveDatum::Vector(vec1.eval(&vec2))
            }
            (PrimitiveDatum::Vector(vec1), PrimitiveDatum::Scalar(sc2)) => vec1.eval(&sc2),
            (PrimitiveDatum::Scalar(sc1), PrimitiveDatum::Vector(vec2)) => (&sc1).eval(vec2),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::PTypeDowncast;
    use vortex_mask::Mask;
    use vortex_vector::Datum;
    use vortex_vector::PrimitiveDatum;
    use vortex_vector::Vector;
    use vortex_vector::primitive::PVector;

    use crate::arithmetic::Add;
    use crate::arithmetic::Arithmetic;

    #[test]
    fn test_datum_arithmetic_in_place() {
        let left = PVector::new(buffer![1f32, 2.0, 3.0, 4.0], Mask::new_true(4));
        let right = PVector::new(buffer![10f32, 20.0, 30.0, 40.0], Mask::new_true(4));
        let left_ptr = left.elements().as_ptr();

        let left_datum = Datum::Vector(Vector::from(left));
        let right_datum = Datum::Vector(Vector::from(right));

        let result =
            Arithmetic::<Add, _>::eval(left_datum.into_primitive(), right_datum.into_primitive());

        let result_vec = match result {
            PrimitiveDatum::Vector(v) => v,
            _ => panic!("Expected primitive vector result"),
        };

        let result_pvec: &PVector<f32> = PTypeDowncast::into_f32(&result_vec);
        let result_ptr = result_pvec.elements().as_ptr();

        assert_eq!(
            left_ptr, result_ptr,
            "Buffer should be modified in place when input has unique ownership"
        );
        assert_eq!(result_pvec.elements(), &buffer![11f32, 22.0, 33.0, 44.0]);
    }

    #[test]
    #[should_panic(expected = "Buffer should be modified in place")]
    fn test_datum_arithmetic_in_place_fail() {
        let left = PVector::new(buffer![1f32, 2.0, 3.0, 4.0], Mask::new_true(4));
        let right = PVector::new(buffer![10f32, 20.0, 30.0, 40.0], Mask::new_true(4));
        let left_ptr = left.elements().as_ptr();

        let left_datum = Datum::Vector(Vector::from(left));
        let _left_datum2 = left_datum.clone();
        let right_datum = Datum::Vector(Vector::from(right));

        let result =
            Arithmetic::<Add, _>::eval(left_datum.into_primitive(), right_datum.into_primitive());

        let result_vec = match result {
            PrimitiveDatum::Vector(v) => v,
            _ => panic!("Expected primitive vector result"),
        };

        let result_pvec: &PVector<f32> = PTypeDowncast::into_f32(&result_vec);
        let result_ptr = result_pvec.elements().as_ptr();

        assert_eq!(
            left_ptr, result_ptr,
            "Buffer should be modified in place when input has unique ownership"
        );
    }

    #[test]
    fn test_datum_vector_scalar_in_place() {
        let left = PVector::new(buffer![1f32, 2.0, 3.0, 4.0], Mask::new_true(4));
        let left_ptr = left.elements().as_ptr();

        let left_datum = PrimitiveDatum::Vector(left.into());
        let right_datum =
            PrimitiveDatum::Scalar(vortex_vector::primitive::PScalar::new(Some(10f32)).into());

        let result = Arithmetic::<Add, _>::eval(left_datum, right_datum);

        let result_vec = match result {
            PrimitiveDatum::Vector(v) => v,
            _ => panic!("Expected primitive vector result"),
        };

        let result_pvec: &PVector<f32> = PTypeDowncast::into_f32(&result_vec);
        let result_ptr = result_pvec.elements().as_ptr();

        assert_eq!(
            left_ptr, result_ptr,
            "Buffer should be modified in place for vector-scalar arithmetic"
        );
        assert_eq!(result_pvec.elements(), &buffer![11f32, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_datum_scalar_vector_in_place() {
        let right = PVector::new(buffer![1f32, 2.0, 3.0, 4.0], Mask::new_true(4));
        let right_ptr = right.elements().as_ptr();

        let left_datum =
            PrimitiveDatum::Scalar(vortex_vector::primitive::PScalar::new(Some(10f32)).into());
        let right_datum = PrimitiveDatum::Vector(right.into());

        let result = Arithmetic::<Add, _>::eval(left_datum, right_datum);

        let result_vec = match result {
            PrimitiveDatum::Vector(v) => v,
            _ => panic!("Expected primitive vector result"),
        };

        let result_pvec = result_vec.into_f32();
        let result_ptr = result_pvec.elements().as_ptr();

        assert_eq!(
            right_ptr, result_ptr,
            "Buffer should be modified in place for scalar-vector arithmetic"
        );
        assert_eq!(result_pvec.elements(), &buffer![11f32, 12.0, 13.0, 14.0]);
    }
}
