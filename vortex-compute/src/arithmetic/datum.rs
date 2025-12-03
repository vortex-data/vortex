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

impl<Op> CheckedArithmetic<Op, &PrimitiveDatum> for &PrimitiveDatum
where
    for<'a> &'a PrimitiveScalar:
        CheckedArithmetic<Op, &'a PrimitiveScalar, Output = PrimitiveScalar>,
    for<'a> &'a PrimitiveVector:
        CheckedArithmetic<Op, &'a PrimitiveVector, Output = PrimitiveVector>,
{
    type Output = PrimitiveDatum;

    fn checked_eval(self, rhs: &PrimitiveDatum) -> Option<Self::Output> {
        match (self, rhs) {
            (PrimitiveDatum::Scalar(sc1), PrimitiveDatum::Scalar(sc2)) => {
                sc1.checked_eval(sc2).map(PrimitiveDatum::Scalar)
            }
            (PrimitiveDatum::Vector(vec1), PrimitiveDatum::Vector(vec2)) => {
                vec1.checked_eval(vec2).map(PrimitiveDatum::Vector)
            }
            (PrimitiveDatum::Vector(vec1), PrimitiveDatum::Scalar(sc2)) => vec1
                .checked_eval(&sc2.repeat(vec1.len()).freeze().into_primitive())
                .map(PrimitiveDatum::Vector),
            (PrimitiveDatum::Scalar(sc1), PrimitiveDatum::Vector(vec2)) => sc1
                .repeat(vec2.len())
                .freeze()
                .into_primitive()
                .checked_eval(vec2)
                .map(PrimitiveDatum::Vector),
        }
    }
}

impl<Op> Arithmetic<Op, &PrimitiveDatum> for &PrimitiveDatum
where
    for<'a> &'a PrimitiveScalar: Arithmetic<Op, &'a PrimitiveScalar, Output = PrimitiveScalar>,
    for<'a> &'a PrimitiveVector: Arithmetic<Op, &'a PrimitiveVector, Output = PrimitiveVector>,
{
    type Output = PrimitiveDatum;

    fn eval(self, rhs: &PrimitiveDatum) -> Self::Output {
        match (self, rhs) {
            (PrimitiveDatum::Scalar(sc1), PrimitiveDatum::Scalar(sc2)) => {
                PrimitiveDatum::Scalar(sc1.eval(sc2))
            }
            (PrimitiveDatum::Vector(vec1), PrimitiveDatum::Vector(vec2)) => {
                PrimitiveDatum::Vector(vec1.eval(vec2))
            }
            _ => unreachable!(""),
        }
    }
}
