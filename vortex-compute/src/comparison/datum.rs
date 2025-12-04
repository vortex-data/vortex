// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_vector::BoolDatum;
use vortex_vector::Datum;
use vortex_vector::PrimitiveDatum;
use vortex_vector::TypedDatum;
use vortex_vector::bool::BoolScalar;
use vortex_vector::bool::BoolVector;
use vortex_vector::primitive::PrimitiveScalar;
use vortex_vector::primitive::PrimitiveVector;

use crate::comparison::Compare;

impl<Op> Compare<Op> for Datum
where
    BoolDatum: Compare<Op, Output = BoolDatum>,
    PrimitiveDatum: Compare<Op, Output = BoolDatum>,
{
    type Output = BoolDatum;

    fn compare(self, rhs: Self) -> Self::Output {
        match (self.into_typed(), rhs.into_typed()) {
            (TypedDatum::Bool(d1), TypedDatum::Bool(d2)) => d1.compare(d2),
            (TypedDatum::Primitive(d1), TypedDatum::Primitive(d2)) => d1.compare(d2),
            _ => todo!(""),
        }
    }
}

impl<Op> Compare<Op> for BoolDatum
where
    BoolVector: Compare<Op, Output = BoolVector>,
    BoolScalar: Compare<Op, Output = BoolScalar>,
{
    type Output = BoolDatum;

    fn compare(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (BoolDatum::Scalar(sc1), BoolDatum::Scalar(sc2)) => BoolDatum::Scalar(sc1.compare(sc2)),
            (BoolDatum::Vector(sc1), BoolDatum::Vector(sc2)) => BoolDatum::Vector(sc1.compare(sc2)),
            _ => unreachable!(""),
        }
    }
}

impl<Op> Compare<Op> for PrimitiveDatum
where
    PrimitiveScalar: Compare<Op, Output = BoolScalar>,
    PrimitiveVector: Compare<Op, Output = BoolVector>,
{
    type Output = BoolDatum;

    fn compare(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (PrimitiveDatum::Scalar(sc1), PrimitiveDatum::Scalar(sc2)) => {
                BoolDatum::Scalar(sc1.compare(sc2))
            }
            (PrimitiveDatum::Vector(sc1), PrimitiveDatum::Vector(sc2)) => {
                BoolDatum::Vector(sc1.compare(sc2))
            }
            _ => unreachable!(""),
        }
    }
}
