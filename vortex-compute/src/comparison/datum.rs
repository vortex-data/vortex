// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_vector::BinaryViewDatum;
use vortex_vector::BoolDatum;
use vortex_vector::Datum;
use vortex_vector::DecimalDatum;
use vortex_vector::PrimitiveDatum;
use vortex_vector::TypedDatum;
use vortex_vector::binaryview::BinaryType;
use vortex_vector::binaryview::BinaryViewScalar;
use vortex_vector::binaryview::BinaryViewType;
use vortex_vector::binaryview::BinaryViewVector;
use vortex_vector::binaryview::StringType;
use vortex_vector::bool::BoolScalar;
use vortex_vector::bool::BoolVector;
use vortex_vector::decimal::DecimalScalar;
use vortex_vector::decimal::DecimalVector;
use vortex_vector::primitive::PrimitiveScalar;
use vortex_vector::primitive::PrimitiveVector;

use crate::comparison::Compare;

impl<Op> Compare<Op> for Datum
where
    BoolDatum: Compare<Op, Output = BoolDatum>,
    PrimitiveDatum: Compare<Op, Output = BoolDatum>,
    DecimalDatum: Compare<Op, Output = BoolDatum>,
    BinaryViewDatum<StringType>: Compare<Op, Output = BoolDatum>,
    BinaryViewDatum<BinaryType>: Compare<Op, Output = BoolDatum>,
{
    type Output = BoolDatum;

    fn compare(self, rhs: Self) -> Self::Output {
        match (self.into_typed(), rhs.into_typed()) {
            (TypedDatum::Bool(d1), TypedDatum::Bool(d2)) => d1.compare(d2),
            (TypedDatum::Primitive(d1), TypedDatum::Primitive(d2)) => d1.compare(d2),
            (TypedDatum::Decimal(d1), TypedDatum::Decimal(d2)) => d1.compare(d2),
            (TypedDatum::String(d1), TypedDatum::String(d2)) => d1.compare(d2),
            (TypedDatum::Binary(d1), TypedDatum::Binary(d2)) => d1.compare(d2),
            _ => unreachable!(""),
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

impl<Op> Compare<Op> for DecimalDatum
where
    DecimalScalar: Compare<Op, Output = BoolScalar>,
    DecimalVector: Compare<Op, Output = BoolVector>,
{
    type Output = BoolDatum;

    fn compare(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (DecimalDatum::Scalar(sc1), DecimalDatum::Scalar(sc2)) => {
                BoolDatum::Scalar(sc1.compare(sc2))
            }
            (DecimalDatum::Vector(sc1), DecimalDatum::Vector(sc2)) => {
                BoolDatum::Vector(sc1.compare(sc2))
            }
            _ => unreachable!(""),
        }
    }
}

impl<Op, T> Compare<Op> for BinaryViewDatum<T>
where
    T: BinaryViewType,
    BinaryViewScalar<T>: Compare<Op, Output = BoolScalar>,
    BinaryViewVector<T>: Compare<Op, Output = BoolVector>,
{
    type Output = BoolDatum;

    fn compare(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (BinaryViewDatum::Scalar(sc1), BinaryViewDatum::Scalar(sc2)) => {
                BoolDatum::Scalar(sc1.compare(sc2))
            }
            (BinaryViewDatum::Vector(sc1), BinaryViewDatum::Vector(sc2)) => {
                BoolDatum::Vector(sc1.compare(sc2))
            }
            _ => unreachable!(""),
        }
    }
}
