// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_vector::BinaryViewDatum;
use vortex_vector::BoolDatum;
use vortex_vector::Datum;
use vortex_vector::DecimalDatum;
use vortex_vector::PrimitiveDatum;
use vortex_vector::ScalarOps;
use vortex_vector::TypedDatum;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
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
            (BoolDatum::Vector(vec), BoolDatum::Scalar(sc)) => {
                let repeated: BoolVector = sc.repeat(vec.len()).into_bool().freeze();
                BoolDatum::Vector(vec.compare(repeated))
            }
            (BoolDatum::Scalar(sc), BoolDatum::Vector(vec)) => {
                let repeated: BoolVector = sc.repeat(vec.len()).into_bool().freeze();
                BoolDatum::Vector(repeated.compare(vec))
            }
            (BoolDatum::Vector(vec1), BoolDatum::Vector(vec2)) => {
                BoolDatum::Vector(vec1.compare(vec2))
            }
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
            (PrimitiveDatum::Vector(vec), PrimitiveDatum::Scalar(sc)) => {
                let repeated: PrimitiveVector = sc.repeat(vec.len()).into_primitive().freeze();
                BoolDatum::Vector(vec.compare(repeated))
            }
            (PrimitiveDatum::Scalar(sc), PrimitiveDatum::Vector(vec)) => {
                let repeated: PrimitiveVector = sc.repeat(vec.len()).into_primitive().freeze();
                BoolDatum::Vector(repeated.compare(vec))
            }
            (PrimitiveDatum::Vector(vec1), PrimitiveDatum::Vector(vec2)) => {
                BoolDatum::Vector(vec1.compare(vec2))
            }
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
            (DecimalDatum::Vector(vec), DecimalDatum::Scalar(sc)) => {
                let repeated = sc.repeat(vec.len()).into_decimal().freeze();
                BoolDatum::Vector(vec.compare(repeated))
            }
            (DecimalDatum::Scalar(sc), DecimalDatum::Vector(vec)) => {
                let repeated = sc.repeat(vec.len()).into_decimal().freeze();
                BoolDatum::Vector(repeated.compare(vec))
            }
            (DecimalDatum::Vector(vec1), DecimalDatum::Vector(vec2)) => {
                BoolDatum::Vector(vec1.compare(vec2))
            }
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
            (BinaryViewDatum::Vector(vec), BinaryViewDatum::Scalar(sc)) => {
                let repeated = T::downcast(sc.repeat(vec.len())).freeze();
                BoolDatum::Vector(vec.compare(repeated))
            }
            (BinaryViewDatum::Scalar(sc), BinaryViewDatum::Vector(vec)) => {
                let repeated = T::downcast(sc.repeat(vec.len())).freeze();
                BoolDatum::Vector(repeated.compare(vec))
            }
            (BinaryViewDatum::Vector(vec1), BinaryViewDatum::Vector(vec2)) => {
                BoolDatum::Vector(vec1.compare(vec2))
            }
        }
    }
}
