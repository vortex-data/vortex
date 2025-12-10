// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Simple binary logical operations: AND, OR, AND NOT.
//!
//! These operations apply a bitwise operation to the bits and AND the validity masks together
//! (null propagates). For Kleene three-valued logic, see the [`kleene`](super::kleene) module.

use std::ops::BitAnd;
use std::ops::BitOr;

use vortex_buffer::BitBuffer;
use vortex_vector::BoolDatum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolScalar;
use vortex_vector::bool::BoolVector;

use super::LogicalAnd;
use super::LogicalAndNot;
use super::LogicalOp;
use super::LogicalOr;

/// Marker type for the AND operation.
pub struct And;

/// Marker type for the OR operation.
pub struct Or;

/// Marker type for the AND NOT operation.
pub struct AndNot;

/// Trait for simple logical binary operations.
///
/// These operations apply a bitwise operation to the bits and AND the validity masks together.
pub trait LogicalBinaryOp {
    /// Apply the operation to two [`BitBuffer`]s.
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer;

    /// Apply the operation to two scalar boolean values.
    fn scalar_op(lhs: bool, rhs: bool) -> bool;
}

impl LogicalBinaryOp for And {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitand(rhs)
    }

    fn scalar_op(lhs: bool, rhs: bool) -> bool {
        lhs && rhs
    }
}

impl LogicalBinaryOp for Or {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitor(rhs)
    }

    fn scalar_op(lhs: bool, rhs: bool) -> bool {
        lhs || rhs
    }
}

impl LogicalBinaryOp for AndNot {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitand_not(rhs)
    }

    fn scalar_op(lhs: bool, rhs: bool) -> bool {
        lhs && !rhs
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Generic `LogicalOp` implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

impl LogicalOp<And> for &BoolScalar {
    type Output = BoolScalar;

    fn op(self, rhs: &BoolScalar) -> BoolScalar {
        binary_scalar_op::<And>(self, rhs)
    }
}

impl LogicalOp<Or> for &BoolScalar {
    type Output = BoolScalar;

    fn op(self, rhs: &BoolScalar) -> BoolScalar {
        binary_scalar_op::<Or>(self, rhs)
    }
}

impl LogicalOp<AndNot> for &BoolScalar {
    type Output = BoolScalar;

    fn op(self, rhs: &BoolScalar) -> BoolScalar {
        binary_scalar_op::<AndNot>(self, rhs)
    }
}

impl LogicalOp<And> for &BoolVector {
    type Output = BoolVector;

    fn op(self, rhs: &BoolVector) -> BoolVector {
        binary_vector_op::<And>(self, rhs)
    }
}

impl LogicalOp<Or> for &BoolVector {
    type Output = BoolVector;

    fn op(self, rhs: &BoolVector) -> BoolVector {
        binary_vector_op::<Or>(self, rhs)
    }
}

impl LogicalOp<AndNot> for &BoolVector {
    type Output = BoolVector;

    fn op(self, rhs: &BoolVector) -> BoolVector {
        binary_vector_op::<AndNot>(self, rhs)
    }
}

impl LogicalOp<And, &BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn op(self, rhs: &BoolDatum) -> BoolDatum {
        binary_datum_op::<And>(self, rhs)
    }
}

impl LogicalOp<Or, &BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn op(self, rhs: &BoolDatum) -> BoolDatum {
        binary_datum_op::<Or>(self, rhs)
    }
}

impl LogicalOp<AndNot, &BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn op(self, rhs: &BoolDatum) -> BoolDatum {
        binary_datum_op::<AndNot>(self, rhs)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Logical helper functions
////////////////////////////////////////////////////////////////////////////////////////////////////

fn binary_scalar_op<Op: LogicalBinaryOp>(lhs: &BoolScalar, rhs: &BoolScalar) -> BoolScalar {
    let result = match (lhs.value(), rhs.value()) {
        (Some(a), Some(b)) => Some(Op::scalar_op(a, b)),
        _ => None, // Null propagation.
    };
    BoolScalar::new(result)
}

fn binary_vector_op<Op: LogicalBinaryOp>(lhs: &BoolVector, rhs: &BoolVector) -> BoolVector {
    assert_eq!(lhs.len(), rhs.len());

    BoolVector::new(
        Op::bit_op(lhs.bits(), rhs.bits()),
        lhs.validity().bitand(rhs.validity()),
    )
}

fn binary_datum_op<Op: LogicalBinaryOp>(lhs: &BoolDatum, rhs: &BoolDatum) -> BoolDatum
where
    for<'a> &'a BoolScalar: LogicalOp<Op, Output = BoolScalar>,
    for<'a> &'a BoolVector: LogicalOp<Op, Output = BoolVector>,
{
    match (lhs, rhs) {
        (BoolDatum::Vector(lhs), BoolDatum::Vector(rhs)) => {
            BoolDatum::Vector(<&BoolVector as LogicalOp<Op>>::op(lhs, rhs))
        }
        (BoolDatum::Scalar(lhs), BoolDatum::Scalar(rhs)) => {
            BoolDatum::Scalar(<&BoolScalar as LogicalOp<Op>>::op(lhs, rhs))
        }
        (BoolDatum::Scalar(sc), BoolDatum::Vector(vec)) => {
            let expanded = sc.repeat(vec.len()).freeze().into_bool();
            BoolDatum::Vector(<&BoolVector as LogicalOp<Op>>::op(&expanded, vec))
        }
        (BoolDatum::Vector(vec), BoolDatum::Scalar(sc)) => {
            let expanded = sc.repeat(vec.len()).freeze().into_bool();
            BoolDatum::Vector(<&BoolVector as LogicalOp<Op>>::op(vec, &expanded))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Convenience trait implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

impl LogicalAnd for &BoolScalar {
    type Output = BoolScalar;

    fn and(self, other: &BoolScalar) -> BoolScalar {
        binary_scalar_op::<And>(self, other)
    }
}

impl LogicalAnd for &BoolVector {
    type Output = BoolVector;

    fn and(self, other: &BoolVector) -> BoolVector {
        binary_vector_op::<And>(self, other)
    }
}

impl LogicalAnd<&BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn and(self, other: &BoolDatum) -> BoolDatum {
        <&BoolDatum as LogicalOp<And, &BoolDatum>>::op(self, other)
    }
}

impl LogicalOr for &BoolScalar {
    type Output = BoolScalar;

    fn or(self, other: &BoolScalar) -> BoolScalar {
        binary_scalar_op::<Or>(self, other)
    }
}

impl LogicalOr for &BoolVector {
    type Output = BoolVector;

    fn or(self, other: &BoolVector) -> BoolVector {
        binary_vector_op::<Or>(self, other)
    }
}

impl LogicalOr<&BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn or(self, other: &BoolDatum) -> BoolDatum {
        <&BoolDatum as LogicalOp<Or, &BoolDatum>>::op(self, other)
    }
}

impl LogicalAndNot for &BoolScalar {
    type Output = BoolScalar;

    fn and_not(self, other: &BoolScalar) -> BoolScalar {
        binary_scalar_op::<AndNot>(self, other)
    }
}

impl LogicalAndNot for &BoolVector {
    type Output = BoolVector;

    fn and_not(self, other: &BoolVector) -> BoolVector {
        binary_vector_op::<AndNot>(self, other)
    }
}

impl LogicalAndNot<&BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn and_not(self, other: &BoolDatum) -> BoolDatum {
        <&BoolDatum as LogicalOp<AndNot, &BoolDatum>>::op(self, other)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;
    use vortex_mask::Mask;
    use vortex_vector::bool::BoolScalar;
    use vortex_vector::bool::BoolVector;

    use super::*;

    // AND tests.

    #[test]
    fn test_and_basic() {
        let left = BoolVector::new(bitbuffer![1 1 0 0], Mask::new_true(4));
        let right = BoolVector::new(bitbuffer![1 0 1 0], Mask::new_true(4));

        let result = left.and(&right);
        assert_eq!(result.bits(), &bitbuffer![1 0 0 0]);
    }

    #[test]
    fn test_and_with_nulls() {
        let left = BoolVector::new(bitbuffer![1 0], Mask::from(bitbuffer![1 0]));
        let right = BoolVector::new(bitbuffer![1 1], Mask::new_true(2));

        let result = left.and(&right);
        // Validity is AND'd, so if either side is null, result is null.
        assert_eq!(result.validity(), &Mask::from(bitbuffer![1 0]));
    }

    #[test]
    fn test_and_scalar() {
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(false));
        assert_eq!((&left).and(&right).value(), Some(false));

        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(true));
        assert_eq!((&left).and(&right).value(), Some(true));

        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(None);
        assert_eq!((&left).and(&right).value(), None);
    }

    // OR tests.

    #[test]
    fn test_or_basic() {
        let left = BoolVector::new(bitbuffer![1 1 0 0], Mask::new_true(4));
        let right = BoolVector::new(bitbuffer![1 0 1 0], Mask::new_true(4));

        let result = left.or(&right);
        assert_eq!(result.bits(), &bitbuffer![1 1 1 0]);
    }

    #[test]
    fn test_or_with_nulls() {
        let left = BoolVector::new(bitbuffer![0 1], Mask::from(bitbuffer![0 1]));
        let right = BoolVector::new(bitbuffer![0 0], Mask::new_true(2));

        let result = left.or(&right);
        // Validity is AND'd, so if either side is null, result is null.
        assert_eq!(result.validity(), &Mask::from(bitbuffer![0 1]));
    }

    #[test]
    fn test_or_scalar() {
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(false));
        assert_eq!((&left).or(&right).value(), Some(true));

        let left = BoolScalar::new(Some(false));
        let right = BoolScalar::new(Some(false));
        assert_eq!((&left).or(&right).value(), Some(false));

        let left = BoolScalar::new(Some(false));
        let right = BoolScalar::new(None);
        assert_eq!((&left).or(&right).value(), None);
    }

    // AND NOT tests.

    #[test]
    fn test_and_not_basic() {
        // left AND (NOT right).
        let left = BoolVector::new(bitbuffer![1 1 0 0], Mask::new_true(4));
        let right = BoolVector::new(bitbuffer![1 0 1 0], Mask::new_true(4));

        let result = left.and_not(&right);
        // 1 & !1 = 0, 1 & !0 = 1, 0 & !1 = 0, 0 & !0 = 0.
        assert_eq!(result.bits(), &bitbuffer![0 1 0 0]);
    }

    #[test]
    fn test_and_not_all_true() {
        let left = BoolVector::new(bitbuffer![1 1], Mask::new_true(2));
        let right = BoolVector::new(bitbuffer![1 1], Mask::new_true(2));

        let result = left.and_not(&right);
        assert_eq!(result.bits(), &bitbuffer![0 0]);
    }

    #[test]
    fn test_and_not_scalar() {
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(true));
        assert_eq!((&left).and_not(&right).value(), Some(false));

        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(false));
        assert_eq!((&left).and_not(&right).value(), Some(true));

        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(None);
        assert_eq!((&left).and_not(&right).value(), None);
    }

    // Datum tests.

    #[test]
    fn test_datum_and_vector_vector() {
        let left = BoolDatum::Vector(BoolVector::new(bitbuffer![1 1 0 0], Mask::new_true(4)));
        let right = BoolDatum::Vector(BoolVector::new(bitbuffer![1 0 1 0], Mask::new_true(4)));

        let result = left.and(&right);
        let BoolDatum::Vector(vec) = result else {
            panic!("Expected Vector");
        };
        assert_eq!(vec.bits(), &bitbuffer![1 0 0 0]);
    }

    #[test]
    fn test_datum_and_scalar_scalar() {
        let left = BoolDatum::Scalar(BoolScalar::new(Some(true)));
        let right = BoolDatum::Scalar(BoolScalar::new(Some(false)));

        let result = left.and(&right);
        let BoolDatum::Scalar(sc) = result else {
            panic!("Expected Scalar");
        };
        assert_eq!(sc.value(), Some(false));
    }

    #[test]
    fn test_datum_and_scalar_vector() {
        let left = BoolDatum::Scalar(BoolScalar::new(Some(true)));
        let right = BoolDatum::Vector(BoolVector::new(bitbuffer![1 0 1 0], Mask::new_true(4)));

        let result = left.and(&right);
        let BoolDatum::Vector(vec) = result else {
            panic!("Expected Vector");
        };
        assert_eq!(vec.bits(), &bitbuffer![1 0 1 0]);
    }

    #[test]
    fn test_datum_or_vector_scalar() {
        let left = BoolDatum::Vector(BoolVector::new(bitbuffer![1 0 1 0], Mask::new_true(4)));
        let right = BoolDatum::Scalar(BoolScalar::new(Some(true)));

        let result = left.or(&right);
        let BoolDatum::Vector(vec) = result else {
            panic!("Expected Vector");
        };
        assert_eq!(vec.bits(), &bitbuffer![1 1 1 1]);
    }
}
