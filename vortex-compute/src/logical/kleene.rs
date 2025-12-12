// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Kleene three-valued logical operations: AND KLEENE, OR KLEENE.
//!
//! These operations implement Kleene's three-valued logic (K3), also used by SQL:
//! - `FALSE AND NULL = FALSE` (false absorbs null)
//! - `TRUE OR NULL = TRUE` (true absorbs null)
//!
//! For simple null-propagating operations, see the [`binary`](super::binary) module.

use std::ops::BitAnd;
use std::ops::BitOr;
use std::ops::Not;

use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_vector::BoolDatum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolScalar;
use vortex_vector::bool::BoolVector;

use super::LogicalAndKleene;
use super::LogicalOp;
use super::LogicalOrKleene;

/// Marker type for the Kleene AND operation.
pub struct KleeneAnd;

/// Marker type for the Kleene OR operation.
pub struct KleeneOr;

/// Trait for Kleene three-valued logical binary operations.
///
/// Absorbing values produce a valid result regardless of the other operand:
/// - For AND: `FALSE` absorbs nulls (`FALSE AND NULL = FALSE`)
/// - For OR: `TRUE` absorbs nulls (`TRUE OR NULL = TRUE`)
pub trait KleeneBinaryOp {
    /// Apply the operation to two [`BitBuffer`]s.
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer;

    /// Returns a mask of positions with absorbing values.
    ///
    /// - AND: `FALSE` absorbs, so return `bits.not()` (false positions).
    /// - OR: `TRUE` absorbs, so return `bits.clone()` (true positions).
    fn absorb_bits(bits: &BitBuffer) -> BitBuffer;

    /// Apply the operation to two scalar `Option<bool>` values with Kleene semantics.
    fn scalar_op(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool>;
}

impl KleeneBinaryOp for KleeneAnd {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitand(rhs)
    }

    fn absorb_bits(bits: &BitBuffer) -> BitBuffer {
        bits.not() // `false` absorbs nulls.
    }

    fn scalar_op(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
        match (lhs, rhs) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        }
    }
}

impl KleeneBinaryOp for KleeneOr {
    fn bit_op(lhs: &BitBuffer, rhs: &BitBuffer) -> BitBuffer {
        lhs.bitor(rhs)
    }

    fn absorb_bits(bits: &BitBuffer) -> BitBuffer {
        bits.clone() // `true` absorbs nulls.
    }

    fn scalar_op(lhs: Option<bool>, rhs: Option<bool>) -> Option<bool> {
        match (lhs, rhs) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Generic `LogicalOp` implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

impl LogicalOp<KleeneAnd> for &BoolScalar {
    type Output = BoolScalar;

    fn op(self, rhs: &BoolScalar) -> BoolScalar {
        kleene_scalar_op::<KleeneAnd>(self, rhs)
    }
}

impl LogicalOp<KleeneOr> for &BoolScalar {
    type Output = BoolScalar;

    fn op(self, rhs: &BoolScalar) -> BoolScalar {
        kleene_scalar_op::<KleeneOr>(self, rhs)
    }
}

impl LogicalOp<KleeneAnd> for &BoolVector {
    type Output = BoolVector;

    fn op(self, rhs: &BoolVector) -> BoolVector {
        kleene_vector_op::<KleeneAnd>(self, rhs)
    }
}

impl LogicalOp<KleeneOr> for &BoolVector {
    type Output = BoolVector;

    fn op(self, rhs: &BoolVector) -> BoolVector {
        kleene_vector_op::<KleeneOr>(self, rhs)
    }
}

impl LogicalOp<KleeneAnd, &BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn op(self, rhs: &BoolDatum) -> BoolDatum {
        kleene_datum_op::<KleeneAnd>(self, rhs)
    }
}

impl LogicalOp<KleeneOr, &BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn op(self, rhs: &BoolDatum) -> BoolDatum {
        kleene_datum_op::<KleeneOr>(self, rhs)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kleene helper functions
////////////////////////////////////////////////////////////////////////////////////////////////////

fn kleene_scalar_op<Op: KleeneBinaryOp>(lhs: &BoolScalar, rhs: &BoolScalar) -> BoolScalar {
    BoolScalar::new(Op::scalar_op(lhs.value(), rhs.value()))
}

fn kleene_vector_op<Op: KleeneBinaryOp>(lhs: &BoolVector, rhs: &BoolVector) -> BoolVector {
    assert_eq!(lhs.len(), rhs.len());
    let len = lhs.len();

    match (lhs.validity(), rhs.validity()) {
        // Everything is valid, so we can just do a simple logical AND over all bits.
        (Mask::AllTrue(_), Mask::AllTrue(_)) => {
            let result_bits = Op::bit_op(lhs.bits(), rhs.bits());
            BoolVector::new(result_bits, Mask::new_true(len))
        }

        // Everything is null, so the entire result vector is null.
        (Mask::AllFalse(_), Mask::AllFalse(_)) => {
            // Since everything in null, we can just reuse the LHS bits.
            let result_bits = lhs.bits().clone();
            BoolVector::new(result_bits, Mask::new_false(len))
        }

        // LHS is all valid, RHS is all null.
        (Mask::AllTrue(_), Mask::AllFalse(_)) => {
            // The result vector is valid where the LHS has an absorbing value. Since only
            // absorbing values produce valid results, and absorbing values equal the result of the
            // operation, we can reuse the LHS bits directly.
            let result_bits = lhs.bits().clone();
            let validity = Op::absorb_bits(lhs.bits());
            BoolVector::new(result_bits, Mask::from(validity))
        }

        // LHS is all null, RHS is all valid.
        (Mask::AllFalse(_), Mask::AllTrue(_)) => {
            // The result vector is valid where the RHS has an absorbing value. Since only
            // absorbing values produce valid results, and absorbing values equal the result of the
            // operation, we can reuse the RHS bits directly.
            let result_bits = rhs.bits().clone();
            let validity = Op::absorb_bits(rhs.bits());
            BoolVector::new(result_bits, Mask::from(validity))
        }

        // LHS is all valid, RHS has specific validity.
        (Mask::AllTrue(_), Mask::Values(rhs_values)) => {
            // The result vector is valid where the RHS is valid OR the LHS has an absorbing value.
            let result_bits = Op::bit_op(lhs.bits(), rhs.bits());
            let validity = rhs_values.bit_buffer().bitor(&Op::absorb_bits(lhs.bits()));
            BoolVector::new(result_bits, Mask::from(validity))
        }

        // LHS has specific validity, RHS is all valid.
        (Mask::Values(lhs_values), Mask::AllTrue(_)) => {
            // The result vector is valid where the LHS is valid OR the RHS has an absorbing value.
            let result_bits = Op::bit_op(lhs.bits(), rhs.bits());
            let validity = lhs_values.bit_buffer().bitor(&Op::absorb_bits(rhs.bits()));
            BoolVector::new(result_bits, Mask::from(validity))
        }

        // LHS is all null, RHS has specific validity.
        (Mask::AllFalse(_), Mask::Values(rhs_values)) => {
            // The result vector is valid where the RHS is valid AND has an absorbing value. Since
            // only absorbing values produce valid results, we can reuse the RHS bits directly.
            let result_bits = rhs.bits().clone();
            let validity = rhs_values.bit_buffer().bitand(&Op::absorb_bits(rhs.bits()));
            BoolVector::new(result_bits, Mask::from(validity))
        }

        // LHS has specific validity, RHS is all null.
        (Mask::Values(lhs_values), Mask::AllFalse(_)) => {
            // The result vector is valid where the LHS is valid AND has an absorbing value. Since
            // only absorbing values produce valid results, we can reuse the LHS bits directly.
            let result_bits = lhs.bits().clone();
            let validity = lhs_values.bit_buffer().bitand(&Op::absorb_bits(lhs.bits()));
            BoolVector::new(result_bits, Mask::from(validity))
        }

        // Both sides have specific validity.
        (Mask::Values(lhs_values), Mask::Values(rhs_values)) => {
            // The result is valid at position `i` iff:
            // 1. Both lhs[i] and rhs[i] are valid (standard case), OR
            // 2. lhs[i] is null but rhs[i] is valid AND has an absorbing value, OR
            // 3. rhs[i] is null but lhs[i] is valid AND has an absorbing value.
            //
            // Absorbing values in Kleene logic:
            // - AND: false absorbs null (false AND null = false).
            // - OR: true absorbs null (true OR null = true).
            //
            // This simplifies to the gosition is valid iff:
            //   - (lhs_valid OR rhs_absorbs) AND
            //   - (rhs_valid OR lhs_absorbs).
            //
            // In other words, each side must either be valid or have an absorbing value that
            // "covers" the other side's null.
            let result_bits = Op::bit_op(lhs.bits(), rhs.bits());

            let lhs_valid_or_rhs_absorbs =
                lhs_values.bit_buffer().bitor(&Op::absorb_bits(rhs.bits()));
            let rhs_valid_or_lhs_absorbs =
                rhs_values.bit_buffer().bitor(&Op::absorb_bits(lhs.bits()));
            let validity = lhs_valid_or_rhs_absorbs.bitand(&rhs_valid_or_lhs_absorbs);

            BoolVector::new(result_bits, Mask::from(validity))
        }
    }
}

fn kleene_datum_op<Op: KleeneBinaryOp>(lhs: &BoolDatum, rhs: &BoolDatum) -> BoolDatum
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
        // TODO: Specialize this instead of using `repeat`.
        (BoolDatum::Scalar(sc), BoolDatum::Vector(vec)) => {
            let expanded = sc.repeat(vec.len()).freeze().into_bool();
            BoolDatum::Vector(<&BoolVector as LogicalOp<Op>>::op(&expanded, vec))
        }
        // TODO: Specialize this instead of using `repeat`.
        (BoolDatum::Vector(vec), BoolDatum::Scalar(sc)) => {
            let expanded = sc.repeat(vec.len()).freeze().into_bool();
            BoolDatum::Vector(<&BoolVector as LogicalOp<Op>>::op(vec, &expanded))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Convenience trait implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

impl LogicalAndKleene for &BoolScalar {
    type Output = BoolScalar;

    fn and_kleene(self, rhs: &BoolScalar) -> BoolScalar {
        kleene_scalar_op::<KleeneAnd>(self, rhs)
    }
}

impl LogicalAndKleene for &BoolVector {
    type Output = BoolVector;

    fn and_kleene(self, rhs: &BoolVector) -> BoolVector {
        kleene_vector_op::<KleeneAnd>(self, rhs)
    }
}

impl LogicalAndKleene<&BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn and_kleene(self, rhs: &BoolDatum) -> BoolDatum {
        <&BoolDatum as LogicalOp<KleeneAnd, &BoolDatum>>::op(self, rhs)
    }
}

impl LogicalOrKleene for &BoolScalar {
    type Output = BoolScalar;

    fn or_kleene(self, rhs: &BoolScalar) -> BoolScalar {
        kleene_scalar_op::<KleeneOr>(self, rhs)
    }
}

impl LogicalOrKleene for &BoolVector {
    type Output = BoolVector;

    fn or_kleene(self, rhs: &BoolVector) -> BoolVector {
        kleene_vector_op::<KleeneOr>(self, rhs)
    }
}

impl LogicalOrKleene<&BoolDatum> for &BoolDatum {
    type Output = BoolDatum;

    fn or_kleene(self, rhs: &BoolDatum) -> BoolDatum {
        <&BoolDatum as LogicalOp<KleeneOr, &BoolDatum>>::op(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::bitbuffer;
    use vortex_mask::Mask;
    use vortex_vector::bool::BoolScalar;
    use vortex_vector::bool::BoolVector;

    use super::*;

    // AND KLEENE vector tests.

    #[test]
    fn test_and_kleene_all_valid() {
        // When both sides are all valid, behaves like regular AND.
        let left = BoolVector::new(bitbuffer![1 0 1], Mask::new_true(3));
        let right = BoolVector::new(bitbuffer![1 1 0], Mask::new_true(3));

        let result = left.and_kleene(&right);
        assert_eq!(result.bits(), &bitbuffer![1 0 0]);
        assert_eq!(result.validity(), &Mask::new_true(3));
    }

    #[test]
    fn test_and_kleene_all_null() {
        // When both are null, result is all null.
        let left = BoolVector::new(bitbuffer![1 1], Mask::new_false(2));
        let right = BoolVector::new(bitbuffer![1 1], Mask::new_false(2));

        let result = left.and_kleene(&right);
        assert_eq!(result.validity(), &Mask::new_false(2));
    }

    #[test]
    fn test_and_kleene_false_and_null() {
        // false AND null = false (Kleene logic).
        let left = BoolVector::new(bitbuffer![0], Mask::new_true(1));
        let right = BoolVector::new(bitbuffer![1], Mask::new_false(1));

        let result = left.and_kleene(&right);
        assert_eq!(result.bits(), &bitbuffer![0]);
        // Result should be valid because false AND anything is false.
        assert_eq!(result.validity(), &Mask::new_true(1));
    }

    // AND KLEENE scalar tests.

    #[test]
    fn test_scalar_and_kleene_true_true() {
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(true));
        assert_eq!(left.and_kleene(&right).value(), Some(true));
    }

    #[test]
    fn test_scalar_and_kleene_true_false() {
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(false));
        assert_eq!(left.and_kleene(&right).value(), Some(false));
    }

    #[test]
    fn test_scalar_and_kleene_false_null() {
        // false AND null = false (Kleene logic).
        let left = BoolScalar::new(Some(false));
        let right = BoolScalar::new(None);
        assert_eq!(left.and_kleene(&right).value(), Some(false));
    }

    #[test]
    fn test_scalar_and_kleene_null_false() {
        // null AND false = false (Kleene logic).
        let left = BoolScalar::new(None);
        let right = BoolScalar::new(Some(false));
        assert_eq!(left.and_kleene(&right).value(), Some(false));
    }

    #[test]
    fn test_scalar_and_kleene_true_null() {
        // true AND null = null.
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(None);
        assert_eq!(left.and_kleene(&right).value(), None);
    }

    #[test]
    fn test_scalar_and_kleene_null_null() {
        let left = BoolScalar::new(None);
        let right = BoolScalar::new(None);
        assert_eq!(left.and_kleene(&right).value(), None);
    }

    // OR KLEENE vector tests.

    #[test]
    fn test_or_kleene_all_valid() {
        // When both sides are all valid, behaves like regular OR.
        let left = BoolVector::new(bitbuffer![1 0 0], Mask::new_true(3));
        let right = BoolVector::new(bitbuffer![0 1 0], Mask::new_true(3));

        let result = left.or_kleene(&right);
        assert_eq!(result.bits(), &bitbuffer![1 1 0]);
        assert_eq!(result.validity(), &Mask::new_true(3));
    }

    #[test]
    fn test_or_kleene_all_null() {
        // When both are null, result is all null.
        let left = BoolVector::new(bitbuffer![0 0], Mask::new_false(2));
        let right = BoolVector::new(bitbuffer![0 0], Mask::new_false(2));

        let result = left.or_kleene(&right);
        assert_eq!(result.validity(), &Mask::new_false(2));
    }

    #[test]
    fn test_or_kleene_true_and_null() {
        // true OR null = true (Kleene logic).
        let left = BoolVector::new(bitbuffer![1], Mask::new_true(1));
        let right = BoolVector::new(bitbuffer![0], Mask::new_false(1));

        let result = left.or_kleene(&right);
        assert_eq!(result.bits(), &bitbuffer![1]);
        // Result should be valid because true OR anything is true.
        assert_eq!(result.validity(), &Mask::new_true(1));
    }

    // OR KLEENE scalar tests.

    #[test]
    fn test_scalar_or_kleene_true_true() {
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(true));
        assert_eq!(left.or_kleene(&right).value(), Some(true));
    }

    #[test]
    fn test_scalar_or_kleene_true_false() {
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(Some(false));
        assert_eq!(left.or_kleene(&right).value(), Some(true));
    }

    #[test]
    fn test_scalar_or_kleene_false_false() {
        let left = BoolScalar::new(Some(false));
        let right = BoolScalar::new(Some(false));
        assert_eq!(left.or_kleene(&right).value(), Some(false));
    }

    #[test]
    fn test_scalar_or_kleene_true_null() {
        // true OR null = true (Kleene logic).
        let left = BoolScalar::new(Some(true));
        let right = BoolScalar::new(None);
        assert_eq!(left.or_kleene(&right).value(), Some(true));
    }

    #[test]
    fn test_scalar_or_kleene_null_true() {
        // null OR true = true (Kleene logic).
        let left = BoolScalar::new(None);
        let right = BoolScalar::new(Some(true));
        assert_eq!(left.or_kleene(&right).value(), Some(true));
    }

    #[test]
    fn test_scalar_or_kleene_false_null() {
        // false OR null = null.
        let left = BoolScalar::new(Some(false));
        let right = BoolScalar::new(None);
        assert_eq!(left.or_kleene(&right).value(), None);
    }

    #[test]
    fn test_scalar_or_kleene_null_null() {
        let left = BoolScalar::new(None);
        let right = BoolScalar::new(None);
        assert_eq!(left.or_kleene(&right).value(), None);
    }

    // Datum tests.

    #[test]
    fn test_datum_and_kleene_vector_vector() {
        let left = BoolDatum::Vector(BoolVector::new(bitbuffer![0], Mask::new_true(1)));
        let right = BoolDatum::Vector(BoolVector::new(bitbuffer![1], Mask::new_false(1)));

        let result = left.and_kleene(&right);
        let BoolDatum::Vector(vec) = result else {
            panic!("Expected Vector");
        };
        // false AND null = false.
        assert_eq!(vec.bits(), &bitbuffer![0]);
        assert_eq!(vec.validity(), &Mask::new_true(1));
    }

    #[test]
    fn test_datum_and_kleene_scalar_scalar() {
        let left = BoolDatum::Scalar(BoolScalar::new(Some(false)));
        let right = BoolDatum::Scalar(BoolScalar::new(None));

        let result = left.and_kleene(&right);
        let BoolDatum::Scalar(sc) = result else {
            panic!("Expected Scalar");
        };
        // false AND null = false.
        assert_eq!(sc.value(), Some(false));
    }

    #[test]
    fn test_datum_or_kleene_scalar_vector() {
        let left = BoolDatum::Scalar(BoolScalar::new(Some(true)));
        let right = BoolDatum::Vector(BoolVector::new(bitbuffer![0 0], Mask::new_false(2)));

        let result = left.or_kleene(&right);
        let BoolDatum::Vector(vec) = result else {
            panic!("Expected Vector");
        };
        // true OR null = true.
        assert_eq!(vec.bits(), &bitbuffer![1 1]);
        assert_eq!(vec.validity(), &Mask::new_true(2));
    }
}
