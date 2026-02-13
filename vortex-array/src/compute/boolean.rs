// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::arrow::FromArrowArray;
use crate::arrow::IntoArrowArray;
use crate::compute::Options;
use crate::expr::Binary;
use crate::expr::ScalarFn;
use crate::expr::operators::Operator;

/// Point-wise logical _and_ between two Boolean arrays.
///
/// This method uses Arrow-style null propagation rather than the Kleene logic semantics. This
/// semantics is also known as "Bochvar logic" and "weak Kleene logic".
///
/// See also [BooleanOperator::And]
#[deprecated(note = "Use and_kleene instead. Non-Kleene boolean ops cannot be lazily evaluated.")]
pub fn and(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    boolean(lhs, rhs, BooleanOperator::And)
}

/// Point-wise Kleene logical _and_ between two Boolean arrays.
///
/// See also [BooleanOperator::AndKleene]
pub fn and_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    boolean(lhs, rhs, BooleanOperator::AndKleene)
}

/// Point-wise logical _or_ between two Boolean arrays.
///
/// This method uses Arrow-style null propagation rather than the Kleene logic semantics. This
/// semantics is also known as "Bochvar logic" and "weak Kleene logic".
///
/// See also [BooleanOperator::Or]
#[deprecated(note = "Use or_kleene instead. Non-Kleene boolean ops cannot be lazily evaluated.")]
pub fn or(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    boolean(lhs, rhs, BooleanOperator::Or)
}

/// Point-wise Kleene logical _or_ between two Boolean arrays.
///
/// See also [BooleanOperator::OrKleene]
pub fn or_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    boolean(lhs, rhs, BooleanOperator::OrKleene)
}

/// Point-wise logical operator between two Boolean arrays.
pub fn boolean(lhs: &dyn Array, rhs: &dyn Array, op: BooleanOperator) -> VortexResult<ArrayRef> {
    match Operator::try_from(op) {
        Ok(expr_op) => Ok(ScalarFnArray::try_new(
            ScalarFn::new(Binary, expr_op),
            vec![lhs.to_array(), rhs.to_array()],
            lhs.len(),
        )?
        .into_array()),
        Err(_) => {
            tracing::trace!(
                "non-Kleene boolean op {op:?} cannot be lazily evaluated, falling back to eager Arrow evaluation"
            );
            arrow_boolean(lhs.to_array(), rhs.to_array(), op)
        }
    }
}

/// Operations over the nullable Boolean values.
///
/// All three operators accept and produce values from the set {true, false, and null}.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOperator {
    /// Logical and, unless either value is null, in which case the result is null.
    ///
    /// | A ∧ B |       | **B** |       |       |
    /// |:-----:|:-----:|:-----:|:-----:|:-----:|
    /// |       |       | **F** | **U** | **T** |
    /// | **A** | **F** | F     | U     | F     |
    /// |       | **U** | U     | U     | U     |
    /// |       | **T** | F     | U     | T     |
    And,
    /// [Kleene (three-valued) logical and](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics).
    ///
    /// | A ∧ B |       | **B** |       |       |
    /// |:-----:|:-----:|:-----:|:-----:|:-----:|
    /// |       |       | **F** | **U** | **T** |
    /// | **A** | **F** | F     | F     | F     |
    /// |       | **U** | F     | U     | U     |
    /// |       | **T** | F     | U     | T     |
    AndKleene,
    /// Logical or, unless either value is null, in which case the result is null.
    ///
    /// | A ∨ B |       | **B** |       |       |
    /// |:-----:|:-----:|:-----:|:-----:|:-----:|
    /// |       |       | **F** | **U** | **T** |
    /// | **A** | **F** | F     | U     | T     |
    /// |       | **U** | U     | U     | U     |
    /// |       | **T** | T     | U     | T     |
    Or,
    /// [Kleene (three-valued) logical or](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics).
    ///
    /// | A ∨ B |       | **B** |       |       |
    /// |:-----:|:-----:|:-----:|:-----:|:-----:|
    /// |       |       | **F** | **U** | **T** |
    /// | **A** | **F** | F     | U     | T     |
    /// |       | **U** | U     | U     | T     |
    /// |       | **T** | T     | T     | T     |
    OrKleene,
    // AndNot,
    // AndNotKleene,
    // Xor,
}

impl Options for BooleanOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Implementation of `BinaryBooleanFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
pub(crate) fn arrow_boolean(
    lhs: ArrayRef,
    rhs: ArrayRef,
    operator: BooleanOperator,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    let lhs = lhs.into_arrow(&DataType::Boolean)?.as_boolean().clone();
    let rhs = rhs.into_arrow(&DataType::Boolean)?.as_boolean().clone();

    let array = match operator {
        BooleanOperator::And => arrow_arith::boolean::and(&lhs, &rhs)?,
        BooleanOperator::AndKleene => arrow_arith::boolean::and_kleene(&lhs, &rhs)?,
        BooleanOperator::Or => arrow_arith::boolean::or(&lhs, &rhs)?,
        BooleanOperator::OrKleene => arrow_arith::boolean::or_kleene(&lhs, &rhs)?,
    };

    ArrayRef::from_arrow(&array, nullable)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::canonical::ToCanonical;
    #[rstest]
    #[case(BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter())
    .into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter())
    .into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter()).into_array())]
    fn test_or(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = or_kleene(&lhs, &rhs).unwrap();

        let r = r.to_bool().into_array();

        let v0 = r.scalar_at(0).unwrap().as_bool().value();
        let v1 = r.scalar_at(1).unwrap().as_bool().value();
        let v2 = r.scalar_at(2).unwrap().as_bool().value();
        let v3 = r.scalar_at(3).unwrap().as_bool().value();

        assert!(v0.unwrap());
        assert!(v1.unwrap());
        assert!(v2.unwrap());
        assert!(!v3.unwrap());
    }

    #[rstest]
    #[case(BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter())
    .into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter())
    .into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter()).into_array())]
    fn test_and(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = and_kleene(&lhs, &rhs).unwrap().to_bool().into_array();

        let v0 = r.scalar_at(0).unwrap().as_bool().value();
        let v1 = r.scalar_at(1).unwrap().as_bool().value();
        let v2 = r.scalar_at(2).unwrap().as_bool().value();
        let v3 = r.scalar_at(3).unwrap().as_bool().value();

        assert!(v0.unwrap());
        assert!(!v1.unwrap());
        assert!(!v2.unwrap());
        assert!(!v3.unwrap());
    }
}
