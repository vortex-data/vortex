// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrow::FromArrowArray;
use crate::arrow::IntoArrowArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;

/// Point-wise Kleene logical _and_ between two Boolean arrays.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn and_kleene(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.clone().binary(rhs.clone(), Operator::And)
}

/// Point-wise Kleene logical _or_ between two Boolean arrays.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn or_kleene(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.clone().binary(rhs.clone(), Operator::Or)
}

/// Execute a Kleene boolean operation between two arrays.
///
/// This is the entry point for boolean operations from the binary expression.
/// Handles constant-constant directly, otherwise falls back to Arrow.
pub(crate) fn execute_boolean(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: Operator,
) -> VortexResult<ArrayRef> {
    if let Some(result) = constant_boolean(lhs, rhs, op)? {
        return Ok(result);
    }
    arrow_execute_boolean(lhs.clone(), rhs.clone(), op)
}

/// Arrow implementation for Kleene boolean operations using [`Operator`].
fn arrow_execute_boolean(lhs: ArrayRef, rhs: ArrayRef, op: Operator) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    let lhs = lhs.into_arrow(&DataType::Boolean)?.as_boolean().clone();
    let rhs = rhs.into_arrow(&DataType::Boolean)?.as_boolean().clone();

    let array = match op {
        Operator::And => arrow_arith::boolean::and_kleene(&lhs, &rhs)?,
        Operator::Or => arrow_arith::boolean::or_kleene(&lhs, &rhs)?,
        other => return Err(vortex_err!("Not a boolean operator: {other}")),
    };

    ArrayRef::from_arrow(&array, nullable)
}

/// Constant-folds a boolean operation between two constant arrays.
fn constant_boolean(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: Operator,
) -> VortexResult<Option<ArrayRef>> {
    let (Some(lhs), Some(rhs)) = (lhs.as_opt::<Constant>(), rhs.as_opt::<Constant>()) else {
        return Ok(None);
    };

    let length = lhs.len();
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let lhs_val = lhs.scalar().as_bool().value();
    let rhs_val = rhs
        .scalar()
        .as_bool_opt()
        .ok_or_else(|| vortex_err!("expected rhs to be boolean"))?
        .value();

    let result = match op {
        Operator::And => match (lhs_val, rhs_val) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (None, _) | (_, None) => None,
            (Some(l), Some(r)) => Some(l & r),
        },
        Operator::Or => match (lhs_val, rhs_val) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (None, _) | (_, None) => None,
            (Some(l), Some(r)) => Some(l | r),
        },
        other => return Err(vortex_err!("Not a boolean operator: {other}")),
    };

    let scalar = result
        .map(|b| Scalar::bool(b, nullable.into()))
        .unwrap_or_else(|| Scalar::null(DType::Bool(nullable.into())));

    Ok(Some(ConstantArray::new(scalar, length).into_array()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::builtins::ArrayBuiltins;
    use crate::canonical::ToCanonical;
    use crate::scalar_fn::fns::operators::Operator;

    #[rstest]
    #[case(
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
    )]
    #[case(
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
    )]
    fn test_or(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = lhs.binary(rhs, Operator::Or).unwrap();
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
    #[case(
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
    )]
    #[case(
        BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)]).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)]).into_array(),
    )]
    fn test_and(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = lhs
            .binary(rhs, Operator::And)
            .unwrap()
            .to_bool()
            .into_array();

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
