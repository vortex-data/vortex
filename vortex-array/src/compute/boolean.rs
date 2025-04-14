use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::encoding::Encoding;
use crate::{Array, ArrayRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    And,
    AndKleene,
    Or,
    OrKleene,
    // AndNot,
    // AndNotKleene,
    // Xor,
}

pub trait BinaryBooleanFn<A> {
    fn binary_boolean(
        &self,
        array: A,
        other: &dyn Array,
        op: BinaryOperator,
    ) -> VortexResult<Option<ArrayRef>>;
}

impl<E: Encoding> BinaryBooleanFn<&dyn Array> for E
where
    E: for<'a> BinaryBooleanFn<&'a E::Array>,
{
    fn binary_boolean(
        &self,
        lhs: &dyn Array,
        rhs: &dyn Array,
        op: BinaryOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let array_ref = lhs
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");

        BinaryBooleanFn::binary_boolean(self, array_ref, rhs, op)
    }
}

/// Point-wise logical _and_ between two Boolean arrays.
///
/// This method uses Arrow-style null propagation rather than the Kleene logic semantics.
pub fn and(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    binary_boolean(lhs, rhs, BinaryOperator::And)
}

/// Point-wise Kleene logical _and_ between two Boolean arrays.
pub fn and_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    binary_boolean(lhs, rhs, BinaryOperator::AndKleene)
}

/// Point-wise logical _or_ between two Boolean arrays.
///
/// This method uses Arrow-style null propagation rather than the Kleene logic semantics.
pub fn or(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    binary_boolean(lhs, rhs, BinaryOperator::Or)
}

/// Point-wise Kleene logical _or_ between two Boolean arrays.
pub fn or_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    binary_boolean(lhs, rhs, BinaryOperator::OrKleene)
}

pub fn binary_boolean(
    lhs: &dyn Array,
    rhs: &dyn Array,
    op: BinaryOperator,
) -> VortexResult<ArrayRef> {
    if lhs.len() != rhs.len() {
        vortex_bail!(
            "Boolean operations aren't supported on arrays of different lengths: {} and {}",
            lhs.len(),
            rhs.len()
        )
    }
    if !lhs.dtype().is_boolean()
        || !rhs.dtype().is_boolean()
        || !lhs.dtype().eq_ignore_nullability(rhs.dtype())
    {
        vortex_bail!(
            "Boolean operations are only supported on boolean arrays: {} and {}",
            lhs.dtype(),
            rhs.dtype()
        )
    }

    let rhs_is_constant = rhs.is_constant();

    // If LHS is constant, then we make sure it's on the RHS.
    if lhs.is_constant() && !rhs_is_constant {
        return binary_boolean(rhs, lhs, op);
    }

    // If the RHS is constant and the LHS is Arrow, we can't do any better than arrow_compare.
    if lhs.is_arrow() && (rhs.is_arrow() || rhs_is_constant) {
        return arrow_boolean(lhs.to_array(), rhs.to_array(), op);
    }

    // Check if either LHS or RHS supports the operation directly.
    if let Some(result) = lhs
        .vtable()
        .binary_boolean_fn()
        .and_then(|f| f.binary_boolean(lhs, rhs, op).transpose())
        .transpose()?
    {
        assert_eq!(
            result.len(),
            lhs.len(),
            "Boolean operation length mismatch {}",
            lhs.encoding()
        );
        assert_eq!(
            result.dtype(),
            &DType::Bool((lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into()),
            "Boolean operation dtype mismatch {}",
            lhs.encoding()
        );
        return Ok(result);
    }

    if let Some(result) = rhs
        .vtable()
        .binary_boolean_fn()
        .and_then(|f| f.binary_boolean(rhs, lhs, op).transpose())
        .transpose()?
    {
        assert_eq!(
            result.len(),
            lhs.len(),
            "Boolean operation length mismatch {}",
            rhs.encoding()
        );
        assert_eq!(
            result.dtype(),
            &DType::Bool((lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into()),
            "Boolean operation dtype mismatch {}",
            rhs.encoding()
        );
        return Ok(result);
    }

    log::debug!(
        "No boolean implementation found for LHS {}, RHS {}, and operator {:?} (or inverse)",
        rhs.encoding(),
        lhs.encoding(),
        op,
    );

    // If neither side implements the trait, then we delegate to Arrow compute.
    arrow_boolean(lhs.to_array(), rhs.to_array(), op)
}

/// Implementation of `BinaryBooleanFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
pub(crate) fn arrow_boolean(
    lhs: ArrayRef,
    rhs: ArrayRef,
    operator: BinaryOperator,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    let lhs = lhs.into_arrow(&DataType::Boolean)?.as_boolean().clone();
    let rhs = rhs.into_arrow(&DataType::Boolean)?.as_boolean().clone();

    let array = match operator {
        BinaryOperator::And => arrow_arith::boolean::and(&lhs, &rhs)?,
        BinaryOperator::AndKleene => arrow_arith::boolean::and_kleene(&lhs, &rhs)?,
        BinaryOperator::Or => arrow_arith::boolean::or(&lhs, &rhs)?,
        BinaryOperator::OrKleene => arrow_arith::boolean::or_kleene(&lhs, &rhs)?,
    };

    Ok(ArrayRef::from_arrow(
        Arc::new(array) as ArrowArrayRef,
        nullable,
    ))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::arrays::BoolArray;
    use crate::canonical::ToCanonical;
    use crate::compute::scalar_at;

    #[rstest]
    #[case(BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter())
    .into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter())
    .into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter()).into_array())]
    fn test_or(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = or(&lhs, &rhs).unwrap();

        let r = r.to_bool().unwrap().into_array();

        let v0 = scalar_at(&r, 0).unwrap().as_bool().value();
        let v1 = scalar_at(&r, 1).unwrap().as_bool().value();
        let v2 = scalar_at(&r, 2).unwrap().as_bool().value();
        let v3 = scalar_at(&r, 3).unwrap().as_bool().value();

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
        let r = and(&lhs, &rhs).unwrap().to_bool().unwrap().into_array();

        let v0 = scalar_at(&r, 0).unwrap().as_bool().value();
        let v1 = scalar_at(&r, 1).unwrap().as_bool().value();
        let v2 = scalar_at(&r, 2).unwrap().as_bool().value();
        let v3 = scalar_at(&r, 3).unwrap().as_bool().value();

        assert!(v0.unwrap());
        assert!(!v1.unwrap());
        assert!(!v2.unwrap());
        assert!(!v3.unwrap());
    }
}
