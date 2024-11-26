use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::ArrayRef;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::arrow::FromArrowArray;
use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData, Canonical, IntoArrayVariant};

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

pub trait BinaryBooleanFn<Array> {
    fn binary_boolean(
        &self,
        array: &Array,
        other: &ArrayData,
        op: BinaryOperator,
    ) -> VortexResult<ArrayData>;
}

impl<E: Encoding> BinaryBooleanFn<ArrayData> for E
where
    E: BinaryBooleanFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn binary_boolean(
        &self,
        lhs: &ArrayData,
        rhs: &ArrayData,
        op: BinaryOperator,
    ) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(lhs)?;
        let encoding = lhs
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        BinaryBooleanFn::binary_boolean(encoding, array_ref, rhs, op)
    }
}

/// Point-wise logical _and_ between two Boolean arrays.
///
/// This method uses Arrow-style null propagation rather than the Kleene logic semantics.
pub fn and(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_boolean(lhs.as_ref(), rhs.as_ref(), BinaryOperator::And)
}

/// Point-wise Kleene logical _and_ between two Boolean arrays.
pub fn and_kleene(
    lhs: impl AsRef<ArrayData>,
    rhs: impl AsRef<ArrayData>,
) -> VortexResult<ArrayData> {
    binary_boolean(lhs.as_ref(), rhs.as_ref(), BinaryOperator::AndKleene)
}

/// Point-wise logical _or_ between two Boolean arrays.
///
/// This method uses Arrow-style null propagation rather than the Kleene logic semantics.
pub fn or(lhs: impl AsRef<ArrayData>, rhs: impl AsRef<ArrayData>) -> VortexResult<ArrayData> {
    binary_boolean(lhs.as_ref(), rhs.as_ref(), BinaryOperator::Or)
}

/// Point-wise Kleene logical _or_ between two Boolean arrays.
pub fn or_kleene(
    lhs: impl AsRef<ArrayData>,
    rhs: impl AsRef<ArrayData>,
) -> VortexResult<ArrayData> {
    binary_boolean(lhs.as_ref(), rhs.as_ref(), BinaryOperator::OrKleene)
}

fn binary_boolean(lhs: &ArrayData, rhs: &ArrayData, op: BinaryOperator) -> VortexResult<ArrayData> {
    if lhs.len() != rhs.len() {
        vortex_bail!("Boolean operations aren't supported on arrays of different lengths")
    }
    if !lhs.dtype().is_boolean() || !rhs.dtype().is_boolean() {
        vortex_bail!("Boolean operations are only supported on boolean arrays")
    }

    // If LHS is constant, then we make sure it's on the RHS.
    if lhs.is_constant() && !rhs.is_constant() {
        return binary_boolean(rhs, lhs, op);
    }

    // Check if either LHS or RHS supports the operation directly.
    if let Some(f) = lhs.encoding().binary_boolean_fn(lhs, rhs) {
        return f.binary_boolean(lhs, rhs, op);
    } else {
        log::debug!(
            "No boolean implementation found for LHS {}, RHS {}, and operator {:?}",
            lhs.encoding().id(),
            rhs.encoding().id(),
            op,
        );
    }
    if let Some(f) = rhs.encoding().binary_boolean_fn(rhs, lhs) {
        return f.binary_boolean(rhs, lhs, op);
    } else {
        log::debug!(
            "No boolean implementation found for LHS {}, RHS {}, and operator {:?}",
            rhs.encoding().id(),
            lhs.encoding().id(),
            op,
        );
    }

    // If neither side implements the trait, then we delegate to Arrow compute.
    arrow_boolean(lhs.clone(), rhs.clone(), op)
}

/// Implementation of `BinaryBooleanFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
pub(crate) fn arrow_boolean(
    lhs: ArrayData,
    rhs: ArrayData,
    operator: BinaryOperator,
) -> VortexResult<ArrayData> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    let lhs = Canonical::Bool(lhs.into_bool()?)
        .into_arrow()?
        .as_boolean()
        .clone();
    let rhs = Canonical::Bool(rhs.into_bool()?)
        .into_arrow()?
        .as_boolean()
        .clone();

    let array = match operator {
        BinaryOperator::And => arrow_arith::boolean::and(&lhs, &rhs)?,
        BinaryOperator::AndKleene => arrow_arith::boolean::and_kleene(&lhs, &rhs)?,
        BinaryOperator::Or => arrow_arith::boolean::or(&lhs, &rhs)?,
        BinaryOperator::OrKleene => arrow_arith::boolean::or_kleene(&lhs, &rhs)?,
    };

    Ok(ArrayData::from_arrow(Arc::new(array) as ArrayRef, nullable))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::array::BoolArray;
    use crate::compute::unary::scalar_at;
    use crate::IntoArrayData;

    #[rstest]
    #[case(BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter())
    .into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter())
    .into_array())]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        BoolArray::from_iter([Some(true), Some(true), Some(false), Some(false)].into_iter()).into_array())]
    fn test_or(#[case] lhs: ArrayData, #[case] rhs: ArrayData) {
        let r = or(&lhs, &rhs).unwrap();

        let r = r.into_bool().unwrap().into_array();

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
    fn test_and(#[case] lhs: ArrayData, #[case] rhs: ArrayData) {
        let r = and(&lhs, &rhs).unwrap().into_bool().unwrap().into_array();

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
