use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::compute::{BooleanKernel, BooleanKernelAdapter, BooleanOperator};
use crate::{Array, ArrayRef, register_kernel};

impl BooleanKernel for ConstantEncoding {
    fn binary_boolean(
        &self,
        lhs: &ConstantArray,
        rhs: &dyn Array,
        op: BooleanOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        // We only implement this for constant <-> constant arrays, otherwise we allow fall back
        // to the Arrow implementation.
        if !rhs.is_constant() {
            return Ok(None);
        }

        let length = lhs.len();
        let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
        let lhs = lhs.scalar().as_bool().value();
        let Some(rhs) = rhs.as_constant() else {
            vortex_bail!("Binary boolean operation requires both sides to be constant");
        };
        let rhs = rhs
            .as_bool_opt()
            .ok_or_else(|| vortex_err!("expected rhs to be boolean"))?
            .value();

        let result = match op {
            BooleanOperator::And => and(lhs, rhs),
            BooleanOperator::AndKleene => kleene_and(lhs, rhs),
            BooleanOperator::Or => or(lhs, rhs),
            BooleanOperator::OrKleene => kleene_or(lhs, rhs),
        };

        let scalar = result
            .map(|b| Scalar::bool(b, nullable.into()))
            .unwrap_or_else(|| Scalar::null(DType::Bool(nullable.into())));

        Ok(Some(ConstantArray::new(scalar, length).into_array()))
    }
}

register_kernel!(BooleanKernelAdapter(ConstantEncoding).lift());

fn and(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    left.zip(right).map(|(l, r)| l & r)
}

fn kleene_and(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(false), _) => Some(false),
        (_, Some(false)) => Some(false),
        (None, _) => None,
        (_, None) => None,
        (Some(l), Some(r)) => Some(l & r),
    }
}

fn or(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    left.zip(right).map(|(l, r)| l | r)
}

fn kleene_or(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(true), _) => Some(true),
        (_, Some(true)) => Some(true),
        (None, _) => None,
        (_, None) => None,
        (Some(l), Some(r)) => Some(l | r),
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::arrays::BoolArray;
    use crate::arrays::constant::ConstantArray;
    use crate::canonical::ToCanonical;
    use crate::compute::{and, or, scalar_at};
    use crate::{Array, ArrayRef};

    #[rstest]
    #[case(ConstantArray::new(true, 4).into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array()
    )]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(), ConstantArray::new(true, 4).into_array()
    )]
    fn test_or(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = or(&lhs, &rhs).unwrap().to_bool().unwrap().into_array();

        let v0 = scalar_at(&r, 0).unwrap().as_bool().value();
        let v1 = scalar_at(&r, 1).unwrap().as_bool().value();
        let v2 = scalar_at(&r, 2).unwrap().as_bool().value();
        let v3 = scalar_at(&r, 3).unwrap().as_bool().value();

        assert!(v0.unwrap());
        assert!(v1.unwrap());
        assert!(v2.unwrap());
        assert!(v3.unwrap());
    }

    #[rstest]
    #[case(ConstantArray::new(true, 4).into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array()
    )]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(),
        ConstantArray::new(true, 4).into_array())]
    fn test_and(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = and(&lhs, &rhs).unwrap().to_bool().unwrap().into_array();

        let v0 = scalar_at(&r, 0).unwrap().as_bool().value();
        let v1 = scalar_at(&r, 1).unwrap().as_bool().value();
        let v2 = scalar_at(&r, 2).unwrap().as_bool().value();
        let v3 = scalar_at(&r, 3).unwrap().as_bool().value();

        assert!(v0.unwrap());
        assert!(!v1.unwrap());
        assert!(v2.unwrap());
        assert!(!v3.unwrap());
    }
}
