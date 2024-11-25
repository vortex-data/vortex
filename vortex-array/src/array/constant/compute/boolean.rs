use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_scalar::Scalar;

use crate::array::{ConstantArray, ConstantEncoding};
use crate::compute::{BinaryBooleanFn, BinaryOperator};
use crate::{ArrayDType, ArrayData, ArrayLen, IntoArrayData};

impl BinaryBooleanFn<ConstantArray> for ConstantEncoding {
    fn binary_boolean(
        &self,
        lhs: &ConstantArray,
        rhs: &ArrayData,
        op: BinaryOperator,
    ) -> VortexResult<ArrayData> {
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
            BinaryOperator::And => and(lhs, rhs),
            BinaryOperator::AndKleene => kleene_and(lhs, rhs),
            BinaryOperator::Or => or(lhs, rhs),
            BinaryOperator::OrKleene => kleene_or(lhs, rhs),
        };

        let scalar = result
            .map(|b| Scalar::bool(b, nullable.into()))
            .unwrap_or_else(|| Scalar::null(DType::Bool(nullable.into())));

        Ok(ConstantArray::new(scalar, length).into_array())
    }
}

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

    use crate::array::constant::ConstantArray;
    use crate::array::BoolArray;
    use crate::compute::unary::scalar_at;
    use crate::compute::{and, or};
    use crate::{ArrayData, IntoArrayData, IntoArrayVariant};

    #[rstest]
    #[case(ConstantArray::new(true, 4).into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array()
    )]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(), ConstantArray::new(true, 4).into_array()
    )]
    fn test_or(#[case] lhs: ArrayData, #[case] rhs: ArrayData) {
        let r = or(&lhs, &rhs).unwrap().into_bool().unwrap().into_array();

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
    fn test_and(#[case] lhs: ArrayData, #[case] rhs: ArrayData) {
        let r = and(&lhs, &rhs).unwrap().into_bool().unwrap().into_array();

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
