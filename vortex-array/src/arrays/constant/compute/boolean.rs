// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::BooleanKernel;
use crate::compute::BooleanKernelAdapter;
use crate::compute::BooleanOperator;
use crate::register_kernel;

impl BooleanKernel for ConstantVTable {
    fn boolean(
        &self,
        lhs: &ConstantArray,
        rhs: &dyn Array,
        op: BooleanOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        // We only implement this for constant <-> constant arrays, otherwise we allow fall back
        // to the Arrow implementation.
        let Some(rhs) = rhs.as_opt::<ConstantVTable>() else {
            return Ok(None);
        };

        let length = lhs.len();
        let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
        let lhs = lhs.scalar().as_bool().value();
        let rhs = rhs
            .scalar()
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

register_kernel!(BooleanKernelAdapter(ConstantVTable).lift());

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

    use crate::Array;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::constant::ConstantArray;
    use crate::canonical::ToCanonical;
    use crate::compute::and;
    use crate::compute::or;

    #[rstest]
    #[case(ConstantArray::new(true, 4).into_array(), BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array()
    )]
    #[case(BoolArray::from_iter([Some(true), Some(false), Some(true), Some(false)].into_iter()).into_array(), ConstantArray::new(true, 4).into_array()
    )]
    fn test_or(#[case] lhs: ArrayRef, #[case] rhs: ArrayRef) {
        let r = or(&lhs, &rhs).unwrap().to_bool().into_array();

        let v0 = r.scalar_at(0).as_bool().value();
        let v1 = r.scalar_at(1).as_bool().value();
        let v2 = r.scalar_at(2).as_bool().value();
        let v3 = r.scalar_at(3).as_bool().value();

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
        let r = and(&lhs, &rhs).unwrap().to_bool().into_array();

        let v0 = r.scalar_at(0).as_bool().value();
        let v1 = r.scalar_at(1).as_bool().value();
        let v2 = r.scalar_at(2).as_bool().value();
        let v3 = r.scalar_at(3).as_bool().value();

        assert!(v0.unwrap());
        assert!(!v1.unwrap());
        assert!(v2.unwrap());
        assert!(!v3.unwrap());
    }
}
