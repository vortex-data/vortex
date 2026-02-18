// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;

/// Point-wise Kleene logical _and_ between two Boolean arrays.
#[deprecated(note = "use expr::and_kleene instead")]
pub fn and_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    crate::expr::and_kleene(lhs, rhs)
}

/// Point-wise Kleene logical _or_ between two Boolean arrays.
#[deprecated(note = "use expr::or_kleene instead")]
pub fn or_kleene(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    crate::expr::or_kleene(lhs, rhs)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::canonical::ToCanonical;
    use crate::expr::and_kleene;
    use crate::expr::or_kleene;

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
