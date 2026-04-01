// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Bool {
    fn cast(array: ArrayView<'_, Bool>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !matches!(dtype, DType::Bool(_)) {
            return Ok(None);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array
            .validity()
            .cast_nullability(new_nullability, array.len())?;
        Ok(Some(
            BoolArray::new(array.to_bit_buffer(), new_validity).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[test]
    fn try_cast_bool_success() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), Some(true)]);

        let res = bool
            .into_array()
            .cast(DType::Bool(Nullability::NonNullable));
        assert!(res.is_ok());
        assert_eq!(res.unwrap().dtype(), &DType::Bool(Nullability::NonNullable));
    }

    #[test]
    #[should_panic]
    fn try_cast_bool_fail() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), None]);
        bool.into_array()
            .cast(DType::Bool(Nullability::NonNullable))
            .unwrap();
    }

    #[rstest]
    #[case(BoolArray::from_iter(vec![true, false, true, true, false]))]
    #[case(BoolArray::from_iter(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case(BoolArray::from_iter(vec![true]))]
    #[case(BoolArray::from_iter(vec![false, false]))]
    fn test_cast_bool_conformance(#[case] array: BoolArray) {
        test_cast_conformance(&array.into_array());
    }
}
