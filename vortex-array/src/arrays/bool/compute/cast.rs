// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayRef;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::dtype::DType;
use crate::expr::CastReduce;
use crate::vtable::ValidityHelper;

impl CastReduce for BoolVTable {
    fn cast(array: &BoolArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !matches!(dtype, DType::Bool(_)) {
            return Ok(None);
        }

        let new_nullability = dtype.nullability();
        let new_validity = array
            .validity()
            .clone()
            .cast_nullability(new_nullability, array.len())?;
        Ok(Some(
            BoolArray::new(array.to_bit_buffer(), new_validity).to_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::arrays::BoolArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[test]
    fn try_cast_bool_success() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), Some(true)]);

        let res = bool.to_array().cast(DType::Bool(Nullability::NonNullable));
        assert!(res.is_ok());
        assert_eq!(res.unwrap().dtype(), &DType::Bool(Nullability::NonNullable));
    }

    #[test]
    #[should_panic]
    fn try_cast_bool_fail() {
        let bool = BoolArray::from_iter(vec![Some(true), Some(false), None]);
        bool.to_array()
            .cast(DType::Bool(Nullability::NonNullable))
            .unwrap();
    }

    #[rstest]
    #[case(BoolArray::from_iter(vec![true, false, true, true, false]))]
    #[case(BoolArray::from_iter(vec![Some(true), Some(false), None, Some(true), None]))]
    #[case(BoolArray::from_iter(vec![true]))]
    #[case(BoolArray::from_iter(vec![false, false]))]
    fn test_cast_bool_conformance(#[case] array: BoolArray) {
        test_cast_conformance(array.as_ref());
    }
}
