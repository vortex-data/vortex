// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullReduce;
use crate::scalar_fn::fns::fill_null::fill_null_constant;

impl FillNullReduce for Constant {
    fn fill_null(
        array: ArrayView<'_, Constant>,
        fill_value: &Scalar,
    ) -> VortexResult<Option<ArrayRef>> {
        fill_null_constant(array, fill_value).map(Some)
    }
}

#[cfg(test)]
mod test {
    use crate::IntoArray as _;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;

    fn scalar_at(array: &crate::ArrayRef, i: usize) -> Scalar {
        array
            .execute_scalar(i, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    }

    #[test]
    fn test_null() {
        let actual = ConstantArray::new(Scalar::null_native::<i32>(), 3)
            .into_array()
            .fill_null(Scalar::from(1))
            .unwrap();
        assert!(!actual.dtype().is_nullable());
        for i in 0..actual.len() {
            assert_eq!(scalar_at(&actual, i).as_primitive().as_::<i32>().unwrap(), Some(1));
        }
    }

    #[test]
    fn test_non_null() {
        let actual = ConstantArray::new(Scalar::from(Some(1)), 3)
            .into_array()
            .fill_null(Scalar::from(1))
            .unwrap();
        assert!(!actual.dtype().is_nullable());
        for i in 0..actual.len() {
            assert_eq!(scalar_at(&actual, i).as_primitive().as_::<i32>().unwrap(), Some(1));
        }
    }

    #[test]
    fn test_non_nullable_with_nullable() {
        let actual = ConstantArray::new(Scalar::from(1), 3)
            .into_array()
            .fill_null(Scalar::new(
                DType::Primitive(PType::I32, Nullability::Nullable),
                Some(1.into()),
            ))
            .unwrap();

        assert!(!Scalar::from(1).dtype().is_nullable());
        assert!(actual.dtype().is_nullable());
        for i in 0..actual.len() {
            assert_eq!(scalar_at(&actual, i).as_primitive().as_::<i32>().unwrap(), Some(1));
        }
    }
}
