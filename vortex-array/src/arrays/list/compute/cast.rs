// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::list::ListArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for List {
    fn cast(array: ArrayView<'_, List>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let Some(target_element_type) = dtype.as_list_element_opt() else {
            return Ok(None);
        };

        let validity = array
            .validity()?
            .cast_nullability(dtype.nullability(), array.len())?;

        let new_elements = array.elements().cast((**target_element_type).clone())?;

        ListArray::try_new(new_elements, array.offsets().clone(), validity)
            .map(|a| Some(a.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::RecursiveCanonical;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[test]
    fn test_cast_list_success() {
        let list = ListArray::try_new(
            buffer![1i32, 2, 3, 4].into_array(),
            buffer![0, 2, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::List(
            Arc::new(DType::Primitive(PType::U64, Nullability::Nullable)),
            Nullability::Nullable,
        );

        let result = list
            .clone()
            .into_array()
            .cast(target_dtype.clone())
            .unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), list.len());
    }

    #[test]
    fn test_cast_to_wrong_type() {
        let list = ListArray::try_new(
            buffer![0i32, 2, 3, 4].into_array(),
            buffer![0, 2, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        // can't cast list to u64

        let result = list.into_array().cast(target_dtype).and_then(|a| {
            #[expect(deprecated)]
            let canonical = a.to_canonical().map(|c| c.into_array());
            canonical
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_cant_cast_nulls_to_non_null() {
        // Test that if list has nulls, the conversion will fail

        // Nulls in the list itself
        let list = ListArray::try_new(
            buffer![0i32, 2, 3, 4].into_array(),
            buffer![0, 2, 3].into_array(),
            Validity::Array(BoolArray::from_iter(vec![false, true]).into_array()),
        )
        .unwrap();

        let target_dtype = DType::List(
            Arc::new(DType::Primitive(PType::U64, Nullability::Nullable)),
            Nullability::NonNullable,
        );

        let result = list.into_array().cast(target_dtype).and_then(|a| {
            #[expect(deprecated)]
            let canonical = a.to_canonical().map(|c| c.into_array());
            canonical
        });
        assert!(result.is_err());

        // Nulls in list element array — the inner cast error is deferred until
        // the elements are executed.
        let list = ListArray::try_new(
            PrimitiveArray::from_option_iter([Some(0i32), Some(2), None, None]).into_array(),
            buffer![0, 2, 3].into_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::List(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            Nullability::NonNullable,
        );

        let result = list.into_array().cast(target_dtype).and_then(|a| {
            a.execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
                .map(|c| c.0.into_array())
        });
        assert!(result.is_err());
    }

    #[rstest]
    #[case(create_simple_list())]
    #[case(create_nullable_list())]
    #[case(create_string_list())]
    #[case(create_nested_list())]
    #[case(create_empty_lists())]
    fn test_cast_list_conformance(#[case] array: ListArray) {
        test_cast_conformance(&array.into_array());
    }

    fn create_simple_list() -> ListArray {
        let data = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let offsets = buffer![0i64, 2, 2, 5, 6].into_array();

        ListArray::try_new(data, offsets, Validity::NonNullable).unwrap()
    }

    fn create_nullable_list() -> ListArray {
        let data = PrimitiveArray::from_option_iter([
            Some(10i64),
            None,
            Some(20),
            Some(30),
            None,
            Some(40),
        ])
        .into_array();
        let offsets = buffer![0i64, 3, 6].into_array();
        let validity = Validity::Array(BoolArray::from_iter(vec![true, false]).into_array());

        ListArray::try_new(data, offsets, validity).unwrap()
    }

    fn create_string_list() -> ListArray {
        let data = VarBinArray::from_iter(
            vec![Some("hello"), Some("world"), Some("foo"), Some("bar")],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let offsets = buffer![0i64, 2, 4].into_array();

        ListArray::try_new(data, offsets, Validity::NonNullable).unwrap()
    }

    fn create_nested_list() -> ListArray {
        // Create inner lists: [[1, 2], [3], [4, 5, 6]]
        let inner_data = buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let inner_offsets = buffer![0i64, 2, 3, 6].into_array();
        let inner_list = ListArray::try_new(inner_data, inner_offsets, Validity::NonNullable)
            .unwrap()
            .into_array();

        // Create outer list: [[[1, 2], [3]], [[4, 5, 6]]]
        let outer_offsets = buffer![0i64, 2, 3].into_array();

        ListArray::try_new(inner_list, outer_offsets, Validity::NonNullable).unwrap()
    }

    fn create_empty_lists() -> ListArray {
        let data = buffer![42u8].into_array();
        let offsets = buffer![0i64, 0, 0, 1].into_array();

        ListArray::try_new(data, offsets, Validity::NonNullable).unwrap()
    }
}
