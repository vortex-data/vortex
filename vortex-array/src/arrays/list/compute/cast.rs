// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{ListArray, ListVTable};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, register_kernel};

impl CastKernel for ListVTable {
    fn cast(&self, array: &Self::Array, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let Some(target_element_type) = dtype.as_list_element_opt() else {
            return Ok(None);
        };

        let validity = array
            .validity()
            .clone()
            .cast_nullability(dtype.nullability(), array.len())?;

        ListArray::try_new(
            cast(array.elements(), target_element_type)?,
            array.offsets().clone(),
            validity,
        )
        .map(|a| Some(a.to_array()))
    }
}

register_kernel!(CastKernelAdapter(ListVTable).lift());

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::IntoArray;
    use crate::arrays::{BoolArray, ListArray, PrimitiveArray, VarBinArray};
    use crate::compute::cast;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::validity::Validity;

    #[test]
    fn test_cast_list_success() {
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4]).to_array(),
            PrimitiveArray::from_iter([0, 2, 3]).to_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::List(
            Arc::new(DType::Primitive(PType::U64, Nullability::Nullable)),
            Nullability::Nullable,
        );

        let result = cast(list.to_array().as_ref(), &target_dtype).unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), list.len());
    }

    #[test]
    fn test_cast_to_wrong_type() {
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([0i32, 2, 3, 4]).to_array(),
            PrimitiveArray::from_iter([0, 2, 3]).to_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        // can't cast list to u64

        let result = cast(list.to_array().as_ref(), &target_dtype);
        assert!(result.is_err());
    }

    #[test]
    fn test_cant_cast_nulls_to_non_null() {
        // Test that if list has nulls, the conversion will fail

        // Nulls in the list itself
        let list = ListArray::try_new(
            PrimitiveArray::from_iter([0i32, 2, 3, 4]).to_array(),
            PrimitiveArray::from_iter([0, 2, 3]).to_array(),
            Validity::Array(BoolArray::from_iter(vec![false, true]).to_array()),
        )
        .unwrap();

        let target_dtype = DType::List(
            Arc::new(DType::Primitive(PType::U64, Nullability::Nullable)),
            Nullability::NonNullable,
        );

        let result = cast(list.to_array().as_ref(), &target_dtype);
        assert!(result.is_err());

        // Nulls in list element array
        let list = ListArray::try_new(
            PrimitiveArray::from_option_iter([Some(0i32), Some(2), None, None]).to_array(),
            PrimitiveArray::from_iter([0, 2, 3]).to_array(),
            Validity::NonNullable,
        )
        .unwrap();

        let target_dtype = DType::List(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            Nullability::NonNullable,
        );

        let result = cast(list.to_array().as_ref(), &target_dtype);
        assert!(result.is_err());
    }

    #[rstest]
    #[case(create_simple_list())]
    #[case(create_nullable_list())]
    #[case(create_string_list())]
    #[case(create_nested_list())]
    #[case(create_empty_lists())]
    fn test_cast_list_conformance(#[case] array: ListArray) {
        test_cast_conformance(array.as_ref());
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
