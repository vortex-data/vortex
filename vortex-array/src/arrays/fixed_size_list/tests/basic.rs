// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::buffer;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::validity::Validity;

#[test]
fn test_basic_fixed_size_list() {
    let len = 4;
    let list_size = 3;

    // Create a FSL of size 3 with 4 lists: [[1,2,3], [4,5,6], [7,8,9], [10,11,12]].
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    assert_eq!(fsl.len(), len);
    assert_eq!(fsl.list_size(), list_size);
    assert_eq!(fsl.elements().len(), (len * list_size as usize));

    // Check the dtype.
    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(elem_dtype, 3, Nullability::NonNullable)
            if matches!(elem_dtype.as_ref(), DType::Primitive(PType::I32, Nullability::NonNullable))
    ));

    // Check the actual values in each list.
    let first_list = fsl.fixed_size_list_elements_at(0).unwrap();
    assert_eq!(
        first_list
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap(),
        1i32.into()
    );
    assert_eq!(
        first_list
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap(),
        2i32.into()
    );
    assert_eq!(
        first_list
            .execute_scalar(2, &mut array_session().create_execution_ctx())
            .unwrap(),
        3i32.into()
    );

    let second_list = fsl.fixed_size_list_elements_at(1).unwrap();
    assert_eq!(
        second_list
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap(),
        4i32.into()
    );
    assert_eq!(
        second_list
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap(),
        5i32.into()
    );
    assert_eq!(
        second_list
            .execute_scalar(2, &mut array_session().create_execution_ctx())
            .unwrap(),
        6i32.into()
    );

    let third_list = fsl.fixed_size_list_elements_at(2).unwrap();
    assert_eq!(
        third_list
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap(),
        7i32.into()
    );
    assert_eq!(
        third_list
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap(),
        8i32.into()
    );
    assert_eq!(
        third_list
            .execute_scalar(2, &mut array_session().create_execution_ctx())
            .unwrap(),
        9i32.into()
    );

    let fourth_list = fsl.fixed_size_list_elements_at(3).unwrap();
    assert_eq!(
        fourth_list
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap(),
        10i32.into()
    );
    assert_eq!(
        fourth_list
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap(),
        11i32.into()
    );
    assert_eq!(
        fourth_list
            .execute_scalar(2, &mut array_session().create_execution_ctx())
            .unwrap(),
        12i32.into()
    );
}

#[test]
fn test_scalar_at() {
    let len = 2;
    let list_size = 3;

    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    // First list: [1, 2, 3].
    let first = fsl
        .execute_scalar(0, &mut array_session().create_execution_ctx())
        .unwrap();
    assert_eq!(
        first,
        Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![1i32.into(), 2i32.into(), 3i32.into()],
            Nullability::NonNullable,
        )
    );

    // Additionally check individual elements via fixed_size_list_at.
    let first_list = fsl.fixed_size_list_elements_at(0).unwrap();
    assert_eq!(
        first_list
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap(),
        1i32.into()
    );
    assert_eq!(
        first_list
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap(),
        2i32.into()
    );
    assert_eq!(
        first_list
            .execute_scalar(2, &mut array_session().create_execution_ctx())
            .unwrap(),
        3i32.into()
    );

    // Second list: [4, 5, 6].
    let second = fsl
        .execute_scalar(1, &mut array_session().create_execution_ctx())
        .unwrap();
    assert_eq!(
        second,
        Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![4i32.into(), 5i32.into(), 6i32.into()],
            Nullability::NonNullable,
        )
    );

    // Additionally check individual elements via fixed_size_list_at.
    let second_list = fsl.fixed_size_list_elements_at(1).unwrap();
    assert_eq!(
        second_list
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap(),
        4i32.into()
    );
    assert_eq!(
        second_list
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap(),
        5i32.into()
    );
    assert_eq!(
        second_list
            .execute_scalar(2, &mut array_session().create_execution_ctx())
            .unwrap(),
        6i32.into()
    );
}

#[test]
fn test_fixed_size_list_at() {
    let len = 3;
    let list_size = 2;

    let elements = buffer![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0].into_array();
    let fsl = FixedSizeListArray::new(elements, list_size, Validity::AllValid, len);

    // Get the first list [1.0, 2.0].
    let first_list = fsl.fixed_size_list_elements_at(0).unwrap();
    assert_eq!(first_list.len(), list_size as usize);
    assert_eq!(
        first_list
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap(),
        1.0f64.into()
    );
    assert_eq!(
        first_list
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap(),
        2.0f64.into()
    );

    // Get the third list [5.0, 6.0].
    let third_list = fsl.fixed_size_list_elements_at(2).unwrap();
    assert_eq!(third_list.len(), list_size as usize);
    assert_eq!(
        third_list
            .execute_scalar(0, &mut array_session().create_execution_ctx())
            .unwrap(),
        5.0f64.into()
    );
    assert_eq!(
        third_list
            .execute_scalar(1, &mut array_session().create_execution_ctx())
            .unwrap(),
        6.0f64.into()
    );
}

#[test]
fn test_validation_error_length_mismatch() {
    let len = 2;
    let list_size = 3;

    // Elements length is not a multiple of list_size.
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let result = FixedSizeListArray::try_new(
        elements.into_array(),
        list_size, // List size is 3, but we have 5 elements (not enough for 2 complete lists).
        Validity::NonNullable,
        len, // Claiming 2 lists would need 6 elements.
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("incorrect number of elements"));
}

#[test]
fn test_validation_error_validity_length() {
    let len = 3;
    let list_size = 2;

    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();

    // Create a validity array with wrong length.
    let validity = Validity::from_iter([true, false]); // Length 2.

    let result = FixedSizeListArray::try_new(
        elements.into_array(),
        list_size,
        validity,
        len, // Array length is 3, but validity has length 2.
    );

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("does not match"));
}
