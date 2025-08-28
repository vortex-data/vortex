// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::{DType, Nullability, PType};
use vortex_scalar::Scalar;

use crate::arrays::{FixedSizeListArray, PrimitiveArray};
use crate::validity::Validity;
use crate::{Array, IntoArray};

#[test]
fn test_basic_fixed_size_list() {
    let len = 4;
    let list_size = 3;

    // Create a FSL of size 3 with 4 lists: [[1,2,3], [4,5,6], [7,8,9], [10,11,12]].
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
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
}

#[test]
fn test_scalar_at() {
    let len = 2;
    let list_size = 3;

    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    // First list: [1, 2, 3].
    let first = fsl.scalar_at(0);
    assert_eq!(
        first,
        Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![1i32.into(), 2i32.into(), 3i32.into()],
            Nullability::NonNullable,
        )
    );

    // Second list: [4, 5, 6].
    let second = fsl.scalar_at(1);
    assert_eq!(
        second,
        Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![4i32.into(), 5i32.into(), 6i32.into()],
            Nullability::NonNullable,
        )
    );
}

#[test]
fn test_fixed_size_list_at() {
    let len = 3;
    let list_size = 2;

    let elements = PrimitiveArray::from_iter([1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::AllValid, len);

    // Get the first list [1.0, 2.0].
    let first_list = fsl.fixed_size_list_at(0);
    assert_eq!(first_list.len(), list_size as usize);
    assert_eq!(first_list.scalar_at(0), 1.0f64.into());
    assert_eq!(first_list.scalar_at(1), 2.0f64.into());

    // Get the third list [5.0, 6.0].
    let third_list = fsl.fixed_size_list_at(2);
    assert_eq!(third_list.len(), list_size as usize);
    assert_eq!(third_list.scalar_at(0), 5.0f64.into());
    assert_eq!(third_list.scalar_at(1), 6.0f64.into());
}

#[test]
fn test_validation_error_length_mismatch() {
    let len = 2;
    let list_size = 3;

    // Elements length is not a multiple of list_size.
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
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

    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);

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
