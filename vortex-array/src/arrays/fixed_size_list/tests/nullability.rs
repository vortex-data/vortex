// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::buffer;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_scalar::Scalar;

use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::validity::Validity;

#[test]
fn test_nullable_fsl_with_nulls() {
    let len = 4;
    let list_size = 2;

    // Create FSL with some null lists.
    let elements = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
    let validity = Validity::from_iter([true, false, true, false]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, validity, len);

    assert_eq!(fsl.len(), len);
    assert_eq!(fsl.list_size(), list_size);

    // First list is valid: [1, 2].
    let first = fsl.scalar_at(0).unwrap();
    assert!(!first.is_null());
    assert_eq!(
        first,
        Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![1i32.into(), 2i32.into()],
            Nullability::Nullable,
        )
    );

    // Check individual elements of the first list.
    let first_list = fsl.fixed_size_list_elements_at(0).unwrap();
    assert_eq!(first_list.scalar_at(0).unwrap(), 1i32.into());
    assert_eq!(first_list.scalar_at(1).unwrap(), 2i32.into());

    // Second list is null.
    let second = fsl.scalar_at(1).unwrap();
    assert!(second.is_null());

    // Third list is valid: [5, 6].
    let third = fsl.scalar_at(2).unwrap();
    assert!(!third.is_null());
    assert_eq!(
        third,
        Scalar::fixed_size_list(
            Arc::new(PType::I32.into()),
            vec![5i32.into(), 6i32.into()],
            Nullability::Nullable,
        )
    );

    // Check individual elements of the third list.
    let third_list = fsl.fixed_size_list_elements_at(2).unwrap();
    assert_eq!(third_list.scalar_at(0).unwrap(), 5i32.into());
    assert_eq!(third_list.scalar_at(1).unwrap(), 6i32.into());

    // Fourth list is null.
    let fourth = fsl.scalar_at(3).unwrap();
    assert!(fourth.is_null());
}

#[test]
fn test_nullable_elements_non_nullable_lists() {
    let len = 2;
    let list_size = 3;

    // Elements array has nulls but the FSL itself is non-nullable.
    let elements =
        PrimitiveArray::from_option_iter(vec![Some(1i32), None, Some(3), Some(4), Some(5), None]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    assert_eq!(fsl.len(), len);

    // Check dtype - FSL is non-nullable but elements are nullable.
    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(elem_dtype, 3, Nullability::NonNullable)
            if matches!(elem_dtype.as_ref(), DType::Primitive(PType::I32, Nullability::Nullable))
    ));

    // First list: [Some(1), None, Some(3)].
    let first = fsl.scalar_at(0).unwrap();
    assert!(!first.is_null());
    assert_eq!(
        first,
        Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![Some(1i32).into(), None::<i32>.into(), Some(3i32).into(),],
            Nullability::NonNullable,
        )
    );

    // Second list: [Some(4), Some(5), None].
    let second = fsl.scalar_at(1).unwrap();
    assert!(!second.is_null());
    assert_eq!(
        second,
        Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            vec![Some(4i32).into(), Some(5i32).into(), None::<i32>.into(),],
            Nullability::NonNullable,
        )
    );
}

#[test]
fn test_nullable_elements_and_nullable_lists() {
    let len = 3;
    let list_size = 2;

    // Both elements and lists can be null.
    let elements =
        PrimitiveArray::from_option_iter(vec![Some(10u16), None, Some(20), Some(30), None, None]);
    let validity = Validity::from_iter([true, false, true]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, validity, len);

    assert_eq!(fsl.len(), len);

    // First list is valid: [Some(10), None].
    let first = fsl.scalar_at(0).unwrap();
    assert!(!first.is_null());
    assert_eq!(
        first,
        Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::U16, Nullability::Nullable)),
            vec![Some(10u16).into(), None::<u16>.into()],
            Nullability::Nullable,
        )
    );

    // Check individual elements of the first list.
    let first_list = fsl.fixed_size_list_elements_at(0).unwrap();
    assert_eq!(first_list.scalar_at(0).unwrap(), Some(10u16).into());
    assert_eq!(first_list.scalar_at(1).unwrap(), None::<u16>.into());

    // Second list is null (but elements would be [Some(20), Some(30)]).
    let second = fsl.scalar_at(1).unwrap();
    assert!(second.is_null());

    // Third list is valid: [None, None].
    let third = fsl.scalar_at(2).unwrap();
    assert!(!third.is_null());
    assert_eq!(
        third,
        Scalar::fixed_size_list(
            Arc::new(DType::Primitive(PType::U16, Nullability::Nullable)),
            vec![None::<u16>.into(), None::<u16>.into()],
            Nullability::Nullable,
        )
    );

    // Check individual elements of the third list.
    let third_list = fsl.fixed_size_list_elements_at(2).unwrap();
    assert_eq!(third_list.scalar_at(0).unwrap(), None::<u16>.into());
    assert_eq!(third_list.scalar_at(1).unwrap(), None::<u16>.into());
}

#[test]
fn test_alternating_nulls() {
    let len = 6;
    let list_size = 1;

    // Alternating null and valid single-element lists.
    let elements = buffer![1u8, 2, 3, 4, 5, 6].into_array();
    let validity = Validity::from_iter([true, false, true, false, true, false]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, validity, len);

    assert_eq!(fsl.len(), len);

    // Check alternating pattern.
    for i in 0..len {
        let scalar = fsl.scalar_at(i).unwrap();
        if i % 2 == 0 {
            assert!(!scalar.is_null());
            let expected_value = u8::try_from(i + 1).unwrap();
            assert_eq!(
                scalar,
                Scalar::fixed_size_list(
                    Arc::new(PType::U8.into()),
                    vec![expected_value.into()],
                    Nullability::Nullable,
                )
            );
        } else {
            assert!(scalar.is_null());
        }
    }
}

#[test]
fn test_validity_types() {
    let len = 4;
    let list_size = 2;

    // Test with different validity buffer configurations.
    let elements = buffer![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into_array();

    // Test with AllInvalid.
    {
        let fsl = FixedSizeListArray::new(elements.clone(), list_size, Validity::AllInvalid, len);
        for i in 0..len {
            assert!(fsl.scalar_at(i).unwrap().is_null());
        }
    }

    // Test with Array validity.
    {
        let validity_array = BoolArray::from_iter([true, true, false, true]);
        let fsl = FixedSizeListArray::new(
            elements,
            list_size,
            Validity::Array(validity_array.into_array()),
            len,
        );

        assert!(!fsl.scalar_at(0).unwrap().is_null());
        assert!(!fsl.scalar_at(1).unwrap().is_null());
        assert!(fsl.scalar_at(2).unwrap().is_null());
        assert!(!fsl.scalar_at(3).unwrap().is_null());
    }
}

#[test]
fn test_mixed_nullability_patterns() {
    let len = 5;
    let list_size = 2;

    // Complex nullability pattern.
    let elements = PrimitiveArray::from_option_iter(vec![
        Some(1i16), // List 0
        None,
        None, // List 1 (null list)
        None,
        Some(5), // List 2
        Some(6),
        Some(7), // List 3
        None,
        None, // List 4
        Some(10),
    ]);
    let validity = Validity::from_iter([true, false, true, true, true]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, validity, len);

    // List 0: valid with [Some(1), None].
    let list0 = fsl.scalar_at(0).unwrap();
    assert!(!list0.is_null());

    // List 1: null.
    let list1 = fsl.scalar_at(1).unwrap();
    assert!(list1.is_null());

    // List 2: valid with [Some(5), Some(6)].
    let list2 = fsl.scalar_at(2).unwrap();
    assert!(!list2.is_null());

    // List 3: valid with [Some(7), None].
    let list3 = fsl.scalar_at(3).unwrap();
    assert!(!list3.is_null());

    // List 4: valid with [None, Some(10)].
    let list4 = fsl.scalar_at(4).unwrap();
    assert!(!list4.is_null());
}
