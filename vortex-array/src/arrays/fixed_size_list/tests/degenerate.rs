// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::buffer;

use crate::IntoArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::validity::Validity;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Non-nullable degenerate cases
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_fsl_size_0_length_0_non_nullable() {
    let len = 0;
    let list_size = 0;

    // FSL of size 0 with length 0.
    let elements = PrimitiveArray::empty::<f32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    assert_eq!(fsl.len(), 0);
    assert_eq!(fsl.list_size(), 0);
    assert_eq!(fsl.elements().len(), 0);

    // Check dtype.
    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(elem_dtype, 0, Nullability::NonNullable)
            if matches!(elem_dtype.as_ref(), DType::Primitive(PType::F32, Nullability::NonNullable))
    ));
}

#[test]
fn test_fsl_size_0_length_1_non_nullable() {
    let len = 1;
    let list_size = 0;

    // FSL of size 0 with length 1 (one empty list).
    let elements = PrimitiveArray::empty::<i16>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    assert_eq!(fsl.len(), 1);
    assert_eq!(fsl.list_size(), 0);
    assert_eq!(fsl.elements().len(), 0);

    // Get the single empty list.
    let scalar = fsl.scalar_at(0).unwrap();
    assert!(!scalar.is_null());
    assert_eq!(
        scalar,
        Scalar::fixed_size_list(
            Arc::new(PType::I16.into()),
            vec![],
            Nullability::NonNullable,
        )
    );
}

#[test]
fn test_fsl_size_0_huge_length_non_nullable() {
    let len = 1_000_000_000_000;
    let list_size = 0;

    // FSL of size 0 with very large length. Should not store anything.
    let elements = PrimitiveArray::empty::<i64>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    assert_eq!(fsl.len(), len);
    assert_eq!(fsl.list_size(), 0);
    assert_eq!(fsl.elements().len(), 0);

    // Spot check a few lists.
    let scalar_first = fsl.scalar_at(0).unwrap();
    assert!(!scalar_first.is_null());
    assert_eq!(
        scalar_first,
        Scalar::fixed_size_list(
            Arc::new(PType::I64.into()),
            vec![],
            Nullability::NonNullable,
        )
    );

    let scalar_middle = fsl.scalar_at(500_000_000_000).unwrap();
    assert!(!scalar_middle.is_null());
    assert_eq!(
        scalar_middle,
        Scalar::fixed_size_list(
            Arc::new(PType::I64.into()),
            vec![],
            Nullability::NonNullable,
        )
    );

    let scalar_end = fsl.scalar_at(999_999_999_999).unwrap();
    assert!(!scalar_end.is_null());
    assert_eq!(
        scalar_end,
        Scalar::fixed_size_list(
            Arc::new(PType::I64.into()),
            vec![],
            Nullability::NonNullable,
        )
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Nullable degenerate cases
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_fsl_size_0_length_0_nullable() {
    let len = 0;
    let list_size = 0;

    // FSL of size 0 with length 0, nullable validity.
    let elements = PrimitiveArray::empty::<i8>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::AllValid, len);

    assert_eq!(fsl.len(), 0);
    assert_eq!(fsl.list_size(), 0);
    assert_eq!(fsl.elements().len(), 0);

    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(_, 0, Nullability::Nullable)
    ));
}

#[test]
fn test_fsl_size_0_length_1_nullable_valid() {
    let len = 1;
    let list_size = 0;

    // FSL of size 0 with length 1 (one empty list), nullable but valid.
    let elements = PrimitiveArray::empty::<u16>(Nullability::NonNullable);
    let validity = Validity::from_iter([true]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, validity, len);

    assert_eq!(fsl.len(), 1);
    assert_eq!(fsl.list_size(), 0);
    assert_eq!(fsl.elements().len(), 0);

    // Get the single empty list (should be valid).
    let scalar = fsl.scalar_at(0).unwrap();
    assert!(!scalar.is_null());
    assert_eq!(
        scalar,
        Scalar::fixed_size_list(Arc::new(PType::U16.into()), vec![], Nullability::Nullable,)
    );
}

#[test]
fn test_fsl_size_0_length_1_nullable_null() {
    let len = 1;
    let list_size = 0;

    // FSL of size 0 with length 1, but the list is null.
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let validity = Validity::from_iter([false]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, validity, len);

    assert_eq!(fsl.len(), 1);
    assert_eq!(fsl.list_size(), 0);
    assert_eq!(fsl.elements().len(), 0);

    // The single list should be null.
    let scalar = fsl.scalar_at(0).unwrap();
    assert!(scalar.is_null());
}

#[test]
fn test_fsl_size_0_length_10_nullable_mixed() {
    let len = 10;
    let list_size = 0;

    // FSL of size 0 with length 10, with mixed null/valid empty lists.
    let elements = PrimitiveArray::empty::<f32>(Nullability::NonNullable);
    let validity = Validity::from_iter([
        true, false, true, true, false, false, true, false, true, true,
    ]);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, validity, len);

    assert_eq!(fsl.len(), 10);
    assert_eq!(fsl.list_size(), 0);
    assert_eq!(fsl.elements().len(), 0);

    // Check validity pattern.
    let expected_valid = [
        true, false, true, true, false, false, true, false, true, true,
    ];
    for i in 0..len {
        let scalar = fsl.scalar_at(i).unwrap();
        if expected_valid[i] {
            assert!(!scalar.is_null());
            assert_eq!(
                scalar,
                Scalar::fixed_size_list(Arc::new(PType::F32.into()), vec![], Nullability::Nullable,)
            );
        } else {
            assert!(scalar.is_null());
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Additional edge cases with nullable elements
////////////////////////////////////////////////////////////////////////////////////////////////////

#[test]
fn test_fsl_size_0_nullable_elements() {
    let len = 5;
    let list_size = 0;

    // FSL of size 0 where the elements array itself has nullable dtype.
    let elements = PrimitiveArray::empty::<i32>(Nullability::Nullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    assert_eq!(fsl.len(), 5);
    assert_eq!(fsl.list_size(), 0);
    assert_eq!(fsl.elements().len(), 0);

    // Check that dtype shows nullable elements.
    assert!(matches!(
        fsl.dtype(),
        DType::FixedSizeList(elem_dtype, 0, Nullability::NonNullable)
            if matches!(elem_dtype.as_ref(), DType::Primitive(PType::I32, Nullability::Nullable))
    ));

    // All lists should be empty but valid.
    for i in 0..len {
        let scalar = fsl.scalar_at(i).unwrap();
        assert!(!scalar.is_null());
    }
}

#[test]
fn test_fsl_large_size_length_0() {
    let len = 0;
    let list_size = 1_000_000_000;

    // FSL with very large list size but zero length.
    let elements = PrimitiveArray::empty::<f64>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), list_size, Validity::NonNullable, len);

    assert_eq!(fsl.len(), 0);
    assert_eq!(fsl.list_size(), 1_000_000_000);
    assert_eq!(fsl.elements().len(), 0);
}

#[test]
fn test_fsl_size_0_validation() {
    // Test that validation works correctly for size 0 FSL.
    let list_size = 0;

    // Should succeed: empty elements array with any length is valid.
    {
        let len = 42;
        let elements = PrimitiveArray::empty::<u32>(Nullability::NonNullable);
        let result = FixedSizeListArray::try_new(
            elements.into_array(),
            list_size,
            Validity::NonNullable,
            len,
        );
        assert!(result.is_ok());
    }

    // Should fail: non-empty elements array with list_size = 0.
    {
        let len = 1;
        let elements = buffer![1i32].into_array();
        let result = FixedSizeListArray::try_new(
            elements.into_array(),
            list_size,
            Validity::NonNullable,
            len,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(
            "a degenerate (`list_size == 0`) `FixedSizeList` should have no underlying elements"
        ));
    }

    // Should succeed: validity length matches array length.
    {
        let len = 5;
        let elements = PrimitiveArray::empty::<i64>(Nullability::NonNullable);
        let validity = Validity::from_iter([true, false, true, false, true]);
        let result = FixedSizeListArray::try_new(elements.into_array(), list_size, validity, len);
        assert!(result.is_ok());
    }

    // Should fail: validity length doesn't match array length.
    {
        let len = 5;
        let elements = PrimitiveArray::empty::<i64>(Nullability::NonNullable);
        let validity = Validity::from_iter([true, false]); // Wrong length.
        let result = FixedSizeListArray::try_new(elements.into_array(), list_size, validity, len);
        assert!(result.is_err());
    }
}
