// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DType, Nullability, PType};
use vortex_scalar::Scalar;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable, PrimitiveArray};
use crate::builders::{ArrayBuilder, FixedSizeListBuilder};
use crate::compute::take;
use crate::validity::Validity;
use crate::{Array, IntoArray};

#[test]
fn test_take_basic() {
    // Create a FSL array with 3 lists, each containing 2 elements.
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Take indices [2, 0, 1].
    let indices = PrimitiveArray::from_iter([2u32, 0, 1]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 3);
    assert_eq!(result_fsl.list_size(), 2);

    // First list should be the original third list [5, 6].
    let first = result_fsl.fixed_size_list_at(0);
    assert_eq!(first.scalar_at(0), 5i32.into());
    assert_eq!(first.scalar_at(1), 6i32.into());

    // Second list should be the original first list [1, 2].
    let second = result_fsl.fixed_size_list_at(1);
    assert_eq!(second.scalar_at(0), 1i32.into());
    assert_eq!(second.scalar_at(1), 2i32.into());

    // Third list should be the original second list [3, 4].
    let third = result_fsl.fixed_size_list_at(2);
    assert_eq!(third.scalar_at(0), 3i32.into());
    assert_eq!(third.scalar_at(1), 4i32.into());
}

#[test]
fn test_take_with_duplicates() {
    // Create a FSL array with 2 lists, each containing 3 elements.
    let elements = PrimitiveArray::from_iter([1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 3, Validity::NonNullable, 2);

    // Take indices [0, 0, 1, 1, 0] - duplicating lists.
    let indices = PrimitiveArray::from_iter([0i32, 0, 1, 1, 0]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 5);
    assert_eq!(result_fsl.list_size(), 3);

    // Verify the lists are correctly duplicated.
    assert_eq!(result_fsl.scalar_at(0), fsl.scalar_at(0));
    assert_eq!(result_fsl.scalar_at(1), fsl.scalar_at(0));
    assert_eq!(result_fsl.scalar_at(2), fsl.scalar_at(1));
    assert_eq!(result_fsl.scalar_at(3), fsl.scalar_at(1));
    assert_eq!(result_fsl.scalar_at(4), fsl.scalar_at(0));
}

#[test]
fn test_take_empty_indices() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 2);

    // Take with empty indices.
    let indices = PrimitiveArray::from_iter(Vec::<u32>::new());
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 0);
    assert_eq!(result_fsl.list_size(), 2);
    assert_eq!(result_fsl.elements().len(), 0);
}

#[test]
fn test_take_single_index() {
    let elements = PrimitiveArray::from_iter([10i64, 20, 30, 40, 50, 60]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Take a single index [1].
    let indices = PrimitiveArray::from_iter([1u64]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 1);
    assert_eq!(result_fsl.list_size(), 2);

    let list = result_fsl.fixed_size_list_at(0);
    assert_eq!(list.scalar_at(0), 30i64.into());
    assert_eq!(list.scalar_at(1), 40i64.into());
}

#[test]
fn test_take_with_null_indices() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Create indices with nulls: [1, null, 0].
    let indices = PrimitiveArray::from_option_iter([Some(1u32), None, Some(0)]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 3);
    assert_eq!(result_fsl.list_size(), 2);

    // First list should be [3, 4].
    assert!(!result_fsl.scalar_at(0).is_null());
    let first = result_fsl.fixed_size_list_at(0);
    assert_eq!(first.scalar_at(0), 3i32.into());
    assert_eq!(first.scalar_at(1), 4i32.into());

    // Second list should be null.
    assert!(result_fsl.scalar_at(1).is_null());

    // Third list should be [1, 2].
    assert!(!result_fsl.scalar_at(2).is_null());
    let third = result_fsl.fixed_size_list_at(2);
    assert_eq!(third.scalar_at(0), 1i32.into());
    assert_eq!(third.scalar_at(1), 2i32.into());
}

#[test]
fn test_take_nullable_array() {
    // Create a nullable FSL array where the second list is null.
    let mut builder = FixedSizeListBuilder::with_capacity(
        DType::Primitive(PType::I32, Nullability::NonNullable).into(),
        2,
        Nullability::Nullable,
        3,
    );

    builder
        .append_value(
            Scalar::list(
                DType::Primitive(PType::I32, Nullability::NonNullable),
                vec![1i32.into(), 2i32.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .unwrap();
    builder.append_null();
    builder
        .append_value(
            Scalar::list(
                DType::Primitive(PType::I32, Nullability::NonNullable),
                vec![5i32.into(), 6i32.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .unwrap();

    let fsl = builder.finish();

    // Take indices [2, 1, 0].
    let indices = PrimitiveArray::from_iter([2u32, 1, 0]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 3);

    // First result should be the third list [5, 6].
    assert!(!result_fsl.scalar_at(0).is_null());

    // Second result should be null (original second list).
    assert!(result_fsl.scalar_at(1).is_null());

    // Third result should be the first list [1, 2].
    assert!(!result_fsl.scalar_at(2).is_null());
}

#[test]
fn test_take_nullable_array_with_null_indices() {
    // Create a nullable FSL array.
    let mut builder = FixedSizeListBuilder::with_capacity(
        DType::Primitive(PType::I32, Nullability::NonNullable).into(),
        2,
        Nullability::Nullable,
        3,
    );

    builder
        .append_value(
            Scalar::list(
                DType::Primitive(PType::I32, Nullability::NonNullable),
                vec![1i32.into(), 2i32.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .unwrap();
    builder.append_null();
    builder
        .append_value(
            Scalar::list(
                DType::Primitive(PType::I32, Nullability::NonNullable),
                vec![5i32.into(), 6i32.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .unwrap();

    let fsl = builder.finish();

    // Create indices with nulls: [0, null, 1, 2].
    let indices = PrimitiveArray::from_option_iter([Some(0u32), None, Some(1), Some(2)]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 4);

    // First result should be [1, 2].
    assert!(!result_fsl.scalar_at(0).is_null());

    // Second result should be null (null index).
    assert!(result_fsl.scalar_at(1).is_null());

    // Third result should be null (array's second element is null).
    assert!(result_fsl.scalar_at(2).is_null());

    // Fourth result should be [5, 6].
    assert!(!result_fsl.scalar_at(3).is_null());
}

#[test]
fn test_take_degenerate_list() {
    // Create a degenerate FSL array with list_size = 0.
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), 0, Validity::NonNullable, 5);

    // Take indices [3, 1, 4, 0, 2].
    let indices = PrimitiveArray::from_iter([3u32, 1, 4, 0, 2]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 5);
    assert_eq!(result_fsl.list_size(), 0);
    assert_eq!(result_fsl.elements().len(), 0);
}

#[test]
fn test_take_degenerate_list_with_nulls() {
    // Create a nullable degenerate FSL array where some lists are null.
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let validity = Validity::from_iter([true, false, true, true, false]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 0, validity, 5);

    // Take indices [1, 3, null, 0].
    let indices = PrimitiveArray::from_option_iter([Some(1u32), Some(3), None, Some(0)]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 4);
    assert_eq!(result_fsl.list_size(), 0);
    assert_eq!(result_fsl.elements().len(), 0);

    // First result should be null (index 1 is null in original).
    assert!(result_fsl.scalar_at(0).is_null());

    // Second result should not be null (index 3 is valid in original).
    assert!(!result_fsl.scalar_at(1).is_null());

    // Third result should be null (null index).
    assert!(result_fsl.scalar_at(2).is_null());

    // Fourth result should not be null (index 0 is valid in original).
    assert!(!result_fsl.scalar_at(3).is_null());
}

#[test]
fn test_take_large_list_size() {
    // Create a FSL array with large list size.
    let elements = PrimitiveArray::from_iter(0i32..30);
    let fsl = FixedSizeListArray::new(elements.into_array(), 10, Validity::NonNullable, 3);

    // Take indices [2, 0].
    let indices = PrimitiveArray::from_iter([2u16, 0]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 2);
    assert_eq!(result_fsl.list_size(), 10);

    // First list should be [20..30].
    let first = result_fsl.fixed_size_list_at(0);
    for i in 0..10i32 {
        assert_eq!(first.scalar_at(i as usize), (20 + i).into());
    }

    // Second list should be [0..10].
    let second = result_fsl.fixed_size_list_at(1);
    for i in 0..10i32 {
        assert_eq!(second.scalar_at(i as usize), i.into());
    }
}

#[test]
fn test_take_all_indices() {
    // Create a FSL array.
    let elements = PrimitiveArray::from_iter([1u8, 2, 3, 4, 5, 6, 7, 8]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 4);

    // Take all indices in order [0, 1, 2, 3].
    let indices = PrimitiveArray::from_iter([0i64, 1, 2, 3]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 4);
    assert_eq!(result_fsl.list_size(), 2);

    // Verify all lists are the same as the original.
    for i in 0..4 {
        assert_eq!(result_fsl.scalar_at(i), fsl.scalar_at(i));
    }
}

#[test]
fn test_take_reverse_order() {
    // Create a FSL array.
    let elements = PrimitiveArray::from_iter([100i16, 200, 300, 400, 500, 600]);
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Take indices in reverse order [2, 1, 0].
    let indices = PrimitiveArray::from_iter([2u32, 1, 0]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 3);

    // Verify the lists are in reverse order.
    assert_eq!(result_fsl.scalar_at(0), fsl.scalar_at(2));
    assert_eq!(result_fsl.scalar_at(1), fsl.scalar_at(1));
    assert_eq!(result_fsl.scalar_at(2), fsl.scalar_at(0));
}
