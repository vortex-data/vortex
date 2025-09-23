// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::{Buffer, buffer};
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
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Take indices [2, 0, 1].
    let indices = buffer![2u32, 0, 1].into_array();
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 3);
    assert_eq!(result_fsl.list_size(), 2);

    // First list should be the original third list [5, 6].
    let first = result_fsl.fixed_size_list_elements_at(0);
    assert_eq!(first.scalar_at(0), 5i32.into());
    assert_eq!(first.scalar_at(1), 6i32.into());

    // Second list should be the original first list [1, 2].
    let second = result_fsl.fixed_size_list_elements_at(1);
    assert_eq!(second.scalar_at(0), 1i32.into());
    assert_eq!(second.scalar_at(1), 2i32.into());

    // Third list should be the original second list [3, 4].
    let third = result_fsl.fixed_size_list_elements_at(2);
    assert_eq!(third.scalar_at(0), 3i32.into());
    assert_eq!(third.scalar_at(1), 4i32.into());
}

#[test]
fn test_take_with_duplicates() {
    // Create a FSL array with 2 lists, each containing 3 elements.
    let elements = buffer![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0].into_array();
    let fsl = FixedSizeListArray::new(elements, 3, Validity::NonNullable, 2);

    // Take indices [0, 0, 1, 1, 0] - duplicating lists.
    let indices = buffer![0i32, 0, 1, 1, 0].into_array();
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
    let elements = buffer![1i32, 2, 3, 4].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 2);

    // Take with empty indices.
    let indices = Buffer::<u32>::empty().into_array();
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 0);
    assert_eq!(result_fsl.list_size(), 2);
    assert_eq!(result_fsl.elements().len(), 0);
}

#[test]
fn test_take_single_index() {
    let elements = buffer![10i64, 20, 30, 40, 50, 60].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Take a single index [1].
    let indices = buffer![1u64].into_array();
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 1);
    assert_eq!(result_fsl.list_size(), 2);

    let list = result_fsl.fixed_size_list_elements_at(0);
    assert_eq!(list.scalar_at(0), 30i64.into());
    assert_eq!(list.scalar_at(1), 40i64.into());
}

#[test]
fn test_take_with_null_indices() {
    let elements = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Create indices with nulls: [1, null, 0].
    let indices = PrimitiveArray::from_option_iter([Some(1u32), None, Some(0)]);
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 3);
    assert_eq!(result_fsl.list_size(), 2);

    // First list should be [3, 4].
    assert!(!result_fsl.scalar_at(0).is_null());
    let first = result_fsl.fixed_size_list_elements_at(0);
    assert_eq!(first.scalar_at(0), 3i32.into());
    assert_eq!(first.scalar_at(1), 4i32.into());

    // Second list should be null.
    assert!(result_fsl.scalar_at(1).is_null());

    // Third list should be [1, 2].
    assert!(!result_fsl.scalar_at(2).is_null());
    let third = result_fsl.fixed_size_list_elements_at(2);
    assert_eq!(third.scalar_at(0), 1i32.into());
    assert_eq!(third.scalar_at(1), 2i32.into());
}

// Parameterized test for nullable array scenarios.
#[rstest]
#[case::nullable_array_basic(
    vec![Some(vec![1i32, 2]), None, Some(vec![5, 6])],
    vec![Some(2u32), Some(1), Some(0)],
    vec![false, true, false] // Expected nulls
)]
#[case::nullable_array_with_null_indices(
    vec![Some(vec![1i32, 2]), None, Some(vec![5, 6])],
    vec![Some(0u32), None, Some(1), Some(2)],
    vec![false, true, true, false] // Expected nulls
)]
fn test_take_nullable_arrays(
    #[case] array_values: Vec<Option<Vec<i32>>>,
    #[case] indices: Vec<Option<u32>>,
    #[case] expected_nulls: Vec<bool>,
) {
    // Build the nullable FSL array.
    let list_size = if let Some(Some(first)) = array_values.first() {
        u32::try_from(first.len()).unwrap()
    } else {
        2 // Default size
    };

    let mut builder = FixedSizeListBuilder::with_capacity(
        DType::Primitive(PType::I32, Nullability::NonNullable).into(),
        list_size,
        Nullability::Nullable,
        array_values.len(),
    );

    for value in array_values {
        match value {
            Some(list) => {
                let scalars: Vec<Scalar> = list.into_iter().map(|v| v.into()).collect();
                builder
                    .append_value(
                        Scalar::list(
                            DType::Primitive(PType::I32, Nullability::NonNullable),
                            scalars,
                            Nullability::NonNullable,
                        )
                        .as_list(),
                    )
                    .unwrap();
            }
            None => builder.append_null(),
        }
    }

    let fsl = builder.finish();

    // Create indices (with possible nulls).
    let indices_array = PrimitiveArray::from_option_iter(indices.clone());
    let result = take(fsl.as_ref(), indices_array.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), indices.len());

    // Check nullability of results.
    for (i, expected_null) in expected_nulls.iter().enumerate() {
        assert_eq!(result_fsl.scalar_at(i).is_null(), *expected_null);
    }
}

// Parameterized test for degenerate (list_size=0) cases.
#[rstest]
#[case::basic_degenerate(
    Validity::NonNullable,
    vec![Some(3u32), Some(1), Some(4), Some(0), Some(2)],
    5,
    vec![false; 5]
)]
#[case::degenerate_with_nulls(
    Validity::from_iter([true, false, true, true, false]),
    vec![Some(1u32), Some(3), None, Some(0)],
    4,
    vec![true, false, true, false] // Expected nulls based on indices
)]
fn test_take_degenerate_lists(
    #[case] validity: Validity,
    #[case] indices: Vec<Option<u32>>,
    #[case] expected_len: usize,
    #[case] expected_nulls: Vec<bool>,
) {
    // Create a degenerate FSL array with list_size = 0.
    let elements = PrimitiveArray::empty::<i32>(Nullability::NonNullable);
    let fsl = FixedSizeListArray::new(elements.into_array(), 0, validity, 5);

    let indices_array = PrimitiveArray::from_option_iter(indices);
    let result = take(fsl.as_ref(), indices_array.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), expected_len);
    assert_eq!(result_fsl.list_size(), 0);
    assert_eq!(result_fsl.elements().len(), 0);

    // Check nullability of results.
    for (i, expected_null) in expected_nulls.iter().enumerate() {
        assert_eq!(result_fsl.scalar_at(i).is_null(), *expected_null);
    }
}

#[test]
fn test_take_large_list_size() {
    // Create a FSL array with large list size.
    let elements = buffer![0i32..30].into_array();
    let fsl = FixedSizeListArray::new(elements, 10, Validity::NonNullable, 3);

    // Take indices [2, 0].
    let indices = buffer![2u16, 0].into_array();
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 2);
    assert_eq!(result_fsl.list_size(), 10);

    // First list should be [20..30].
    let first = result_fsl.fixed_size_list_elements_at(0);
    for i in 0..10i32 {
        assert_eq!(first.scalar_at(i as usize), (20 + i).into());
    }

    // Second list should be [0..10].
    let second = result_fsl.fixed_size_list_elements_at(1);
    for i in 0..10i32 {
        assert_eq!(second.scalar_at(i as usize), i.into());
    }
}

#[test]
fn test_take_all_indices() {
    // Create a FSL array.
    let elements = buffer![1u8, 2, 3, 4, 5, 6, 7, 8].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 4);

    // Take all indices in order [0, 1, 2, 3].
    let indices = buffer![0i64, 1, 2, 3].into_array();
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
    let elements = buffer![100i16, 200, 300, 400, 500, 600].into_array();
    let fsl = FixedSizeListArray::new(elements.into_array(), 2, Validity::NonNullable, 3);

    // Take indices in reverse order [2, 1, 0].
    let indices = buffer![2u32, 1, 0].into_array();
    let result = take(fsl.as_ref(), indices.as_ref()).unwrap();
    let result_fsl = result.as_::<FixedSizeListVTable>();

    assert_eq!(result_fsl.len(), 3);

    // Verify the lists are in reverse order.
    assert_eq!(result_fsl.scalar_at(0), fsl.scalar_at(2));
    assert_eq!(result_fsl.scalar_at(1), fsl.scalar_at(1));
    assert_eq!(result_fsl.scalar_at(2), fsl.scalar_at(0));
}
