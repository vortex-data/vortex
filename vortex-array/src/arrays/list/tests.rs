// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::*;
use crate::Canonical;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::FilterArray;
use crate::arrays::List;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builders::ArrayBuilder;
use crate::builders::ListBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType::I32;
use crate::scalar::Scalar;
use crate::validity::Validity;

#[test]
fn test_empty_list_array() {
    let elements = PrimitiveArray::empty::<u32>(Nullability::NonNullable);
    let offsets = buffer![0].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets, validity).unwrap();

    assert_eq!(0, list.len());
}

#[test]
fn test_simple_list_array() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0, 2, 4, 5].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements, offsets, validity).unwrap();

    assert_eq!(
        Scalar::list(
            Arc::new(I32.into()),
            vec![1.into(), 2.into()],
            Nullability::Nullable
        ),
        list.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert_eq!(
        Scalar::list(
            Arc::new(I32.into()),
            vec![3.into(), 4.into()],
            Nullability::Nullable
        ),
        list.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert_eq!(
        Scalar::list(Arc::new(I32.into()), vec![5.into()], Nullability::Nullable),
        list.execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
}

#[test]
fn test_simple_list_array_from_iter() {
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = buffer![0, 2, 3].into_array();
    let validity = Validity::NonNullable;

    let list = ListArray::try_new(elements, offsets, validity).unwrap();

    let list_from_iter =
        ListArray::from_iter_slow::<u32, _>(vec![vec![1i32, 2], vec![3]], Arc::new(I32.into()))
            .unwrap();

    assert_eq!(list.len(), list_from_iter.len());
    assert_eq!(
        list.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        list_from_iter
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
    assert_eq!(
        list.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        list_from_iter
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
}

#[test]
fn test_simple_list_filter() {
    let elements = PrimitiveArray::from_option_iter([None, Some(2), Some(3), Some(4), Some(5)]);
    let offsets = buffer![0, 2, 4, 5].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets, validity)
        .unwrap()
        .into_array();

    let filtered = list.filter(Mask::from(BitBuffer::from(vec![false, true, true])));

    assert!(filtered.is_ok())
}

#[test]
fn test_list_filter_dense_mask() {
    // Test filtering with a dense mask (high density of true values).
    let elements = buffer![0..100].into_array();
    let offsets = buffer![0, 10, 25, 40, 60, 85, 100].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements, offsets, validity)
        .unwrap()
        .into_array();

    // Dense mask: keep most elements (indices 1, 2, 3, 4, 5).
    let mask = Mask::from(BitBuffer::from(vec![false, true, true, true, true, true]));

    let filtered = list.filter(mask).unwrap();

    // Should have 5 lists remaining.
    assert_eq!(filtered.len(), 5);

    // Construct expected ListArray with elements from indices 1, 2, 3, 4, 5.
    // [10..25], [25..40], [40..60], [60..85], [85..100].
    let expected = ListArray::try_new(
        buffer![10..100].into_array(),
        buffer![0u32, 15, 30, 50, 75, 90].into_array(),
        Validity::AllValid,
    )
    .unwrap();

    assert_arrays_eq!(filtered, expected);
}

#[test]
fn test_list_filter_sparse_mask() {
    // Test filtering with a sparse mask (low density of true values).
    let elements = buffer![0..100].into_array();
    let offsets = buffer![0, 10, 25, 40, 60, 85, 100].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements, offsets, validity)
        .unwrap()
        .into_array();

    // Sparse mask: keep only a few elements (indices 0 and 5).
    let mask = Mask::from(BitBuffer::from(vec![
        true, false, false, false, false, true,
    ]));

    let filtered = list.filter(mask).unwrap();

    // Should have 2 lists remaining.
    assert_eq!(filtered.len(), 2);

    // Construct expected: lists at indices 0 ([0..10]) and 5 ([85..100]).
    let expected_elements: Vec<i32> = (0..10).chain(85..100).collect();
    let expected = ListArray::try_new(
        PrimitiveArray::from_iter(expected_elements).into_array(),
        buffer![0u32, 10, 25].into_array(),
        Validity::AllValid,
    )
    .unwrap();

    assert_arrays_eq!(filtered, expected);
}

#[test]
fn test_list_filter_empty_lists() {
    // Test filtering arrays that contain empty lists.
    let elements = buffer![0..10].into_array();
    let offsets = buffer![0, 0, 3, 3, 7, 10, 10].into_array(); // Lists at indices 0, 2, 5 are empty.
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements, offsets, validity)
        .unwrap()
        .into_array();

    let mask = Mask::from(BitBuffer::from(vec![true, true, true, false, false, true]));

    let filtered = list.filter(mask).unwrap();

    assert_eq!(filtered.len(), 4);

    // Construct expected: keep indices 0 (empty), 1 ([0..3]), 2 (empty), 5 (empty).
    let expected = ListArray::try_new(
        buffer![0..3].into_array(),
        buffer![0u32, 0, 3, 3, 3].into_array(), // empty, [0..3], empty, empty.
        Validity::AllValid,
    )
    .unwrap();

    assert_arrays_eq!(filtered, expected);
}

#[test]
fn test_list_filter_with_nulls() {
    // Test filtering lists with null validity.
    let elements = buffer![0..15].into_array();
    let offsets = buffer![0, 3, 7, 10, 12, 15].into_array();
    let validity = Validity::from_mask(
        Mask::from(BitBuffer::from(vec![true, false, true, false, true])),
        Nullability::Nullable,
    );

    let list = ListArray::try_new(elements, offsets, validity)
        .unwrap()
        .into_array();

    let mask = Mask::from(BitBuffer::from(vec![true, true, false, true, true]));

    let filtered = list.filter(mask).unwrap();

    assert_eq!(filtered.len(), 4);

    // Check validity of filtered array using scalar_at (works on any array).
    assert!(
        filtered
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_valid()
    );
    assert!(
        !filtered
            .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_valid()
    ); // Was null.
    assert!(
        !filtered
            .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_valid()
    ); // Was null.
    assert!(
        filtered
            .execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
            .is_valid()
    );
}

#[test]
fn test_list_filter_all_true() {
    // Test filtering with an all-true mask.
    let elements = buffer![0..20].into_array();
    let offsets = buffer![0, 5, 10, 15, 20].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.clone(), offsets.clone(), validity.clone())
        .unwrap()
        .into_array();

    let mask = Mask::AllTrue(4);

    let filtered = list.filter(mask).unwrap();

    // All lists should be preserved.
    assert_eq!(filtered.len(), 4);

    let expected = ListArray::try_new(elements, offsets, validity).unwrap();
    assert_arrays_eq!(filtered, expected);
}

#[test]
fn test_list_filter_all_false() {
    // Test filtering with an all-false mask.
    let elements = buffer![0..20].into_array();
    let offsets = buffer![0, 5, 10, 15, 20].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements, offsets, validity)
        .unwrap()
        .into_array();

    let mask = Mask::AllFalse(4);

    let filtered = list.filter(mask).unwrap();

    // When mask is AllFalse, filter returns a canonical empty array (ListViewArray).
    // We need to check the length directly without casting to a specific type.
    assert_eq!(filtered.len(), 0);
}

#[test]
fn test_list_filter_single_element() {
    // Test filtering to keep only one element.
    let elements = buffer![0..50].into_array();
    let offsets = buffer![0, 10, 20, 30, 40, 50].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements, offsets, validity)
        .unwrap()
        .into_array();

    let mask = Mask::from(BitBuffer::from(vec![false, false, true, false, false]));

    let filtered = list.filter(mask).unwrap();

    assert_eq!(filtered.len(), 1);

    // Construct expected: single list with elements [20..30].
    let expected = ListArray::try_new(
        buffer![20..30].into_array(),
        buffer![0u32, 10].into_array(),
        Validity::AllValid,
    )
    .unwrap();

    assert_arrays_eq!(filtered, expected);
}

#[test]
fn test_list_filter_alternating_pattern() {
    // Test filtering with an alternating pattern.
    let elements = buffer![0..60].into_array();
    let offsets = buffer![0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements, offsets, validity)
        .unwrap()
        .into_array();

    // Keep every other list.
    let mask = Mask::from(BitBuffer::from(vec![
        true, false, true, false, true, false, true, false, true, false, true, false,
    ]));

    let filtered = list.filter(mask).unwrap();

    assert_eq!(filtered.len(), 6);

    // Construct expected: [0..5], [10..15], [20..25], [30..35], [40..45], [50..55].
    let expected_elements: Vec<i32> = [0, 10, 20, 30, 40, 50]
        .iter()
        .flat_map(|&start| start..start + 5)
        .collect();
    let expected = ListArray::try_new(
        PrimitiveArray::from_iter(expected_elements).into_array(),
        buffer![0u32, 5, 10, 15, 20, 25, 30].into_array(),
        Validity::AllValid,
    )
    .unwrap();

    assert_arrays_eq!(filtered, expected);
}

#[test]
fn test_list_filter_variable_sizes() {
    // Test filtering lists with highly variable sizes.
    let elements = buffer![0..100].into_array();
    let offsets = buffer![0, 1, 2, 5, 10, 20, 35, 60, 100].into_array();
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements, offsets, validity)
        .unwrap()
        .into_array();

    let mask = Mask::from(BitBuffer::from(vec![
        true, false, true, true, false, true, true, true,
    ]));

    let filtered = list.filter(mask).unwrap();

    assert_eq!(filtered.len(), 6);

    // Construct expected: indices 0, 2, 3, 5, 6, 7.
    // Sizes: 1, 3, 5, 15, 25, 40.
    // Elements: [0], [2..5], [5..10], [20..35], [35..60], [60..100].
    let expected_elements: Vec<i32> = vec![0]
        .into_iter()
        .chain(2..5)
        .chain(5..10)
        .chain(20..35)
        .chain(35..60)
        .chain(60..100)
        .collect();
    let expected = ListArray::try_new(
        PrimitiveArray::from_iter(expected_elements).into_array(),
        buffer![0u32, 1, 4, 9, 24, 49, 89].into_array(),
        Validity::AllValid,
    )
    .unwrap();

    assert_arrays_eq!(filtered, expected);
}

#[test]
fn test_offset_to_0() {
    let mut builder =
        ListBuilder::<u32>::with_capacity(Arc::new(I32.into()), Nullability::NonNullable, 10, 5);
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![1.into(), 2.into(), 3.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_expect("operation should succeed in test");
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![4.into(), 5.into(), 6.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_expect("operation should succeed in test");
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![7.into(), 8.into(), 9.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_expect("operation should succeed in test");
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![10.into(), 11.into(), 12.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_expect("operation should succeed in test");
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![13.into(), 14.into(), 15.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_expect("operation should succeed in test");
    let list = builder.finish().slice(2..4).unwrap();

    // The sliced list should be a ListArray since we built it with ListBuilder
    // and slice doesn't change the encoding
    assert_eq!(list.len(), 2);

    // For a sliced ListArray, we need to check it's still a ListArray
    let list_array = list.as_::<List>();

    // Check the offsets array has correct length (n+1 for n lists)
    assert_eq!(list_array.offsets().len(), 3);

    // Each list has 3 elements
    assert_eq!(list_array.list_elements_at(0).unwrap().len(), 3);
    assert_eq!(list_array.list_elements_at(1).unwrap().len(), 3);
}

type OptVec<T> = Vec<Option<T>>;

// Helper function to create a list of lists from a 3D vector with Option types.
#[expect(clippy::cast_possible_truncation)]
fn create_list_of_lists_nullable(data: OptVec<OptVec<OptVec<i32>>>) -> ListArray {
    // Flatten all elements and track offsets and validity.
    let mut all_elements = Vec::new();
    let mut element_validity = Vec::new();
    let mut inner_offsets = vec![0u32];
    let mut inner_validity = Vec::new();
    let mut outer_offsets = vec![0u32];
    let mut outer_validity = Vec::new();

    for outer_opt in &data {
        outer_validity.push(outer_opt.is_some());

        if let Some(outer_list) = outer_opt {
            for inner_opt in outer_list {
                inner_validity.push(inner_opt.is_some());

                if let Some(inner_list) = inner_opt {
                    for elem_opt in inner_list {
                        element_validity.push(elem_opt.is_some());
                        all_elements.push(elem_opt.unwrap_or(0));
                    }
                }
                inner_offsets.push(all_elements.len() as u32);
            }
        }
        outer_offsets.push(inner_offsets.len() as u32 - 1);
    }

    // Determine nullabilities based on presence of None values.
    let has_null_elements = element_validity.iter().any(|&v| !v);
    let has_null_inner = inner_validity.iter().any(|&v| !v);
    let has_null_outer = outer_validity.iter().any(|&v| !v);

    // Create the innermost i32 elements array.
    let i32_elements = if has_null_elements {
        PrimitiveArray::from_option_iter(
            all_elements
                .iter()
                .zip(&element_validity)
                .map(|(&val, &valid)| valid.then_some(val)),
        )
    } else {
        PrimitiveArray::from_iter(all_elements)
    };

    // Verify i32 elements have correct nullability.
    let expected_elem_nullability = if has_null_elements {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    assert_eq!(
        i32_elements.dtype().nullability(),
        expected_elem_nullability,
        "i32 elements array has incorrect nullability"
    );

    // Create inner validity if needed.
    let inner_list_validity = if has_null_inner {
        Validity::from_mask(
            Mask::from(BitBuffer::from(inner_validity)),
            Nullability::Nullable,
        )
    } else {
        Validity::NonNullable
    };

    // Create the inner list array (list of i32).
    let inner_lists = ListArray::try_new(
        i32_elements.into_array(),
        PrimitiveArray::from_iter(inner_offsets).into_array(),
        inner_list_validity,
    )
    .unwrap();

    // Verify inner list array has correct nullability.
    let expected_inner_nullability = if has_null_inner {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    assert_eq!(
        inner_lists.dtype().nullability(),
        expected_inner_nullability,
        "Inner list array has incorrect nullability"
    );

    // Create outer validity if needed.
    let outer_list_validity = if has_null_outer {
        Validity::from_mask(
            Mask::from(BitBuffer::from(outer_validity)),
            Nullability::Nullable,
        )
    } else {
        Validity::NonNullable
    };

    // Create the outer list array (list of lists).
    let list_of_lists = ListArray::try_new(
        inner_lists.into_array(),
        PrimitiveArray::from_iter(outer_offsets).into_array(),
        outer_list_validity,
    )
    .unwrap();

    // Verify outer list array has correct nullability.
    let expected_outer_nullability = if has_null_outer {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    assert_eq!(
        list_of_lists.dtype().nullability(),
        expected_outer_nullability,
        "Outer list array has incorrect nullability"
    );

    list_of_lists
}

#[test]
#[expect(clippy::cognitive_complexity)]
fn test_list_of_lists() {
    let data = vec![
        Some(vec![Some(vec![Some(1), Some(2)]), Some(vec![Some(3)])]),
        Some(vec![Some(vec![Some(4), Some(5), Some(6)])]),
        Some(vec![]),
        Some(vec![Some(vec![Some(7)])]),
    ];

    let list_of_lists = create_list_of_lists_nullable(data);

    // Verify the structure.
    assert_eq!(list_of_lists.len(), 4);

    // Check the dtype is List<List<i32>>.
    assert!(matches!(
        list_of_lists.dtype(),
        DType::List(inner_dtype, _)
            if matches!(
                inner_dtype.as_ref(),
                DType::List(elem_dtype, _)
                    if matches!(elem_dtype.as_ref(), DType::Primitive(I32, _))
            )
    ));

    // Access the first list of lists and verify its contents.
    let first_outer = list_of_lists.list_elements_at(0).unwrap();
    let first_outer_list = first_outer.as_::<List>();
    assert_eq!(first_outer_list.len(), 2);

    // Check first inner list [1, 2].
    let first_inner = first_outer_list.list_elements_at(0).unwrap();
    assert_arrays_eq!(first_inner, PrimitiveArray::from_iter([1, 2]));

    // Check second inner list [3].
    let second_inner = first_outer_list.list_elements_at(1).unwrap();
    assert_arrays_eq!(second_inner, PrimitiveArray::from_iter([3]));

    // Check the second list of lists [[4, 5, 6]].
    let second_outer = list_of_lists.list_elements_at(1).unwrap();
    let second_outer_list = second_outer.as_::<List>();
    assert_eq!(second_outer_list.len(), 1);

    let inner = second_outer_list.list_elements_at(0).unwrap();
    assert_arrays_eq!(inner, PrimitiveArray::from_iter([4, 5, 6]));

    // Check the third list of lists (empty).
    let third_outer = list_of_lists.list_elements_at(2).unwrap();
    // Empty slices return canonical form (`ListViewArray`), so we check length directly.
    assert_eq!(third_outer.len(), 0);

    // Check the fourth list of lists [[7]].
    let fourth_outer = list_of_lists.list_elements_at(3).unwrap();
    let fourth_outer_list = fourth_outer.as_::<List>();
    assert_eq!(fourth_outer_list.len(), 1);

    let inner = fourth_outer_list.list_elements_at(0).unwrap();
    assert_arrays_eq!(inner, PrimitiveArray::from_iter([7]));

    // Test scalar conversion.
    let scalar = list_of_lists
        .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(matches!(scalar.dtype(), DType::List(_, _)));
    let list_scalar = scalar.as_list();
    assert_eq!(list_scalar.len(), 2);

    // Test slicing.
    let sliced = list_of_lists.slice(1..3).unwrap();
    let sliced_list = sliced.as_::<List>();
    assert_eq!(sliced_list.len(), 2);

    // First element of slice should be [[4, 5, 6]].
    let first_sliced = sliced_list.list_elements_at(0).unwrap();
    let first_sliced_list = first_sliced.as_::<List>();
    assert_eq!(first_sliced_list.len(), 1);

    // Second element of slice should be empty [].
    let second_sliced = sliced_list.list_elements_at(1).unwrap();
    // Empty slices return canonical form (`ListViewArray`), so we check length directly
    assert_eq!(second_sliced.len(), 0);
}

#[test]
fn test_list_of_lists_nullable_outer() {
    // Create list of lists with nullable outer, non-nullable inner.
    // Structure: [[[1, 2], [3]], null, [[4, 5, 6]], [[7]]]
    let data = vec![
        Some(vec![Some(vec![Some(1), Some(2)]), Some(vec![Some(3)])]),
        None,
        Some(vec![Some(vec![Some(4), Some(5), Some(6)])]),
        Some(vec![Some(vec![Some(7)])]),
    ];

    let list_of_lists = create_list_of_lists_nullable(data);

    // Verify structure.
    assert_eq!(list_of_lists.len(), 4);

    // Check dtype - outer is nullable, inner is not.
    assert!(matches!(
        list_of_lists.dtype(),
        DType::List(inner_dtype, Nullability::Nullable)
            if matches!(inner_dtype.as_ref(), DType::List(_, Nullability::NonNullable))
    ));

    // First element should be [[1, 2], [3]].
    let first = list_of_lists
        .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(!first.is_null());

    // Second element should be null.
    let second = list_of_lists
        .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(second.is_null());

    // Third element should be [[4, 5, 6]].
    let third = list_of_lists.list_elements_at(2).unwrap();
    let third_list = third.as_::<List>();
    assert_eq!(third_list.len(), 1);
    let inner = third_list.list_elements_at(0).unwrap();
    assert_eq!(inner.len(), 3);

    // Fourth element should be [[7]].
    let fourth = list_of_lists.list_elements_at(3).unwrap();
    let fourth_list = fourth.as_::<List>();
    assert_eq!(fourth_list.len(), 1);
}

#[test]
fn test_list_of_lists_nullable_inner() {
    // Create list of lists with non-nullable outer, nullable inner.
    // Structure: [[[1, 2], null, [3]], [[4, 5, 6]], [], [[null, 7]]]
    let data = vec![
        Some(vec![
            Some(vec![Some(1), Some(2)]),
            None,
            Some(vec![Some(3)]),
        ]),
        Some(vec![Some(vec![Some(4), Some(5), Some(6)])]),
        Some(vec![]),
        Some(vec![Some(vec![None, Some(7)])]),
    ];

    let list_of_lists = create_list_of_lists_nullable(data);

    // Verify structure.
    assert_eq!(list_of_lists.len(), 4);

    // Check dtype - outer is non-nullable, inner is nullable.
    assert!(matches!(
        list_of_lists.dtype(),
        DType::List(inner_dtype, Nullability::NonNullable)
            if matches!(
                inner_dtype.as_ref(),
                DType::List(elem_dtype, Nullability::Nullable)
                    if matches!(elem_dtype.as_ref(), DType::Primitive(I32, Nullability::Nullable))
            )
    ));

    // First outer list should have 3 inner lists with the second being null.
    let first_outer = list_of_lists.list_elements_at(0).unwrap();
    let first_list = first_outer.as_::<List>();
    assert_eq!(first_list.len(), 3);

    // Check that second inner list is null.
    let second_inner = first_list
        .array()
        .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(second_inner.is_null());
}

#[test]
fn test_list_of_lists_both_nullable() {
    // Create list of lists with both nullable.
    // Structure: [[[1, 2], null], null, [[3]], [null]]
    let data = vec![
        Some(vec![Some(vec![Some(1), Some(2)]), None]),
        None,
        Some(vec![Some(vec![Some(3)])]),
        Some(vec![None]),
    ];

    let list_of_lists = create_list_of_lists_nullable(data);

    // Verify structure.
    assert_eq!(list_of_lists.len(), 4);

    // Check dtype - both nullable.
    assert!(matches!(
        list_of_lists.dtype(),
        DType::List(inner_dtype, Nullability::Nullable)
            if matches!(inner_dtype.as_ref(), DType::List(_, Nullability::Nullable))
    ));

    // First outer list should have 2 elements, second is null inner list.
    let first_outer = list_of_lists
        .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(!first_outer.is_null());
    let first_outer_array = list_of_lists.list_elements_at(0).unwrap();
    let first_list = first_outer_array.as_::<List>();
    assert_eq!(first_list.len(), 2);

    // First inner list should be [1, 2].
    let first_inner = first_list.list_elements_at(0).unwrap();
    assert_eq!(first_inner.len(), 2);

    // Second inner list should be null.
    let second_inner = first_list
        .array()
        .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(second_inner.is_null());

    // Second outer list should be null.
    let second_outer = list_of_lists
        .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(second_outer.is_null());

    // Third outer list should have [3].
    let third_outer = list_of_lists.list_elements_at(2).unwrap();
    let third_list = third_outer.as_::<List>();
    assert_eq!(third_list.len(), 1);
    let inner = third_list.list_elements_at(0).unwrap();
    assert_arrays_eq!(inner, PrimitiveArray::from_iter([3]));

    // Fourth outer list should have a null inner list.
    let fourth_outer = list_of_lists.list_elements_at(3).unwrap();
    let fourth_list = fourth_outer.as_::<List>();
    assert_eq!(fourth_list.len(), 1);
    let inner = fourth_list
        .array()
        .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();
    assert!(inner.is_null());
}

#[test]
#[should_panic(expected = "offsets minimum -1 outside valid range")]
fn test_negative_offset_values() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![-1i32, 2, 4, 5].into_array();
    let validity = Validity::AllValid;

    ListArray::try_new(elements, offsets, validity).unwrap();
}

#[test]
#[should_panic(expected = "offsets must be sorted")]
fn test_unsorted_offsets() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 3, 2, 5].into_array();
    let validity = Validity::AllValid;

    ListArray::try_new(elements, offsets, validity).unwrap();
}

#[test]
#[should_panic(expected = "Max offset 7 is beyond the length of the elements array 5")]
fn test_offset_exceeding_elements_length() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 2, 4, 7].into_array();
    let validity = Validity::AllValid;

    ListArray::try_new(elements, offsets, validity).unwrap();
}

#[test]
#[should_panic(expected = "validity with size 2 does not match array size 4")]
fn test_validity_length_mismatch() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0u32, 2, 4, 5, 5].into_array();
    let validity = Validity::from_mask(
        Mask::from(BitBuffer::from(vec![true, false])),
        Nullability::Nullable,
    );

    ListArray::try_new(elements, offsets, validity).unwrap();
}

#[test]
#[should_panic(expected = "offsets have invalid type")]
fn test_nullable_offsets() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = PrimitiveArray::from_option_iter([Some(0u32), Some(2), None, Some(5)]);
    let validity = Validity::AllValid;

    ListArray::try_new(elements, offsets.into_array(), validity).unwrap();
}

#[test]
#[should_panic(expected = "Offsets must have at least one element, [0] for an empty list")]
fn test_empty_offsets_array() {
    let elements = buffer![1i32, 2, 3].into_array();
    let offsets = PrimitiveArray::empty::<u32>(Nullability::NonNullable);
    let validity = Validity::AllValid;

    ListArray::try_new(elements, offsets.into_array(), validity).unwrap();
}

#[test]
#[should_panic(expected = "offsets have invalid type")]
fn test_non_integer_offsets() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![0.0f32, 2.0, 4.0, 5.0].into_array();
    let validity = Validity::AllValid;

    ListArray::try_new(elements, offsets, validity).unwrap();
}

#[test]
fn test_offsets_constant() {
    let elements = buffer![1i32, 2, 3, 4, 5].into_array();
    let offsets = buffer![5u32, 5, 5, 5].into_array();
    let validity = Validity::AllValid;

    // This should succeed as it represents empty lists
    let list = ListArray::try_new(elements, offsets, validity).unwrap();
    assert_eq!(list.len(), 3);
    assert_eq!(list.list_elements_at(0).unwrap().len(), 0);
    assert_eq!(list.list_elements_at(1).unwrap().len(), 0);
    assert_eq!(list.list_elements_at(2).unwrap().len(), 0);
}

#[test]
fn test_recursive_compact_list_of_lists() {
    // Create a nested list structure: [[[1,2,3], [4,5]], [[6,7,8,9]], [[10], [11,12]]]
    let nested_data = vec![
        Some(vec![
            Some(vec![Some(1), Some(2), Some(3)]),
            Some(vec![Some(4), Some(5)]),
        ]),
        Some(vec![Some(vec![Some(6), Some(7), Some(8), Some(9)])]),
        Some(vec![Some(vec![Some(10)]), Some(vec![Some(11), Some(12)])]),
    ];

    let original = create_list_of_lists_nullable(nested_data);
    // Slice to remove prefix - creates wasted space since offsets no longer reference early elements
    let sliced = original.slice(1..3).unwrap();
    let sliced_list = sliced.as_::<List>();

    // Test non-recursive compaction: only resets outer list offsets
    let non_recursive = sliced_list.reset_offsets(false).unwrap();
    // Test recursive compaction: resets offsets and compacts inner canonical arrays
    let recursive = sliced_list.reset_offsets(true).unwrap();

    assert_eq!(non_recursive.len(), 2);
    assert_eq!(recursive.len(), 2);

    // Check the flattened elements - this shows the actual compaction difference
    let non_recursive_inner = non_recursive.elements().as_::<List>();
    let non_recursive_flat_elements = non_recursive_inner.elements();
    let recursive_inner = recursive.elements().as_::<List>();
    let recursive_flat_elements = recursive_inner.elements();

    // Non-recursive should still have all original elements [1,2,3,4,5,6,7,8,9,10,11,12]
    assert_eq!(non_recursive_flat_elements.len(), 12);

    // Recursive should only have elements still referenced [6,7,8,9,10,11,12]
    assert_eq!(recursive_flat_elements.len(), 7);

    // Verify data integrity is preserved
    let non_recursive_array = non_recursive.into_array();
    let recursive_array = recursive.into_array();
    assert_eq!(
        non_recursive_array
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap(),
        recursive_array
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap()
    );
}

#[test]
fn test_filter_sliced_list_array() -> VortexResult<()> {
    let list = ListArray::try_new(
        buffer![0..50].into_array(),
        buffer![0, 10, 20, 30, 40, 50].into_array(),
        Validity::AllValid,
    )?
    .into_array()
    .slice(2..5)?;

    let mask = Mask::from(BitBuffer::from(vec![true, false, true]));
    let filter_array = FilterArray::new(list, mask).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let result = filter_array.execute::<Canonical>(&mut ctx)?;

    assert_eq!(result.len(), 2);
    Ok(())
}
