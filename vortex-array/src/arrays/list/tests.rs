// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_buffer::BooleanBuffer;
use vortex_dtype::PType::I32;
use vortex_dtype::{DType, Nullability};
use vortex_error::VortexUnwrap;
use vortex_mask::Mask;

use super::*;
use crate::arrays::PrimitiveArray;
use crate::compute::filter;

#[test]
fn test_empty_list_array() {
    let elements = PrimitiveArray::empty::<u32>(Nullability::NonNullable);
    let offsets = PrimitiveArray::from_iter([0]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

    assert_eq!(0, list.len());
}

#[test]
fn test_simple_list_array() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
    let offsets = PrimitiveArray::from_iter([0, 2, 4, 5]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

    assert_eq!(
        Scalar::list(
            Arc::new(I32.into()),
            vec![1.into(), 2.into()],
            Nullability::Nullable
        ),
        list.scalar_at(0)
    );
    assert_eq!(
        Scalar::list(
            Arc::new(I32.into()),
            vec![3.into(), 4.into()],
            Nullability::Nullable
        ),
        list.scalar_at(1)
    );
    assert_eq!(
        Scalar::list(Arc::new(I32.into()), vec![5.into()], Nullability::Nullable),
        list.scalar_at(2)
    );
}

#[test]
fn test_simple_list_array_from_iter() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3]);
    let offsets = PrimitiveArray::from_iter([0, 2, 3]);
    let validity = Validity::NonNullable;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();

    let list_from_iter =
        ListArray::from_iter_slow::<u32, _>(vec![vec![1i32, 2], vec![3]], Arc::new(I32.into()))
            .unwrap();

    assert_eq!(list.len(), list_from_iter.len());
    assert_eq!(list.scalar_at(0), list_from_iter.scalar_at(0));
    assert_eq!(list.scalar_at(1), list_from_iter.scalar_at(1));
}

#[test]
fn test_simple_list_filter() {
    let elements = PrimitiveArray::from_option_iter([None, Some(2), Some(3), Some(4), Some(5)]);
    let offsets = PrimitiveArray::from_iter([0, 2, 4, 5]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    let filtered = filter(
        &list,
        &Mask::from(BooleanBuffer::from(vec![false, true, true])),
    );

    assert!(filtered.is_ok())
}

#[test]
fn test_list_filter_dense_mask() {
    // Test filtering with a dense mask (high density of true values).
    let elements = PrimitiveArray::from_iter(0..100);
    let offsets = PrimitiveArray::from_iter([0, 10, 25, 40, 60, 85, 100]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    // Dense mask: keep most elements (indices 1, 2, 3, 4, 5).
    let mask = Mask::from(BooleanBuffer::from(vec![
        false, true, true, true, true, true,
    ]));

    let filtered = filter(&list, &mask).unwrap();
    let filtered_list = filtered.as_::<ListVTable>();

    // Should have 5 lists remaining.
    assert_eq!(filtered_list.len(), 5);

    // Verify first remaining list (originally index 1) has correct elements.
    let first_list = filtered_list.elements_at(0);
    assert_eq!(first_list.len(), 15); // 25 - 10
    assert_eq!(first_list.scalar_at(0), 10.into());
    assert_eq!(first_list.scalar_at(14), 24.into());
}

#[test]
fn test_list_filter_sparse_mask() {
    // Test filtering with a sparse mask (low density of true values).
    let elements = PrimitiveArray::from_iter(0..100);
    let offsets = PrimitiveArray::from_iter([0, 10, 25, 40, 60, 85, 100]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    // Sparse mask: keep only a few elements (indices 0 and 5).
    let mask = Mask::from(BooleanBuffer::from(vec![
        true, false, false, false, false, true,
    ]));

    let filtered = filter(&list, &mask).unwrap();
    let filtered_list = filtered.as_::<ListVTable>();

    // Should have 2 lists remaining.
    assert_eq!(filtered_list.len(), 2);

    // Verify first list (originally index 0).
    let first_list = filtered_list.elements_at(0);
    assert_eq!(first_list.len(), 10);
    assert_eq!(first_list.scalar_at(0), 0.into());
    assert_eq!(first_list.scalar_at(9), 9.into());

    // Verify second list (originally index 5).
    let second_list = filtered_list.elements_at(1);
    assert_eq!(second_list.len(), 15); // 100 - 85
    assert_eq!(second_list.scalar_at(0), 85.into());
    assert_eq!(second_list.scalar_at(14), 99.into());
}

#[test]
fn test_list_filter_empty_lists() {
    // Test filtering arrays that contain empty lists.
    let elements = PrimitiveArray::from_iter(0..10);
    let offsets = PrimitiveArray::from_iter([0, 0, 3, 3, 7, 10, 10]); // Lists at indices 0, 2, 5 are empty.
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    let mask = Mask::from(BooleanBuffer::from(vec![
        true, true, true, false, false, true,
    ]));

    let filtered = filter(&list, &mask).unwrap();
    let filtered_list = filtered.as_::<ListVTable>();

    assert_eq!(filtered_list.len(), 4);

    // First list is empty.
    assert_eq!(filtered_list.elements_at(0).len(), 0);

    // Second list has 3 elements.
    let second_list = filtered_list.elements_at(1);
    assert_eq!(second_list.len(), 3);
    assert_eq!(second_list.scalar_at(0), 0.into());

    // Third list is empty.
    assert_eq!(filtered_list.elements_at(2).len(), 0);

    // Fourth list is empty.
    assert_eq!(filtered_list.elements_at(3).len(), 0);
}

#[test]
fn test_list_filter_with_nulls() {
    // Test filtering lists with null validity.
    let elements = PrimitiveArray::from_iter(0..15);
    let offsets = PrimitiveArray::from_iter([0, 3, 7, 10, 12, 15]);
    let validity = Validity::from_mask(
        Mask::from(BooleanBuffer::from(vec![true, false, true, false, true])),
        Nullability::Nullable,
    );

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    let mask = Mask::from(BooleanBuffer::from(vec![true, true, false, true, true]));

    let filtered = filter(&list, &mask).unwrap();
    let filtered_list = filtered.as_::<ListVTable>();

    assert_eq!(filtered_list.len(), 4);

    // Check validity of filtered array.
    assert!(filtered_list.scalar_at(0).is_valid());
    assert!(!filtered_list.scalar_at(1).is_valid()); // Was null.
    assert!(!filtered_list.scalar_at(2).is_valid()); // Was null.
    assert!(filtered_list.scalar_at(3).is_valid());
}

#[test]
fn test_list_filter_all_true() {
    // Test filtering with an all-true mask.
    let elements = PrimitiveArray::from_iter(0..20);
    let offsets = PrimitiveArray::from_iter([0, 5, 10, 15, 20]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    let mask = Mask::AllTrue(4);

    let filtered = filter(&list, &mask).unwrap();
    let filtered_list = filtered.as_::<ListVTable>();

    // All lists should be preserved.
    assert_eq!(filtered_list.len(), 4);

    // Verify all lists are intact.
    for i in 0..4i32 {
        let list_at_i = filtered_list.elements_at(i as usize);
        assert_eq!(list_at_i.len(), 5);
        assert_eq!(list_at_i.scalar_at(0), (i * 5).into());
    }
}

#[test]
fn test_list_filter_all_false() {
    // Test filtering with an all-false mask.
    let elements = PrimitiveArray::from_iter(0..20);
    let offsets = PrimitiveArray::from_iter([0, 5, 10, 15, 20]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    let mask = Mask::AllFalse(4);

    let filtered = filter(&list, &mask).unwrap();
    let filtered_list = filtered.as_::<ListVTable>();

    // No lists should remain.
    assert_eq!(filtered_list.len(), 0);
}

#[test]
fn test_list_filter_single_element() {
    // Test filtering to keep only one element.
    let elements = PrimitiveArray::from_iter(0..50);
    let offsets = PrimitiveArray::from_iter([0, 10, 20, 30, 40, 50]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    let mask = Mask::from(BooleanBuffer::from(vec![false, false, true, false, false]));

    let filtered = filter(&list, &mask).unwrap();
    let filtered_list = filtered.as_::<ListVTable>();

    assert_eq!(filtered_list.len(), 1);

    let single_list = filtered_list.elements_at(0);
    assert_eq!(single_list.len(), 10);
    assert_eq!(single_list.scalar_at(0), 20.into());
    assert_eq!(single_list.scalar_at(9), 29.into());
}

#[test]
fn test_list_filter_alternating_pattern() {
    // Test filtering with an alternating pattern.
    let elements = PrimitiveArray::from_iter(0..60);
    let offsets = PrimitiveArray::from_iter([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    // Keep every other list.
    let mask = Mask::from(BooleanBuffer::from(vec![
        true, false, true, false, true, false, true, false, true, false, true, false,
    ]));

    let filtered = filter(&list, &mask).unwrap();
    let filtered_list = filtered.as_::<ListVTable>();

    assert_eq!(filtered_list.len(), 6);

    // Verify that we have the correct lists (0, 2, 4, 6, 8, 10).
    for (i, expected_start) in [0, 10, 20, 30, 40, 50].iter().enumerate() {
        let list_at_i = filtered_list.elements_at(i);
        assert_eq!(list_at_i.len(), 5);
        assert_eq!(list_at_i.scalar_at(0), (*expected_start).into());
    }
}

#[test]
fn test_list_filter_variable_sizes() {
    // Test filtering lists with highly variable sizes.
    let elements = PrimitiveArray::from_iter(0..100);
    let offsets = PrimitiveArray::from_iter([0, 1, 2, 5, 10, 20, 35, 60, 100]);
    let validity = Validity::AllValid;

    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity)
        .unwrap()
        .into_array();

    let mask = Mask::from(BooleanBuffer::from(vec![
        true, false, true, true, false, true, true, true,
    ]));

    let filtered = filter(&list, &mask).unwrap();
    let filtered_list = filtered.as_::<ListVTable>();

    assert_eq!(filtered_list.len(), 6);

    // Verify sizes of filtered lists.
    assert_eq!(filtered_list.elements_at(0).len(), 1); // Size 1
    assert_eq!(filtered_list.elements_at(1).len(), 3); // Size 3
    assert_eq!(filtered_list.elements_at(2).len(), 5); // Size 5
    assert_eq!(filtered_list.elements_at(3).len(), 15); // Size 15
    assert_eq!(filtered_list.elements_at(4).len(), 25); // Size 25
    assert_eq!(filtered_list.elements_at(5).len(), 40); // Size 40
}

#[test]
fn test_offset_to_0() {
    let mut builder =
        ListBuilder::<u32>::with_capacity(Arc::new(I32.into()), Nullability::NonNullable, 5);
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![1.into(), 2.into(), 3.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![4.into(), 5.into(), 6.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![7.into(), 8.into(), 9.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![10.into(), 11.into(), 12.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    builder
        .append_value(
            Scalar::list(
                Arc::new(I32.into()),
                vec![13.into(), 14.into(), 15.into()],
                Nullability::NonNullable,
            )
            .as_list(),
        )
        .vortex_unwrap();
    let list = builder.finish().slice(2..4);
    let list = list.as_::<ListVTable>().reset_offsets().unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list.offsets().len(), 3);
    assert_eq!(list.elements().len(), 6);
    assert_eq!(list.offsets().scalar_at(0), 0u32.into());
}

type OptVec<T> = Vec<Option<T>>;

// Helper function to create a list of lists from a 3D vector with Option types.
#[allow(clippy::cast_possible_truncation)]
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
            Mask::from(BooleanBuffer::from(inner_validity)),
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
            Mask::from(BooleanBuffer::from(outer_validity)),
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
#[allow(clippy::cognitive_complexity)]
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
    let first_outer = list_of_lists.elements_at(0);
    let first_outer_list = first_outer.as_::<ListVTable>();
    assert_eq!(first_outer_list.len(), 2);

    // Check first inner list [1, 2].
    let first_inner = first_outer_list.elements_at(0);
    assert_eq!(first_inner.len(), 2);
    assert_eq!(first_inner.scalar_at(0), 1.into());
    assert_eq!(first_inner.scalar_at(1), 2.into());

    // Check second inner list [3].
    let second_inner = first_outer_list.elements_at(1);
    assert_eq!(second_inner.len(), 1);
    assert_eq!(second_inner.scalar_at(0), 3.into());

    // Check the second list of lists [[4, 5, 6]].
    let second_outer = list_of_lists.elements_at(1);
    let second_outer_list = second_outer.as_::<ListVTable>();
    assert_eq!(second_outer_list.len(), 1);

    let inner = second_outer_list.elements_at(0);
    assert_eq!(inner.len(), 3);
    assert_eq!(inner.scalar_at(0), 4.into());
    assert_eq!(inner.scalar_at(1), 5.into());
    assert_eq!(inner.scalar_at(2), 6.into());

    // Check the third list of lists (empty).
    let third_outer = list_of_lists.elements_at(2);
    let third_outer_list = third_outer.as_::<ListVTable>();
    assert_eq!(third_outer_list.len(), 0);

    // Check the fourth list of lists [[7]].
    let fourth_outer = list_of_lists.elements_at(3);
    let fourth_outer_list = fourth_outer.as_::<ListVTable>();
    assert_eq!(fourth_outer_list.len(), 1);

    let inner = fourth_outer_list.elements_at(0);
    assert_eq!(inner.len(), 1);
    assert_eq!(inner.scalar_at(0), 7.into());

    // Test scalar conversion.
    let scalar = list_of_lists.scalar_at(0);
    assert!(matches!(scalar.dtype(), DType::List(_, _)));
    let list_scalar = scalar.as_list();
    assert_eq!(list_scalar.len(), 2);

    // Test slicing.
    let sliced = list_of_lists.slice(1..3);
    let sliced_list = sliced.as_::<ListVTable>();
    assert_eq!(sliced_list.len(), 2);

    // First element of slice should be [[4, 5, 6]].
    let first_sliced = sliced_list.elements_at(0);
    let first_sliced_list = first_sliced.as_::<ListVTable>();
    assert_eq!(first_sliced_list.len(), 1);

    // Second element of slice should be empty [].
    let second_sliced = sliced_list.elements_at(1);
    let second_sliced_list = second_sliced.as_::<ListVTable>();
    assert_eq!(second_sliced_list.len(), 0);
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
    let first = list_of_lists.scalar_at(0);
    assert!(!first.is_null());

    // Second element should be null.
    let second = list_of_lists.scalar_at(1);
    assert!(second.is_null());

    // Third element should be [[4, 5, 6]].
    let third = list_of_lists.elements_at(2);
    let third_list = third.as_::<ListVTable>();
    assert_eq!(third_list.len(), 1);
    let inner = third_list.elements_at(0);
    assert_eq!(inner.len(), 3);

    // Fourth element should be [[7]].
    let fourth = list_of_lists.elements_at(3);
    let fourth_list = fourth.as_::<ListVTable>();
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
    let first_outer = list_of_lists.elements_at(0);
    let first_list = first_outer.as_::<ListVTable>();
    assert_eq!(first_list.len(), 3);

    // Check that second inner list is null.
    let second_inner = first_list.scalar_at(1);
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
    let first_outer = list_of_lists.scalar_at(0);
    assert!(!first_outer.is_null());
    let first_outer_array = list_of_lists.elements_at(0);
    let first_list = first_outer_array.as_::<ListVTable>();
    assert_eq!(first_list.len(), 2);

    // First inner list should be [1, 2].
    let first_inner = first_list.elements_at(0);
    assert_eq!(first_inner.len(), 2);

    // Second inner list should be null.
    let second_inner = first_list.scalar_at(1);
    assert!(second_inner.is_null());

    // Second outer list should be null.
    let second_outer = list_of_lists.scalar_at(1);
    assert!(second_outer.is_null());

    // Third outer list should have [3].
    let third_outer = list_of_lists.elements_at(2);
    let third_list = third_outer.as_::<ListVTable>();
    assert_eq!(third_list.len(), 1);
    let inner = third_list.elements_at(0);
    assert_eq!(inner.len(), 1);
    assert_eq!(inner.scalar_at(0), 3.into());

    // Fourth outer list should have a null inner list.
    let fourth_outer = list_of_lists.elements_at(3);
    let fourth_list = fourth_outer.as_::<ListVTable>();
    assert_eq!(fourth_list.len(), 1);
    let inner = fourth_list.scalar_at(0);
    assert!(inner.is_null());
}

#[test]
#[should_panic(expected = "offsets minimum -1 outside valid range")]
fn test_negative_offset_values() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
    let offsets = PrimitiveArray::from_iter([-1i32, 2, 4, 5]);
    let validity = Validity::AllValid;

    let _ = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();
}

#[test]
#[should_panic(expected = "offsets must be sorted")]
fn test_unsorted_offsets() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
    let offsets = PrimitiveArray::from_iter([0u32, 3, 2, 5]);
    let validity = Validity::AllValid;

    let _ = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();
}

#[test]
#[should_panic(expected = "Max offset 7 is beyond the length of the elements array 5")]
fn test_offset_exceeding_elements_length() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
    let offsets = PrimitiveArray::from_iter([0u32, 2, 4, 7]);
    let validity = Validity::AllValid;

    let _ = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();
}

#[test]
#[should_panic(expected = "validity with size 2 does not match array size 4")]
fn test_validity_length_mismatch() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
    let offsets = PrimitiveArray::from_iter([0u32, 2, 4, 5, 5]);
    let validity = Validity::from_mask(
        Mask::from(BooleanBuffer::from(vec![true, false])),
        Nullability::Nullable,
    );

    let _ = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();
}

#[test]
#[should_panic(expected = "offsets have invalid type")]
fn test_nullable_offsets() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
    let offsets = PrimitiveArray::from_option_iter([Some(0u32), Some(2), None, Some(5)]);
    let validity = Validity::AllValid;

    let _ = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();
}

#[test]
#[should_panic(expected = "Offsets must have at least one element, [0] for an empty list")]
fn test_empty_offsets_array() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3]);
    let offsets = PrimitiveArray::empty::<u32>(Nullability::NonNullable);
    let validity = Validity::AllValid;

    let _ = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();
}

#[test]
#[should_panic(expected = "offsets have invalid type")]
fn test_non_integer_offsets() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
    let offsets = PrimitiveArray::from_iter([0.0f32, 2.0, 4.0, 5.0]);
    let validity = Validity::AllValid;

    let _ = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();
}

#[test]
fn test_offsets_constant() {
    let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
    let offsets = PrimitiveArray::from_iter([5u32, 5, 5, 5]);
    let validity = Validity::AllValid;

    // This should succeed as it represents empty lists
    let list = ListArray::try_new(elements.into_array(), offsets.into_array(), validity).unwrap();
    assert_eq!(list.len(), 3);
    assert_eq!(list.elements_at(0).len(), 0);
    assert_eq!(list.elements_at(1).len(), 0);
    assert_eq!(list.elements_at(2).len(), 0);
}
