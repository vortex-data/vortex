// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_mask::Mask;
use vortex_mask::MaskMut;

use crate::Vector;
use crate::VectorMutOps;
use crate::VectorOps;
use crate::listview::ListViewVector;
use crate::listview::ListViewVectorMut;
use crate::primitive::PVectorMut;
use crate::primitive::PrimitiveVector;

// TODO(connor): This should probably be a method directly on the vector.
// Helper function to get list values at index
fn get_list_values(list: &ListViewVector, list_idx: usize) -> Vec<i32> {
    let offsets = list.offsets();
    let sizes = list.sizes();

    // Get offset and size for this list entry
    let offset = match offsets {
        PrimitiveVector::U32(pvec) => *pvec.get(list_idx).unwrap_or(&0) as usize,
        PrimitiveVector::I32(pvec) => *pvec.get(list_idx).unwrap_or(&0) as usize,
        _ => panic!("Unsupported offset type in test"),
    };

    let size = match sizes {
        PrimitiveVector::U32(pvec) => *pvec.get(list_idx).unwrap_or(&0) as usize,
        PrimitiveVector::I32(pvec) => *pvec.get(list_idx).unwrap_or(&0) as usize,
        _ => panic!("Unsupported size type in test"),
    };

    // Extract values from elements vector
    let elements = list.elements();
    if let Vector::Primitive(PrimitiveVector::I32(pvec)) = &**elements {
        let mut values = Vec::new();
        for i in offset..(offset + size) {
            if let Some(val) = pvec.get(i) {
                values.push(*val);
            }
        }
        return values;
    }
    panic!("Elements not i32 in test");
}

#[test]
fn test_basic_list_operations_with_values() {
    // Create a list vector with:
    // List 0: [1, 2, 3]
    // List 1: [4, 5]
    // List 2: [6, 7, 8, 9]

    let elements: PrimitiveVector = PVectorMut::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8, 9])
        .freeze()
        .into();
    let offsets: PrimitiveVector = PVectorMut::from_iter([0u32, 3, 5]).freeze().into();
    let sizes: PrimitiveVector = PVectorMut::from_iter([3u32, 2, 4]).freeze().into();
    let validity = Mask::new_true(3);

    let list = ListViewVector::new(Arc::new(elements.into()), offsets, sizes, validity);

    // Verify length
    assert_eq!(list.len(), 3);

    // Verify actual values in each list
    assert_eq!(get_list_values(&list, 0), vec![1, 2, 3]);
    assert_eq!(get_list_values(&list, 1), vec![4, 5]);
    assert_eq!(get_list_values(&list, 2), vec![6, 7, 8, 9]);
}

#[test]
fn test_overlapping_views() {
    // Create overlapping views into the same elements
    // Elements: [10, 20, 30, 40, 50]
    // List 0: [10, 20, 30] (offset=0, size=3)
    // List 1: [20, 30, 40] (offset=1, size=3)
    // List 2: [30, 40, 50] (offset=2, size=3)

    let elements: PrimitiveVector = PVectorMut::from_iter([10i32, 20, 30, 40, 50])
        .freeze()
        .into();
    let offsets: PrimitiveVector = PVectorMut::from_iter([0u32, 1, 2]).freeze().into();
    let sizes: PrimitiveVector = PVectorMut::from_iter([3u32, 3, 3]).freeze().into();
    let validity = Mask::new_true(3);

    let list = ListViewVector::new(Arc::new(elements.into()), offsets, sizes, validity);

    // Verify the overlapping views show correct values
    assert_eq!(get_list_values(&list, 0), vec![10, 20, 30]);
    assert_eq!(get_list_values(&list, 1), vec![20, 30, 40]);
    assert_eq!(get_list_values(&list, 2), vec![30, 40, 50]);
}

#[test]
fn test_non_contiguous_views() {
    // Create non-contiguous views (with gaps)
    // Elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    // List 0: [1, 2] (offset=0, size=2)
    // List 1: [5, 6] (offset=4, size=2)  -- skips 3, 4
    // List 2: [9, 10] (offset=8, size=2) -- skips 7, 8

    let elements: PrimitiveVector = PVectorMut::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .freeze()
        .into();
    let offsets: PrimitiveVector = PVectorMut::from_iter([0u32, 4, 8]).freeze().into();
    let sizes: PrimitiveVector = PVectorMut::from_iter([2u32, 2, 2]).freeze().into();
    let validity = Mask::new_true(3);

    let list = ListViewVector::new(Arc::new(elements.into()), offsets, sizes, validity);

    // Verify the non-contiguous views
    assert_eq!(get_list_values(&list, 0), vec![1, 2]);
    assert_eq!(get_list_values(&list, 1), vec![5, 6]);
    assert_eq!(get_list_values(&list, 2), vec![9, 10]);
}

#[test]
fn test_unsorted_offsets() {
    // Test that unsorted offsets work correctly
    // Elements: [100, 200, 300, 400, 500]
    // List 0: [300, 400] (offset=2, size=2)
    // List 1: [100, 200] (offset=0, size=2)  -- out of order!
    // List 2: [500] (offset=4, size=1)

    let elements: PrimitiveVector = PVectorMut::from_iter([100i32, 200, 300, 400, 500])
        .freeze()
        .into();
    let offsets: PrimitiveVector = PVectorMut::from_iter([2u32, 0, 4]).freeze().into();
    let sizes: PrimitiveVector = PVectorMut::from_iter([2u32, 2, 1]).freeze().into();
    let validity = Mask::new_true(3);

    let list = ListViewVector::new(Arc::new(elements.into()), offsets, sizes, validity);

    // Verify values respect the unsorted offsets
    assert_eq!(get_list_values(&list, 0), vec![300, 400]);
    assert_eq!(get_list_values(&list, 1), vec![100, 200]);
    assert_eq!(get_list_values(&list, 2), vec![500]);
}

#[test]
fn test_empty_lists() {
    // Test lists with size=0
    // Elements: [1, 2, 3]
    // List 0: [] (offset=0, size=0)
    // List 1: [1, 2] (offset=0, size=2)
    // List 2: [] (offset=2, size=0)
    // List 3: [3] (offset=2, size=1)

    let elements: PrimitiveVector = PVectorMut::from_iter([1i32, 2, 3]).freeze().into();
    let offsets: PrimitiveVector = PVectorMut::from_iter([0u32, 0, 2, 2]).freeze().into();
    let sizes: PrimitiveVector = PVectorMut::from_iter([0u32, 2, 0, 1]).freeze().into();
    let validity = Mask::new_true(4);

    let list = ListViewVector::new(Arc::new(elements.into()), offsets, sizes, validity);

    assert_eq!(list.len(), 4);
    assert_eq!(get_list_values(&list, 0), Vec::<i32>::new());
    assert_eq!(get_list_values(&list, 1), vec![1, 2]);
    assert_eq!(get_list_values(&list, 2), Vec::<i32>::new());
    assert_eq!(get_list_values(&list, 3), vec![3]);
}

#[test]
fn test_extend_with_offset_adjustment() {
    // Test that extend_from_vector properly adjusts offsets
    let elements1 = PVectorMut::from_iter([10i32, 20, 30, 40]);
    let offsets1 = PVectorMut::from_iter([0u32, 2]).into();
    let sizes1 = PVectorMut::from_iter([2u32, 2]).into();
    let validity1 = MaskMut::new_true(2);

    let mut list = ListViewVectorMut::new(Box::new(elements1.into()), offsets1, sizes1, validity1);

    // Second vector with its own offsets
    let elements2: PrimitiveVector = PVectorMut::from_iter([50i32, 60, 70]).freeze().into();
    let offsets2: PrimitiveVector = PVectorMut::from_iter([0u32, 1]).freeze().into();
    let sizes2: PrimitiveVector = PVectorMut::from_iter([1u32, 2]).freeze().into();
    let validity2 = Mask::new_true(2);

    let list2 = ListViewVector::new(Arc::new(elements2.into()), offsets2, sizes2, validity2);

    list.extend_from_vector(&list2);

    // Freeze and check values
    let frozen = list.freeze();
    assert_eq!(frozen.len(), 4);

    // Original lists
    assert_eq!(get_list_values(&frozen, 0), vec![10, 20]);
    assert_eq!(get_list_values(&frozen, 1), vec![30, 40]);

    // Extended lists - offsets should be adjusted by 4 (original elements length)
    // List 2: offset was 0, now 4, size 1 -> [50]
    // List 3: offset was 1, now 5, size 2 -> [60, 70]
    assert_eq!(get_list_values(&frozen, 2), vec![50]);
    assert_eq!(get_list_values(&frozen, 3), vec![60, 70]);
}

#[test]
fn test_nulls_with_valid_metadata() {
    // Test that null entries still require valid offset/size bounds
    let elements: PrimitiveVector = PVectorMut::from_iter([1i32, 2, 3, 4, 5]).freeze().into();
    let offsets: PrimitiveVector = PVectorMut::from_iter([0u32, 2, 3]).freeze().into();
    let sizes: PrimitiveVector = PVectorMut::from_iter([2u32, 1, 2]).freeze().into();
    // Mark the middle list as null
    let validity = Mask::from_indices(3, vec![0, 2]);

    let list = ListViewVector::new(Arc::new(elements.into()), offsets, sizes, validity);

    assert_eq!(list.len(), 3);

    // Check validity
    assert!(list.validity().value(0)); // First list is valid
    assert!(!list.validity().value(1)); // Second list is null
    assert!(list.validity().value(2)); // Third list is valid

    // Check values of valid lists
    assert_eq!(get_list_values(&list, 0), vec![1, 2]);
    assert_eq!(get_list_values(&list, 2), vec![4, 5]);
}

#[test]
fn test_validation_errors() {
    let elements: PrimitiveVector = PVectorMut::from_iter([1i32, 2, 3, 4, 5]).freeze().into();

    // Test length mismatch
    let offsets: PrimitiveVector = PVectorMut::from_iter([0u32, 2, 3]).freeze().into();
    let sizes: PrimitiveVector = PVectorMut::from_iter([2u32, 1]).freeze().into(); // Wrong length!
    let validity = Mask::new_true(3);

    let result = ListViewVector::try_new(
        Arc::new(elements.clone().into()),
        offsets,
        sizes,
        validity.clone(),
    );
    assert!(result.is_err());

    // Test bounds violation
    let bad_offsets = PVectorMut::from_iter([0u32, 2, 4]).freeze().into();
    let bad_sizes = PVectorMut::from_iter([2u32, 1, 2]).freeze().into(); // 4 + 2 = 6 > 5 elements
    let result = ListViewVector::try_new(
        Arc::new(elements.clone().into()),
        bad_offsets,
        bad_sizes,
        validity.clone(),
    );
    assert!(result.is_err());

    // Test nulls in metadata
    let null_offsets = PVectorMut::from_iter([Some(0u32), None, Some(3)])
        .freeze()
        .into();
    let sizes: PrimitiveVector = PVectorMut::from_iter([2u32, 1, 1]).freeze().into();
    let result = ListViewVector::try_new(Arc::new(elements.into()), null_offsets, sizes, validity);
    assert!(result.is_err());
}

#[test]
fn test_different_integer_types() {
    // Test various combinations of integer types
    let elements: PrimitiveVector = PVectorMut::from_iter([10i32, 20, 30, 40, 50, 60])
        .freeze()
        .into();

    // u8/u8 combination
    let offsets_u8: PrimitiveVector = PVectorMut::from_iter([0u8, 2, 4]).freeze().into();
    let sizes_u8: PrimitiveVector = PVectorMut::from_iter([2u8, 2, 2]).freeze().into();
    let validity = Mask::new_true(3);

    let list = ListViewVector::new(
        Arc::new(elements.clone().into()),
        offsets_u8,
        sizes_u8,
        validity.clone(),
    );
    assert_eq!(list.len(), 3);

    // i64/i32 combination (signed types)
    let offsets_i64: PrimitiveVector = PVectorMut::from_iter([0i64, 2, 4]).freeze().into();
    let sizes_i32: PrimitiveVector = PVectorMut::from_iter([2i32, 2, 2]).freeze().into();

    let list2 = ListViewVector::new(Arc::new(elements.into()), offsets_i64, sizes_i32, validity);
    assert_eq!(list2.len(), 3);
}

#[test]
fn test_list_of_lists() {
    // Create a 2-level nested structure
    // Inner level: multiple lists of i32
    let inner_elements = PVectorMut::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8, 9]);
    let inner_offsets = PVectorMut::from_iter([0u32, 2, 3, 5, 7]).into();
    let inner_sizes = PVectorMut::from_iter([2u32, 1, 2, 2, 2]).into();
    let inner_validity = MaskMut::new_true(5);

    let inner_list = ListViewVectorMut::new(
        Box::new(inner_elements.into()),
        inner_offsets,
        inner_sizes,
        inner_validity,
    );

    // Outer level: lists that reference the inner lists
    // Outer list 0: inner lists [0,1,2] -> [[1,2], [3], [4,5]]
    // Outer list 1: inner lists [3,4] -> [[6,7], [8,9]]
    let outer_offsets = PVectorMut::from_iter([0u32, 3]).into();
    let outer_sizes = PVectorMut::from_iter([3u32, 2]).into();
    let outer_validity = MaskMut::new_true(2);

    let outer_list = ListViewVectorMut::new(
        Box::new(inner_list.into()),
        outer_offsets,
        outer_sizes,
        outer_validity,
    );

    let frozen = outer_list.freeze();
    assert_eq!(frozen.len(), 2);

    // We have a 2-level structure
    // The outer list contains 2 lists
    // Each outer list contains multiple inner lists
}

#[test]
fn test_append_nulls() {
    let elements = PVectorMut::from_iter([100i32, 200]);
    let offsets = PVectorMut::from_iter([0u32]).into();
    let sizes = PVectorMut::from_iter([2u32]).into();
    let validity = MaskMut::new_true(1);

    let mut list = ListViewVectorMut::new(Box::new(elements.into()), offsets, sizes, validity);

    assert_eq!(list.len(), 1);
    list.append_nulls(3);
    assert_eq!(list.len(), 4);

    let frozen = list.freeze();

    // Check validity
    assert!(frozen.validity().value(0)); // Original is valid
    assert!(!frozen.validity().value(1)); // Appended nulls
    assert!(!frozen.validity().value(2));
    assert!(!frozen.validity().value(3));

    // Check original values are preserved
    assert_eq!(get_list_values(&frozen, 0), vec![100, 200]);
}

#[test]
fn test_try_into_mut() {
    // Test: try_into_mut behavior
    // Note: try_into_mut may fail even when seemingly solely owned because
    // ALL components (elements, offsets, sizes, validity) must be convertible to mutable

    let elements: PrimitiveVector = PVectorMut::from_iter([1i32, 2, 3, 4]).freeze().into();
    let offsets: PrimitiveVector = PVectorMut::from_iter([0u32, 2]).freeze().into();
    let sizes: PrimitiveVector = PVectorMut::from_iter([2u32, 2]).freeze().into();
    let validity = Mask::new_true(2);

    let list = ListViewVector::new(Arc::new(elements.into()), offsets, sizes, validity);

    // Try to convert to mutable
    let mut_result = list.try_into_mut();

    match mut_result {
        Ok(mut list_mut) => {
            // If conversion succeeds, verify mutation works
            assert_eq!(list_mut.len(), 2);

            list_mut.append_nulls(1);
            assert_eq!(list_mut.len(), 3);

            let frozen = list_mut.freeze();
            assert_eq!(get_list_values(&frozen, 0), vec![1, 2]);
            assert_eq!(get_list_values(&frozen, 1), vec![3, 4]);
        }
        Err(list_back) => {
            // If conversion fails (which is OK because internal components might be shared),
            // verify we get back a valid immutable vector
            assert_eq!(list_back.len(), 2);
            assert_eq!(get_list_values(&list_back, 0), vec![1, 2]);
            assert_eq!(get_list_values(&list_back, 1), vec![3, 4]);
        }
    }

    // Test explicit sharing - should definitely fail
    let elements2: PrimitiveVector = PVectorMut::from_iter([10i32, 20, 30]).freeze().into();
    let offsets2: PrimitiveVector = PVectorMut::from_iter([0u32, 2]).freeze().into();
    let sizes2: PrimitiveVector = PVectorMut::from_iter([2u32, 1]).freeze().into();
    let validity2 = Mask::new_true(2);

    let shared_elements: Arc<Vector> = Arc::new(elements2.into());
    let list2 = ListViewVector::new(
        shared_elements.clone(), // Clone the Arc to create another reference
        offsets2,
        sizes2,
        validity2,
    );

    // Keep an extra reference to force sharing
    let _keep_ref = shared_elements;

    // Should fail because elements are shared
    let mut_result2 = list2.try_into_mut();
    assert!(mut_result2.is_err());

    // Verify we get back the original immutable vector
    let original = mut_result2.unwrap_err();
    assert_eq!(original.len(), 2);
}
