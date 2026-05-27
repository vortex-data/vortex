// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::BitBuffer;

use crate::AllOr;
use crate::Mask;
use crate::MaskIter;

// Basic mask creation and properties tests
#[test]
fn mask_all_true() {
    let mask = Mask::new_true(5);
    assert_eq!(mask.len(), 5);
    assert_eq!(mask.true_count(), 5);
    assert_eq!(mask.density(), 1.0);
    assert_eq!(mask.indices(), AllOr::All);
    assert_eq!(mask.slices(), AllOr::All);
    assert_eq!(mask.bit_buffer(), AllOr::All,);
}

#[test]
fn mask_all_false() {
    let mask = Mask::new_false(5);
    assert_eq!(mask.len(), 5);
    assert_eq!(mask.true_count(), 0);
    assert_eq!(mask.density(), 0.0);
    assert_eq!(mask.indices(), AllOr::None);
    assert_eq!(mask.slices(), AllOr::None);
    assert_eq!(mask.bit_buffer(), AllOr::None,);
}

#[test]
fn mask_from() {
    let masks = [
        Mask::from_indices(5, vec![0, 2, 3]),
        Mask::from_slices(5, vec![(0, 1), (2, 4)]),
        Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false])),
    ];

    for mask in &masks {
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.true_count(), 3);
        assert_eq!(mask.density(), 0.6);
        assert_eq!(mask.indices(), AllOr::Some(&[0, 2, 3][..]));
        assert_eq!(mask.slices(), AllOr::Some(&[(0, 1), (2, 4)][..]));
        assert_eq!(
            mask.bit_buffer(),
            AllOr::Some(&BitBuffer::from_iter([true, false, true, true, false]))
        );
    }
}

#[test]
fn length_zero_masks() {
    let all_false = Mask::new_false(0);
    let all_true = Mask::new_true(0);
    let buffer_set = Mask::from_buffer(BitBuffer::new_set(0));
    let buffer_unset = Mask::from_buffer(BitBuffer::new_unset(0));

    assert!(all_false.all_false());
    assert!(all_false.all_true());
    assert!(all_true.all_false());
    assert!(all_true.all_true());
    assert!(buffer_set.all_false());
    assert!(buffer_set.all_true());
    assert!(buffer_unset.all_false());
    assert!(buffer_unset.all_true());
}

// Mask value access tests
#[test]
fn test_mask_value() {
    let all_true = Mask::new_true(5);
    assert!(all_true.value(0));
    assert!(all_true.value(4));

    let all_false = Mask::new_false(5);
    assert!(!all_false.value(0));
    assert!(!all_false.value(4));

    let values = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
    assert!(values.value(0));
    assert!(!values.value(1));
    assert!(values.value(2));
    assert!(!values.value(3));
    assert!(values.value(4));
}

#[test]
fn test_mask_first() {
    assert_eq!(Mask::new_true(5).first(), Some(0));
    assert_eq!(Mask::new_false(5).first(), None);
    assert_eq!(Mask::new_true(0).first(), None);

    let values = Mask::from_buffer(BitBuffer::from_iter([false, false, true, false, true]));
    assert_eq!(values.first(), Some(2));

    let values_indices = Mask::from_indices(5, vec![2, 4]);
    assert_eq!(values_indices.first(), Some(2));

    let values_slices = Mask::from_slices(5, vec![(2, 3), (4, 5)]);
    assert_eq!(values_slices.first(), Some(2));
}

#[test]
fn test_mask_false_count() {
    assert_eq!(Mask::new_true(5).false_count(), 0);
    assert_eq!(Mask::new_false(5).false_count(), 5);

    let values = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
    assert_eq!(values.false_count(), 2);
}

// Slice operations
#[test]
fn test_mask_slice() {
    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false]));

    let sliced = mask.slice(1..4);
    assert_eq!(sliced.len(), 3);
    assert_eq!(sliced.true_count(), 2);
    assert!(!sliced.value(0)); // false from index 1
    assert!(sliced.value(1)); // true from index 2
    assert!(sliced.value(2)); // true from index 3

    let all_true = Mask::new_true(10);
    let sliced_true = all_true.slice(2..7);
    assert!(sliced_true.all_true());
    assert_eq!(sliced_true.len(), 5);

    let all_false = Mask::new_false(10);
    let sliced_false = all_false.slice(2..7);
    assert!(sliced_false.all_false());
    assert_eq!(sliced_false.len(), 5);
}

#[test]
#[should_panic]
fn test_mask_slice_out_of_bounds() {
    let mask = Mask::new_true(5);
    mask.slice(3..8); // offset + length > len
}

// Limit operations
#[test]
fn limit_all_true_mask() {
    let all_true = Mask::new_true(4);
    let limited_mask = all_true.clone().limit(2);
    assert_eq!(all_true.len(), limited_mask.len());
    assert_eq!(limited_mask.true_count(), 2);
    assert_eq!(
        limited_mask.bit_buffer(),
        AllOr::Some(&BitBuffer::from_iter([true, true, false, false]))
    );

    let limited_mask = all_true.clone().limit(5);
    assert_eq!(limited_mask, all_true);
}

#[test]
fn limit_mask_values() {
    let original_mask = Mask::from_iter([true, true, false, true, false, true]);
    let limited_mask = original_mask.clone().limit(2);

    assert_eq!(
        limited_mask.bit_buffer(),
        AllOr::Some(&BitBuffer::from_iter([
            true, true, false, false, false, false
        ]))
    );
    assert_eq!(limited_mask.true_count(), 2);

    let limited_mask = original_mask.limit(3);

    assert_eq!(
        limited_mask.bit_buffer(),
        AllOr::Some(&BitBuffer::from_iter([
            true, true, false, true, false, false
        ]))
    );
    assert_eq!(limited_mask.true_count(), 3);

    let original_mask = Mask::from_iter([true, true, false, true, false, true]);
    let limited_mask = original_mask.clone().limit(100);

    assert_eq!(original_mask, limited_mask);
}

#[test]
fn test_limit_all_false_mask() {
    let all_false = Mask::new_false(10);
    let limited = all_false.clone().limit(5);
    assert_eq!(limited, all_false);
    assert!(limited.all_false());
}

#[test]
fn test_limit_mask_exact() {
    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
    let limited = mask.clone().limit(3);
    assert_eq!(limited.true_count(), 3);
    assert_eq!(limited, mask);
}

#[test]
fn test_limit_mask_zero() {
    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
    let limited = mask.limit(0);
    assert!(limited.all_false());
    assert_eq!(limited.true_count(), 0);
}

// Buffer conversion tests
#[test]
fn test_mask_to_bit_buffer() {
    let all_true = Mask::new_true(5);
    let buffer = all_true.to_bit_buffer();
    assert_eq!(buffer.true_count(), 5);
    assert_eq!(buffer.len(), 5);

    let all_false = Mask::new_false(5);
    let buffer = all_false.to_bit_buffer();
    assert_eq!(buffer.true_count(), 0);
    assert_eq!(buffer.len(), 5);

    let values = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
    let buffer = values.to_bit_buffer();
    assert_eq!(buffer.true_count(), 3);
    assert_eq!(buffer.len(), 5);
}

// MaskValues tests
#[test]
fn test_mask_values() {
    let all_true = Mask::new_true(5);
    assert!(all_true.values().is_none());

    let all_false = Mask::new_false(5);
    assert!(all_false.values().is_none());

    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true]));
    let values = mask.values().unwrap();
    assert_eq!(values.len(), 5);
    assert_eq!(values.true_count(), 3);
    assert!(values.value(0));
    assert!(!values.value(1));
}

#[test]
fn test_mask_values_threshold_iter() {
    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false]));
    let values = mask.values().unwrap();

    // With low threshold, should prefer indices
    match values.threshold_iter(0.7) {
        MaskIter::Indices(indices) => {
            assert_eq!(indices, &[0, 2, 3]);
        }
        _ => panic!("Expected indices iterator"),
    }

    // With high threshold, should prefer slices
    match values.threshold_iter(0.5) {
        MaskIter::Slices(slices) => {
            assert_eq!(slices, &[(0, 1), (2, 4)]);
        }
        _ => panic!("Expected slices iterator"),
    }
}

#[test]
fn test_mask_values_cached_representations() {
    let from_buffer = Mask::from_buffer(BitBuffer::from_iter([true, false, true]));
    let values = from_buffer.values().unwrap();
    assert!(values.cached_indices().is_none());
    assert!(values.cached_slices().is_none());

    let from_indices = Mask::from_indices(5, [1, 3]);
    let values = from_indices.values().unwrap();
    assert_eq!(values.cached_indices(), Some([1, 3].as_slice()));
    assert!(values.cached_slices().is_none());

    let from_slices = Mask::from_slices(6, vec![(1, 3), (5, 6)]);
    let values = from_slices.values().unwrap();
    assert!(values.cached_indices().is_none());
    assert_eq!(values.cached_slices(), Some([(1, 3), (5, 6)].as_slice()));
}

#[test]
fn test_mask_values_is_empty() {
    let empty_mask = Mask::from_buffer(BitBuffer::new_unset(0));
    if let Some(values) = empty_mask.values() {
        assert!(values.is_empty());
    }

    let non_empty_mask = Mask::from_buffer(BitBuffer::from_iter([true, false]));
    if let Some(values) = non_empty_mask.values() {
        assert!(!values.is_empty());
    }
}

// Creation from excluded indices
#[test]
fn test_mask_from_excluded_indices() {
    let mask = Mask::from_excluded_indices(5, vec![1, 3]);
    assert_eq!(mask.len(), 5);
    assert_eq!(mask.true_count(), 3);
    assert!(mask.value(0));
    assert!(!mask.value(1));
    assert!(mask.value(2));
    assert!(!mask.value(3));
    assert!(mask.value(4));

    let mask_empty = Mask::from_excluded_indices(5, vec![]);
    assert!(mask_empty.all_true());
    // Verify it returns the optimized AllTrue variant
    assert!(matches!(mask_empty, Mask::AllTrue(5)));

    let mask_all = Mask::from_excluded_indices(3, vec![0, 1, 2]);
    assert!(mask_all.all_false());
    // Verify it returns the optimized AllFalse variant
    assert!(matches!(mask_all, Mask::AllFalse(3)));
}

// Intersection tests
#[test]
fn test_mask_from_intersection_indices() {
    let lhs = vec![0, 2, 3, 5, 7, 9];
    let rhs = vec![1, 2, 3, 4, 7, 8, 9];

    let mask = Mask::from_intersection_indices(10, lhs.into_iter(), rhs.into_iter());
    assert_eq!(mask.len(), 10);
    assert_eq!(mask.true_count(), 4); // 2, 3, 7, 9
    assert!(!mask.value(0));
    assert!(!mask.value(1));
    assert!(mask.value(2));
    assert!(mask.value(3));
    assert!(!mask.value(4));
    assert!(!mask.value(5));
    assert!(!mask.value(6));
    assert!(mask.value(7));
    assert!(!mask.value(8));
    assert!(mask.value(9));
}

#[test]
fn test_mask_from_intersection_indices_empty() {
    let lhs = vec![0, 2, 4];
    let rhs = vec![1, 3, 5];

    let mask = Mask::from_intersection_indices(6, lhs.into_iter(), rhs.into_iter());
    assert!(mask.all_false());
    assert_eq!(mask.true_count(), 0);
}

#[test]
fn test_mask_from_intersection_indices_same() {
    let indices = vec![1, 3, 5, 7];

    let mask =
        Mask::from_intersection_indices(10, indices.clone().into_iter(), indices.into_iter());
    assert_eq!(mask.len(), 10);
    assert_eq!(mask.true_count(), 4);
    assert!(mask.value(1));
    assert!(mask.value(3));
    assert!(mask.value(5));
    assert!(mask.value(7));
}

// Valid counts tests
#[test]
fn test_mask_valid_counts_for_indices() {
    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false, true]));
    let indices = vec![0, 2, 4, 6];
    let counts = mask.valid_counts_for_indices(&indices);
    assert_eq!(counts, vec![0, 1, 3, 4]);

    let all_true = Mask::new_true(6);
    let counts = all_true.valid_counts_for_indices(&indices);
    assert_eq!(counts, vec![0, 2, 4, 6]);

    let all_false = Mask::new_false(6);
    let counts = all_false.valid_counts_for_indices(&indices);
    assert_eq!(counts, vec![0, 0, 0, 0]);
}

#[test]
#[should_panic]
fn test_mask_valid_counts_for_indices_error() {
    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true]));
    let indices = vec![0, 2, 5]; // 5 is out of bounds
    mask.valid_counts_for_indices(&indices);
}

// FromIterator tests
#[test]
fn test_mask_from_iter_masks() {
    let masks = vec![
        Mask::from_buffer(BitBuffer::from_iter([true, false])),
        Mask::from_buffer(BitBuffer::from_iter([true, true, false])),
        Mask::from_buffer(BitBuffer::from_iter([false, true])),
    ];

    let combined = Mask::from_iter(masks);
    assert_eq!(combined.len(), 7);
    assert_eq!(combined.true_count(), 4);
    assert!(combined.value(0));
    assert!(!combined.value(1));
    assert!(combined.value(2));
    assert!(combined.value(3));
    assert!(!combined.value(4));
    assert!(!combined.value(5));
    assert!(combined.value(6));
}

#[test]
fn test_mask_from_iter_all_true() {
    let masks = vec![Mask::new_true(3), Mask::new_true(2), Mask::new_true(4)];

    let combined = Mask::from_iter(masks);
    assert!(combined.all_true());
    assert_eq!(combined.len(), 9);
}

#[test]
fn test_mask_from_iter_all_false() {
    let masks = vec![Mask::new_false(3), Mask::new_false(2), Mask::new_false(4)];

    let combined = Mask::from_iter(masks);
    assert!(combined.all_false());
    assert_eq!(combined.len(), 9);
}

#[test]
fn test_mask_from_iter_empty() {
    let masks: Vec<Mask> = vec![];
    let combined = Mask::from_iter(masks);
    assert_eq!(combined.len(), 0);
    assert!(combined.is_empty());
}

#[test]
fn test_mask_from_iter_with_empty_masks() {
    let masks = vec![
        Mask::new_true(3),
        Mask::new_true(0), // empty mask
        Mask::new_false(2),
    ];

    let combined = Mask::from_iter(masks);
    assert_eq!(combined.len(), 5);
    assert_eq!(combined.true_count(), 3);
}

// Panic tests for invalid inputs
#[test]
#[should_panic]
fn test_mask_from_indices_unsorted() {
    Mask::from_indices(5, vec![2, 0, 3]); // Not sorted
}

#[test]
#[should_panic]
fn test_mask_from_indices_duplicate() {
    Mask::from_indices(5, vec![0, 2, 2]); // Not unique
}

#[test]
#[should_panic]
fn test_mask_from_indices_out_of_bounds() {
    Mask::from_indices(5, vec![0, 2, 5]); // 5 is out of bounds
}

#[test]
#[should_panic]
fn test_mask_from_slices_invalid_range() {
    Mask::from_slices(5, vec![(2, 2)]); // Invalid range where start == end
}

#[test]
#[should_panic]
fn test_mask_from_slices_out_of_bounds() {
    Mask::from_slices(5, vec![(0, 6)]); // end > len
}

#[test]
#[should_panic]
fn test_mask_from_slices_unsorted() {
    Mask::from_slices(5, vec![(2, 3), (0, 1)]); // Not sorted
}

#[test]
#[should_panic]
fn test_mask_from_slices_overlapping() {
    Mask::from_slices(5, vec![(0, 3), (2, 4)]); // Overlapping ranges
}

// Threshold iterator tests
#[test]
fn test_mask_threshold_iter() {
    let all_true = Mask::new_true(5);
    assert!(matches!(all_true.threshold_iter(0.5), AllOr::All));

    let all_false = Mask::new_false(5);
    assert!(matches!(all_false.threshold_iter(0.5), AllOr::None));

    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false]));
    if let AllOr::Some(MaskIter::Indices(indices)) = mask.threshold_iter(0.7) {
        assert_eq!(indices, &[0, 2, 3]);
    } else {
        panic!("Expected indices iterator");
    }
}

// Caching tests
#[test]
fn test_mask_indices_caching() {
    // Test that indices are properly cached
    let mask = Mask::from_slices(10, vec![(0, 3), (5, 7), (9, 10)]);

    // First call should compute indices
    let indices1 = mask.indices();
    // Second call should return cached value
    let indices2 = mask.indices();

    match (indices1, indices2) {
        (AllOr::Some(i1), AllOr::Some(i2)) => {
            assert_eq!(i1, i2);
            assert_eq!(i1, &[0, 1, 2, 5, 6, 9]);
            // Verify they're the same reference (cached)
            assert!(std::ptr::eq(i1, i2));
        }
        _ => panic!("Expected Some variant"),
    }
}

#[test]
fn test_mask_slices_caching() {
    // Test that slices are properly cached
    let mask = Mask::from_indices(10, vec![0, 1, 2, 5, 6, 9]);

    // First call should compute slices
    let slices1 = mask.slices();
    // Second call should return cached value
    let slices2 = mask.slices();

    match (slices1, slices2) {
        (AllOr::Some(s1), AllOr::Some(s2)) => {
            assert_eq!(s1, s2);
            assert_eq!(s1, &[(0, 3), (5, 7), (9, 10)]);
            // Verify they're the same reference (cached)
            assert!(std::ptr::eq(s1, s2));
        }
        _ => panic!("Expected Some variant"),
    }
}

// AllOr tests
#[test]
fn test_allor_unwrap_or_else() {
    let all: AllOr<i32> = AllOr::All;
    assert_eq!(all.unwrap_or_else(|| 42, || 0), 42);

    let none: AllOr<i32> = AllOr::None;
    assert_eq!(none.unwrap_or_else(|| 42, || 0), 0);

    let some: AllOr<i32> = AllOr::Some(10);
    assert_eq!(some.unwrap_or_else(|| 42, || 0), 10);
}

#[test]
fn test_allor_cloned() {
    let original = vec![1, 2, 3];
    let all: AllOr<&Vec<i32>> = AllOr::All;
    assert!(matches!(all.cloned(), AllOr::All));

    let none: AllOr<&Vec<i32>> = AllOr::None;
    assert!(matches!(none.cloned(), AllOr::None));

    let some: AllOr<&Vec<i32>> = AllOr::Some(&original);
    if let AllOr::Some(cloned) = some.cloned() {
        assert_eq!(cloned, original);
    } else {
        panic!("Expected Some variant");
    }
}

#[test]
fn test_allor_eq() {
    let all1: AllOr<i32> = AllOr::All;
    let all2: AllOr<i32> = AllOr::All;
    assert_eq!(all1, all2);

    let none1: AllOr<i32> = AllOr::None;
    let none2: AllOr<i32> = AllOr::None;
    assert_eq!(none1, none2);

    let some1: AllOr<i32> = AllOr::Some(42);
    let some2: AllOr<i32> = AllOr::Some(42);
    let some3: AllOr<i32> = AllOr::Some(43);
    assert_eq!(some1, some2);
    assert_ne!(some1, some3);

    assert_ne!(all1, none1);
    assert_ne!(all1, some1);
    assert_ne!(none1, some1);
}

// Parameterized tests with rstest
#[rstest]
#[case::all_true(Mask::new_true(5), 5, 5, 0, 1.0)]
#[case::all_false(Mask::new_false(5), 5, 0, 5, 0.0)]
#[case::mixed(
        Mask::from_buffer(BitBuffer::from_iter([true, false, true, false, true])),
        5, 3, 2, 0.6
    )]
#[case::single_true(Mask::from_indices(10, vec![5]), 10, 1, 9, 0.1)]
#[case::dense(Mask::from_indices(10, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]), 10, 9, 1, 0.9)]
fn test_mask_properties(
    #[case] mask: Mask,
    #[case] expected_len: usize,
    #[case] expected_true: usize,
    #[case] expected_false: usize,
    #[case] expected_density: f64,
) {
    assert_eq!(mask.len(), expected_len);
    assert_eq!(mask.true_count(), expected_true);
    assert_eq!(mask.false_count(), expected_false);
    assert!((mask.density() - expected_density).abs() < 1e-10);
}

#[rstest]
#[case::indices(vec![0, 2, 4], vec![(0, 1), (2, 3), (4, 5)])]
#[case::consecutive(vec![0, 1, 2], vec![(0, 3)])]
#[case::gap(vec![0, 1, 4, 5], vec![(0, 2), (4, 6)])]
#[case::single(vec![3], vec![(3, 4)])]
fn test_indices_to_slices_conversion(
    #[case] indices: Vec<usize>,
    #[case] expected_slices: Vec<(usize, usize)>,
) {
    let mask = Mask::from_indices(10, indices.clone());

    // Check indices
    if let AllOr::Some(actual_indices) = mask.indices() {
        assert_eq!(actual_indices, &indices[..]);
    } else {
        panic!("Expected Some variant for indices");
    }

    // Check slices
    if let AllOr::Some(actual_slices) = mask.slices() {
        assert_eq!(actual_slices, &expected_slices[..]);
    } else {
        panic!("Expected Some variant for slices");
    }
}

#[rstest]
#[case::empty_intersection(vec![0, 2, 4], vec![1, 3, 5], vec![])]
#[case::full_intersection(vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5])]
#[case::partial_intersection(vec![0, 1, 2, 3], vec![2, 3, 4, 5], vec![2, 3])]
#[case::subset_left(vec![1, 2], vec![0, 1, 2, 3], vec![1, 2])]
#[case::subset_right(vec![0, 1, 2, 3], vec![1, 2], vec![1, 2])]
fn test_intersection_indices(
    #[case] left: Vec<usize>,
    #[case] right: Vec<usize>,
    #[case] expected: Vec<usize>,
) {
    let mask = Mask::from_intersection_indices(10, left.into_iter(), right.into_iter());

    match mask.indices() {
        AllOr::Some(indices) if expected.is_empty() => assert!(indices.is_empty()),
        AllOr::Some(indices) => assert_eq!(indices, &expected[..]),
        AllOr::None if expected.is_empty() => {}
        AllOr::None | AllOr::All => panic!("Unexpected result for intersection"),
    }
}

// Concat operation tests
#[test]
fn test_mask_concat_empty() {
    let masks: Vec<Mask> = vec![];
    let result = Mask::concat(masks.iter()).unwrap();
    assert_eq!(result.len(), 0);
    assert!(result.is_empty());
}

#[test]
fn test_mask_concat_all_true() {
    let masks = [Mask::new_true(3), Mask::new_true(2)];
    let result = Mask::concat(masks.iter()).unwrap();
    assert_eq!(result.len(), 5);
    assert!(result.all_true());
}

#[test]
fn test_mask_concat_all_false() {
    let masks = [Mask::new_false(3), Mask::new_false(2)];
    let result = Mask::concat(masks.iter()).unwrap();
    assert_eq!(result.len(), 5);
    assert!(result.all_false());
}

#[test]
fn test_mask_concat_mixed_types() {
    let masks = [
        Mask::from_buffer(BitBuffer::from_iter([true, false, true])),
        Mask::new_true(2),
        Mask::new_false(3),
    ];

    let result = Mask::concat(masks.iter()).unwrap();
    assert_eq!(result.len(), 8);
    assert_eq!(result.true_count(), 4);

    // Verify the concatenated values
    assert!(result.value(0)); // from buffer
    assert!(!result.value(1)); // from buffer
    assert!(result.value(2)); // from buffer
    assert!(result.value(3)); // from all_true
    assert!(result.value(4)); // from all_true
    assert!(!result.value(5)); // from all_false
    assert!(!result.value(6)); // from all_false
    assert!(!result.value(7)); // from all_false
}
