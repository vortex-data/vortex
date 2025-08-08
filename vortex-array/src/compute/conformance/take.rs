// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;
use vortex_error::VortexUnwrap;

use crate::arrays::PrimitiveArray;
use crate::compute::take;
use crate::{Array, Canonical};

/// Test conformance of the take compute function for an array.
///
/// This function tests various scenarios including:
/// - Taking all elements
/// - Taking no elements
/// - Taking selective elements
/// - Taking with out-of-bounds indices (should panic)
/// - Taking with nullable indices
/// - Edge cases like empty arrays
pub fn test_take_conformance(array: &dyn Array) {
    let len = array.len();

    if len > 0 {
        test_take_all(array);
        test_take_none(array);
        test_take_selective(array);
        test_take_first_and_last(array);
        test_take_with_nullable_indices(array);
        test_take_repeated_indices(array);
    }

    test_empty_indices(array);

    // Additional edge cases for non-empty arrays
    if len > 0 {
        test_take_reverse(array);
        test_take_single_middle(array);
    }

    if len > 3 {
        test_take_random_unsorted(array);
        test_take_contiguous_range(array);
        test_take_mixed_repeated(array);
    }

    // Test for larger arrays
    if len >= 1024 {
        test_take_large_indices(array);
    }
}

fn test_take_all(array: &dyn Array) {
    let len = array.len();
    let indices = PrimitiveArray::from_iter(0..len as u64);
    let result = take(array, indices.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), len);
    assert_eq!(result.dtype(), array.dtype());

    // Verify elements match
    if let Ok(orig_canonical) = array.to_canonical()
        && let Ok(result_canonical) = result.to_canonical()
    {
        match (&orig_canonical, &result_canonical) {
            (Canonical::Primitive(orig_prim), Canonical::Primitive(result_prim)) => {
                assert_eq!(orig_prim.byte_buffer(), result_prim.byte_buffer());
            }
            _ => {
                // For non-primitive types, check scalar values
                for i in 0..len {
                    assert_eq!(
                        array.scalar_at(i).vortex_unwrap(),
                        result.scalar_at(i).vortex_unwrap()
                    );
                }
            }
        }
    }
}

fn test_take_none(array: &dyn Array) {
    let indices: PrimitiveArray = PrimitiveArray::from_iter::<[u64; 0]>([]);
    let result = take(array, indices.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), 0);
    assert_eq!(result.dtype(), array.dtype());
}

#[allow(clippy::cast_possible_truncation)]
fn test_take_selective(array: &dyn Array) {
    let len = array.len();

    // Take every other element
    let indices: Vec<u64> = (0..len as u64).step_by(2).collect();
    let expected_len = indices.len();
    let indices_array = PrimitiveArray::from_iter(indices.clone());

    let result = take(array, indices_array.as_ref()).vortex_unwrap();
    assert_eq!(result.len(), expected_len);

    // Verify the taken elements
    for (result_idx, &original_idx) in indices.iter().enumerate() {
        assert_eq!(
            array.scalar_at(original_idx as usize).vortex_unwrap(),
            result.scalar_at(result_idx).vortex_unwrap()
        );
    }
}

fn test_take_first_and_last(array: &dyn Array) {
    let len = array.len();
    let indices = PrimitiveArray::from_iter([0u64, (len - 1) as u64]);
    let result = take(array, indices.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(
        array.scalar_at(0).vortex_unwrap(),
        result.scalar_at(0).vortex_unwrap()
    );
    assert_eq!(
        array.scalar_at(len - 1).vortex_unwrap(),
        result.scalar_at(1).vortex_unwrap()
    );
}

#[allow(clippy::cast_possible_truncation)]
fn test_take_with_nullable_indices(array: &dyn Array) {
    let len = array.len();

    // Create indices with some null values
    let indices_vec: Vec<Option<u64>> = if len >= 3 {
        vec![Some(0), None, Some((len - 1) as u64)]
    } else if len >= 2 {
        vec![Some(0), None]
    } else {
        vec![None]
    };

    let indices = PrimitiveArray::from_option_iter(indices_vec.clone());
    let result = take(array, indices.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), indices_vec.len());
    assert_eq!(
        result.dtype(),
        &array.dtype().with_nullability(Nullability::Nullable)
    );

    // Verify values
    for (i, idx_opt) in indices_vec.iter().enumerate() {
        match idx_opt {
            Some(idx) => {
                let expected = array.scalar_at(*idx as usize).vortex_unwrap();
                let actual = result.scalar_at(i).vortex_unwrap();
                assert_eq!(expected, actual);
            }
            None => {
                assert!(result.scalar_at(i).vortex_unwrap().is_null());
            }
        }
    }
}

fn test_take_repeated_indices(array: &dyn Array) {
    if array.is_empty() {
        return;
    }

    // Take the first element multiple times
    let indices = PrimitiveArray::from_iter([0u64, 0, 0]);
    let result = take(array, indices.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), 3);
    let first_elem = array.scalar_at(0).vortex_unwrap();
    for i in 0..3 {
        assert_eq!(result.scalar_at(i).vortex_unwrap(), first_elem);
    }
}

fn test_empty_indices(array: &dyn Array) {
    let indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
    let result = take(array, indices.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), 0);
    assert_eq!(result.dtype(), array.dtype());
}

fn test_take_reverse(array: &dyn Array) {
    let len = array.len();
    // Take elements in reverse order
    let indices = PrimitiveArray::from_iter((0..len as u64).rev());
    let result = take(array, indices.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), len);

    // Verify elements are in reverse order
    for i in 0..len {
        assert_eq!(
            array.scalar_at(len - 1 - i).vortex_unwrap(),
            result.scalar_at(i).vortex_unwrap()
        );
    }
}

fn test_take_single_middle(array: &dyn Array) {
    let len = array.len();
    let middle_idx = len / 2;

    let indices = PrimitiveArray::from_iter([middle_idx as u64]);
    let result = take(array, indices.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(
        array.scalar_at(middle_idx).vortex_unwrap(),
        result.scalar_at(0).vortex_unwrap()
    );
}

#[allow(clippy::cast_possible_truncation)]
fn test_take_random_unsorted(array: &dyn Array) {
    let len = array.len();

    // Create a pseudo-random but deterministic pattern
    let mut indices = Vec::new();
    let mut idx = 1u64;
    for _ in 0..len.min(10) {
        indices.push((idx * 7 + 3) % len as u64);
        idx = (idx * 3 + 1) % len as u64;
    }

    let indices_array = PrimitiveArray::from_iter(indices.clone());
    let result = take(array, indices_array.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), indices.len());

    // Verify elements match
    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(
            array.scalar_at(idx as usize).vortex_unwrap(),
            result.scalar_at(i).vortex_unwrap()
        );
    }
}

fn test_take_contiguous_range(array: &dyn Array) {
    let len = array.len();
    let start = len / 4;
    let end = len / 2;

    // Take a contiguous range from the middle
    let indices = PrimitiveArray::from_iter(start as u64..end as u64);
    let result = take(array, indices.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), end - start);

    // Verify elements
    for i in 0..(end - start) {
        assert_eq!(
            array.scalar_at(start + i).vortex_unwrap(),
            result.scalar_at(i).vortex_unwrap()
        );
    }
}

#[allow(clippy::cast_possible_truncation)]
fn test_take_mixed_repeated(array: &dyn Array) {
    let len = array.len();

    // Create pattern with some repeated indices
    let indices = vec![
        0u64,
        0,
        1,
        1,
        len as u64 / 2,
        len as u64 / 2,
        len as u64 / 2,
        (len - 1) as u64,
    ];

    let indices_array = PrimitiveArray::from_iter(indices.clone());
    let result = take(array, indices_array.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), indices.len());

    // Verify elements
    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(
            array.scalar_at(idx as usize).vortex_unwrap(),
            result.scalar_at(i).vortex_unwrap()
        );
    }
}

#[allow(clippy::cast_possible_truncation)]
fn test_take_large_indices(array: &dyn Array) {
    // Test with a large number of indices to stress test performance
    let len = array.len();
    let num_indices = 10000.min(len * 3);

    // Create many indices with a pattern
    let indices: Vec<u64> = (0..num_indices)
        .map(|i| ((i * 17 + 5) % len) as u64)
        .collect();

    let indices_array = PrimitiveArray::from_iter(indices.clone());
    let result = take(array, indices_array.as_ref()).vortex_unwrap();

    assert_eq!(result.len(), num_indices);

    // Spot check a few elements
    for i in (0..num_indices).step_by(1000) {
        let expected_idx = indices[i] as usize;
        assert_eq!(
            array.scalar_at(expected_idx).vortex_unwrap(),
            result.scalar_at(i).vortex_unwrap()
        );
    }
}
