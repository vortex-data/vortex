// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray as _;
use crate::arrays::PrimitiveArray;
use crate::dtype::Nullability;

/// Test conformance of the take compute function for an array.
///
/// This function tests various scenarios including:
/// - Taking all elements
/// - Taking no elements
/// - Taking selective elements
/// - Taking with out-of-bounds indices (should panic)
/// - Taking with nullable indices
/// - Edge cases like empty arrays
pub fn test_take_conformance(array: &ArrayRef) {
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

fn test_take_all(array: &ArrayRef) {
    let len = array.len();
    let indices = PrimitiveArray::from_iter(0..len as u64);
    let result = array
        .take(indices.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), len);
    assert_eq!(result.dtype(), array.dtype());

    // Verify elements match
    match (
        array
            .to_canonical()
            .vortex_expect("to_canonical failed on array"),
        result
            .to_canonical()
            .vortex_expect("to_canonical failed on result"),
    ) {
        (Canonical::Primitive(orig_prim), Canonical::Primitive(result_prim)) => {
            assert_eq!(
                orig_prim.buffer_handle().to_host_sync(),
                result_prim.buffer_handle().to_host_sync()
            );
        }
        _ => {
            // For non-primitive types, check scalar values
            for i in 0..len {
                assert_eq!(
                    array
                        .scalar_at(i)
                        .vortex_expect("scalar_at should succeed in conformance test"),
                    result
                        .scalar_at(i)
                        .vortex_expect("scalar_at should succeed in conformance test")
                );
            }
        }
    }
}

fn test_take_none(array: &ArrayRef) {
    let indices: PrimitiveArray = PrimitiveArray::from_iter::<[u64; 0]>([]);
    let result = array
        .take(indices.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), 0);
    assert_eq!(result.dtype(), array.dtype());
}

#[expect(clippy::cast_possible_truncation)]
fn test_take_selective(array: &ArrayRef) {
    let len = array.len();

    // Take every other element
    let indices: Vec<u64> = (0..len as u64).step_by(2).collect();
    let expected_len = indices.len();
    let indices_array = PrimitiveArray::from_iter(indices.clone());

    let result = array
        .take(indices_array.into_array())
        .vortex_expect("take should succeed in conformance test");
    assert_eq!(result.len(), expected_len);

    // Verify the taken elements
    for (result_idx, &original_idx) in indices.iter().enumerate() {
        assert_eq!(
            array
                .scalar_at(original_idx as usize)
                .vortex_expect("scalar_at should succeed in conformance test"),
            result
                .scalar_at(result_idx)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

fn test_take_first_and_last(array: &ArrayRef) {
    let len = array.len();
    let indices = PrimitiveArray::from_iter([0u64, (len - 1) as u64]);
    let result = array
        .take(indices.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), 2);
    assert_eq!(
        array
            .scalar_at(0)
            .vortex_expect("scalar_at should succeed in conformance test"),
        result
            .scalar_at(0)
            .vortex_expect("scalar_at should succeed in conformance test")
    );
    assert_eq!(
        array
            .scalar_at(len - 1)
            .vortex_expect("scalar_at should succeed in conformance test"),
        result
            .scalar_at(1)
            .vortex_expect("scalar_at should succeed in conformance test")
    );
}

#[expect(clippy::cast_possible_truncation)]
fn test_take_with_nullable_indices(array: &ArrayRef) {
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
    let result = array
        .take(indices.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), indices_vec.len());
    assert_eq!(
        result.dtype(),
        &array.dtype().with_nullability(Nullability::Nullable)
    );

    // Verify values
    for (i, idx_opt) in indices_vec.iter().enumerate() {
        match idx_opt {
            Some(idx) => {
                let expected = array
                    .scalar_at(*idx as usize)
                    .vortex_expect("scalar_at should succeed in conformance test");
                let actual = result
                    .scalar_at(i)
                    .vortex_expect("scalar_at should succeed in conformance test");
                assert_eq!(expected, actual);
            }
            None => {
                assert!(
                    result
                        .scalar_at(i)
                        .vortex_expect("scalar_at should succeed in conformance test")
                        .is_null()
                );
            }
        }
    }
}

fn test_take_repeated_indices(array: &ArrayRef) {
    if array.is_empty() {
        return;
    }

    // Take the first element multiple times
    let indices = buffer![0u64, 0, 0].into_array();
    let result = array
        .take(indices)
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), 3);
    let first_elem = array
        .scalar_at(0)
        .vortex_expect("scalar_at should succeed in conformance test");
    for i in 0..3 {
        assert_eq!(
            result
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test"),
            first_elem
        );
    }
}

fn test_empty_indices(array: &ArrayRef) {
    let indices = PrimitiveArray::empty::<u64>(Nullability::NonNullable);
    let result = array
        .take(indices.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), 0);
    assert_eq!(result.dtype(), array.dtype());
}

fn test_take_reverse(array: &ArrayRef) {
    let len = array.len();
    // Take elements in reverse order
    let indices = PrimitiveArray::from_iter((0..len as u64).rev());
    let result = array
        .take(indices.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), len);

    // Verify elements are in reverse order
    for i in 0..len {
        assert_eq!(
            array
                .scalar_at(len - 1 - i)
                .vortex_expect("scalar_at should succeed in conformance test"),
            result
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

fn test_take_single_middle(array: &ArrayRef) {
    let len = array.len();
    let middle_idx = len / 2;

    let indices = PrimitiveArray::from_iter([middle_idx as u64]);
    let result = array
        .take(indices.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), 1);
    assert_eq!(
        array
            .scalar_at(middle_idx)
            .vortex_expect("scalar_at should succeed in conformance test"),
        result
            .scalar_at(0)
            .vortex_expect("scalar_at should succeed in conformance test")
    );
}

#[expect(clippy::cast_possible_truncation)]
fn test_take_random_unsorted(array: &ArrayRef) {
    let len = array.len();

    // Create a pseudo-random but deterministic pattern
    let mut indices = Vec::new();
    let mut idx = 1u64;
    for _ in 0..len.min(10) {
        indices.push((idx * 7 + 3) % len as u64);
        idx = (idx * 3 + 1) % len as u64;
    }

    let indices_array = PrimitiveArray::from_iter(indices.clone());
    let result = array
        .take(indices_array.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), indices.len());

    // Verify elements match
    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(
            array
                .scalar_at(idx as usize)
                .vortex_expect("scalar_at should succeed in conformance test"),
            result
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

fn test_take_contiguous_range(array: &ArrayRef) {
    let len = array.len();
    let start = len / 4;
    let end = len / 2;

    // Take a contiguous range from the middle
    let indices = PrimitiveArray::from_iter(start as u64..end as u64);
    let result = array
        .take(indices.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), end - start);

    // Verify elements
    for i in 0..(end - start) {
        assert_eq!(
            array
                .scalar_at(start + i)
                .vortex_expect("scalar_at should succeed in conformance test"),
            result
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

#[expect(clippy::cast_possible_truncation)]
fn test_take_mixed_repeated(array: &ArrayRef) {
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
    let result = array
        .take(indices_array.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), indices.len());

    // Verify elements
    for (i, &idx) in indices.iter().enumerate() {
        assert_eq!(
            array
                .scalar_at(idx as usize)
                .vortex_expect("scalar_at should succeed in conformance test"),
            result
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}

#[expect(clippy::cast_possible_truncation)]
fn test_take_large_indices(array: &ArrayRef) {
    // Test with a large number of indices to stress test performance
    let len = array.len();
    let num_indices = 10000.min(len * 3);

    // Create many indices with a pattern
    let indices: Vec<u64> = (0..num_indices)
        .map(|i| ((i * 17 + 5) % len) as u64)
        .collect();

    let indices_array = PrimitiveArray::from_iter(indices.clone());
    let result = array
        .take(indices_array.into_array())
        .vortex_expect("take should succeed in conformance test");

    assert_eq!(result.len(), num_indices);

    // Spot check a few elements
    for i in (0..num_indices).step_by(1000) {
        let expected_idx = indices[i] as usize;
        assert_eq!(
            array
                .scalar_at(expected_idx)
                .vortex_expect("scalar_at should succeed in conformance test"),
            result
                .scalar_at(i)
                .vortex_expect("scalar_at should succeed in conformance test")
        );
    }
}
