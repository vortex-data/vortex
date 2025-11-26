// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_vector::VectorMutOps;
use vortex_vector::bool::BoolVector;
use vortex_vector::bool::BoolVectorMut;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PVectorMut;
use vortex_vector::primitive::PrimitiveVector;

use crate::take::Take;

/// Tests `Take` on `PVector` with mixed validity in both data and indices.
///
/// This test covers:
/// - Taking from a vector with some null values.
/// - Using indices that have some null values.
/// - Verifying that nulls in the data propagate correctly.
/// - Verifying that nulls in indices result in null outputs.
/// - Using non-sequential indices to test the general case.
#[test]
fn test_pvector_take_with_nullable_indices() {
    // Data: [10, null, 30, 40, null, 60]
    let data: PVectorMut<i32> = [Some(10), None, Some(30), Some(40), None, Some(60)]
        .into_iter()
        .collect();
    let data = data.freeze();

    // Indices: [0, null, 2, 5, null] (u32 indices with nulls)
    let indices: PVectorMut<u32> = [Some(0), None, Some(2), Some(5), None]
        .into_iter()
        .collect();
    let indices = indices.freeze();

    let result = (&data).take(&indices);

    // Expected: [10, null, 30, 60, null]
    // - Index 0 -> data[0] = 10 (valid)
    // - Index null -> null (null index produces null)
    // - Index 2 -> data[2] = 30 (valid)
    // - Index 5 -> data[5] = 60 (valid)
    // - Index null -> null (null index produces null)
    assert_eq!(result.get(0), Some(&10));
    assert_eq!(result.get(1), None); // Null index.
    assert_eq!(result.get(2), Some(&30));
    assert_eq!(result.get(3), Some(&60));
    assert_eq!(result.get(4), None); // Null index.
}

/// Tests `Take` on `PVector` using `PrimitiveVector` indices (type-erased).
///
/// This ensures the generic `Take<PrimitiveVector>` impl works correctly by dispatching to the
/// typed implementation.
#[test]
fn test_pvector_take_with_primitive_vector_indices() {
    // Data: [100, 200, null, 400, 500]
    let data: PVectorMut<i64> = [Some(100), Some(200), None, Some(400), Some(500)]
        .into_iter()
        .collect();
    let data = data.freeze();

    // Indices as PrimitiveVector (u16): [4, 2, 0, 1]
    let indices: PVectorMut<u16> = [4u16, 2, 0, 1].into_iter().collect();
    let indices: PrimitiveVector = indices.freeze().into();

    let result: PVector<i64> = (&data).take(&indices);

    // Expected: [500, null, 100, 200]
    assert_eq!(result.get(0), Some(&500));
    assert_eq!(result.get(1), None); // data[2] is null.
    assert_eq!(result.get(2), Some(&100));
    assert_eq!(result.get(3), Some(&200));
}

/// Tests `Take` on `BoolVector` with mixed validity in both data and indices.
///
/// This test covers:
/// - Taking from a boolean vector with some null values.
/// - Using indices that have some null values.
/// - Verifying that nulls in the data propagate correctly.
/// - Verifying that nulls in indices result in null outputs.
#[test]
fn test_bool_vector_take_with_nullable_indices() {
    // Data: [true, null, false, true, null, false]
    let data: BoolVectorMut = [Some(true), None, Some(false), Some(true), None, Some(false)]
        .into_iter()
        .collect();
    let data = data.freeze();

    // Indices: [5, null, 0, 3, null, 2] (u32 indices with nulls)
    let indices: PVectorMut<u32> = [Some(5), None, Some(0), Some(3), None, Some(2)]
        .into_iter()
        .collect();
    let indices = indices.freeze();

    let result = (&data).take(&indices);

    // Expected: [false, null, true, true, null, false]
    // - Index 5 -> data[5] = false (valid)
    // - Index null -> null (null index produces null)
    // - Index 0 -> data[0] = true (valid)
    // - Index 3 -> data[3] = true (valid)
    // - Index null -> null (null index produces null)
    // - Index 2 -> data[2] = false (valid)
    assert_eq!(result.get(0), Some(false));
    assert_eq!(result.get(1), None); // Null index.
    assert_eq!(result.get(2), Some(true));
    assert_eq!(result.get(3), Some(true));
    assert_eq!(result.get(4), None); // Null index.
    assert_eq!(result.get(5), Some(false));
}

/// Tests `Take` on `BoolVector` using `PrimitiveVector` indices (type-erased).
///
/// This ensures the generic `Take<PrimitiveVector>` impl works correctly for `BoolVector`.
#[test]
fn test_bool_vector_take_with_primitive_vector_indices() {
    // Data: [true, false, null, true, false]
    let data: BoolVectorMut = [Some(true), Some(false), None, Some(true), Some(false)]
        .into_iter()
        .collect();
    let data = data.freeze();

    // Indices as PrimitiveVector (u64): [4, 2, 1, 0, 3]
    let indices: PVectorMut<u64> = [4u64, 2, 1, 0, 3].into_iter().collect();
    let indices: PrimitiveVector = indices.freeze().into();

    let result: BoolVector = (&data).take(&indices);

    // Expected: [false, null, false, true, true]
    assert_eq!(result.get(0), Some(false)); // data[4]
    assert_eq!(result.get(1), None); // data[2] is null.
    assert_eq!(result.get(2), Some(false)); // data[1]
    assert_eq!(result.get(3), Some(true)); // data[0]
    assert_eq!(result.get(4), Some(true)); // data[3]
}

/// Tests `Take` on `BitBuffer` covering both code paths.
///
/// This test covers:
/// - The `take_byte_bool` path (small buffer, len <= 4096).
/// - The `take_bool` path (large buffer, len > 4096).
/// - Non-sequential indices to verify correct bit extraction.
#[test]
fn test_bit_buffer_take_small_and_large() {
    use vortex_buffer::BitBuffer;

    // Small buffer (uses take_byte_bool path).
    let small: BitBuffer = [true, false, true, true, false, true, false, false]
        .into_iter()
        .collect();
    let indices = [7u32, 0, 2, 5, 1];
    let result = (&small).take(&indices[..]);

    assert_eq!(result.len(), 5);
    assert!(!result.value(0)); // small[7] = false
    assert!(result.value(1)); // small[0] = true
    assert!(result.value(2)); // small[2] = true
    assert!(result.value(3)); // small[5] = true
    assert!(!result.value(4)); // small[1] = false

    // Large buffer (uses take_bool path, len > 4096).
    let large: BitBuffer = (0..5000).map(|i| i % 3 == 0).collect();
    let indices = [4999u32, 0, 1, 2, 3, 4998];
    let result = (&large).take(&indices[..]);

    assert_eq!(result.len(), 6);
    assert!(!result.value(0)); // 4999 % 3 != 0
    assert!(result.value(1)); // 0 % 3 == 0
    assert!(!result.value(2)); // 1 % 3 != 0
    assert!(!result.value(3)); // 2 % 3 != 0
    assert!(result.value(4)); // 3 % 3 == 0
    assert!(result.value(5)); // 4998 % 3 == 0
}

/// Tests `Take` on `Mask` covering all mask variants and nullable index handling.
///
/// This test covers:
/// - `Mask::AllTrue` with slice indices.
/// - `Mask::AllFalse` with slice indices.
/// - `Mask::Values` with slice indices.
/// - `Mask::AllTrue` with nullable `PVector` indices (returns cloned validity).
/// - `Mask::AllFalse` with nullable `PVector` indices.
/// - `Mask::Values` with nullable `PVector` indices (both small and large paths).
#[test]
fn test_mask_take_all_variants() {
    use vortex_mask::Mask;

    // Test AllTrue with slice indices.
    let all_true = Mask::AllTrue(10);
    let indices = [9u32, 0, 5];
    let result = (&all_true).take(&indices[..]);
    assert!(result.all_true());
    assert_eq!(result.len(), 3);

    // Test AllFalse with slice indices.
    let all_false = Mask::AllFalse(10);
    let result = (&all_false).take(&indices[..]);
    assert!(result.all_false());
    assert_eq!(result.len(), 3);

    // Test Values with slice indices.
    let values = Mask::from_iter([true, false, true, true, false, true]);
    let indices = [5u32, 1, 0, 4];
    let result = (&values).take(&indices[..]);
    assert_eq!(result.len(), 4);
    assert!(result.value(0)); // values[5] = true
    assert!(!result.value(1)); // values[1] = false
    assert!(result.value(2)); // values[0] = true
    assert!(!result.value(3)); // values[4] = false

    // Test AllTrue with nullable PVector indices (some indices are null).
    // When self is AllTrue, result validity equals indices validity.
    let all_true = Mask::AllTrue(10);
    let indices: PVectorMut<u32> = [Some(0), None, Some(5), None, Some(9)]
        .into_iter()
        .collect();
    let indices = indices.freeze();
    let result = (&all_true).take(&indices);
    assert_eq!(result.len(), 5);
    assert!(result.value(0)); // Valid index.
    assert!(!result.value(1)); // Null index -> false.
    assert!(result.value(2)); // Valid index.
    assert!(!result.value(3)); // Null index -> false.
    assert!(result.value(4)); // Valid index.

    // Test AllFalse with nullable PVector indices (result is always AllFalse).
    let all_false = Mask::AllFalse(10);
    let result = (&all_false).take(&indices);
    assert!(result.all_false());
    assert_eq!(result.len(), 5);

    // Test Values with nullable PVector indices.
    // Combines mask values with index nullability.
    let values = Mask::from_iter([true, false, true, false, true, false]);
    let indices: PVectorMut<u32> = [Some(0), None, Some(1), Some(4), None]
        .into_iter()
        .collect();
    let indices = indices.freeze();
    let result = (&values).take(&indices);
    assert_eq!(result.len(), 5);
    assert!(result.value(0)); // values[0] = true, index valid.
    assert!(!result.value(1)); // Null index -> false.
    assert!(!result.value(2)); // values[1] = false.
    assert!(result.value(3)); // values[4] = true, index valid.
    assert!(!result.value(4)); // Null index -> false.
}

/// Tests `Take` on `PrimitiveVector` using `PVector` indices.
///
/// This ensures the type-erased `PrimitiveVector` can be taken with typed `PVector` indices,
/// covering the `Take<PVector<I>> for &PrimitiveVector` implementation.
#[test]
fn test_primitive_vector_take_with_pvector_indices() {
    // Data as PrimitiveVector (i32): [10, 20, null, 40, 50]
    let data: PVectorMut<i32> = [Some(10), Some(20), None, Some(40), Some(50)]
        .into_iter()
        .collect();
    let data: PrimitiveVector = data.freeze().into();

    // Indices as PVector<u16> with some nulls: [4, null, 2, 0, null]
    let indices: PVectorMut<u16> = [Some(4), None, Some(2), Some(0), None]
        .into_iter()
        .collect();
    let indices = indices.freeze();

    let result = (&data).take(&indices);

    // Expected: [50, null, null, 10, null]
    // - Index 4 -> data[4] = 50 (valid)
    // - Index null -> null (null index produces null)
    // - Index 2 -> data[2] = null (data is null)
    // - Index 0 -> data[0] = 10 (valid)
    // - Index null -> null (null index produces null)
    let PrimitiveVector::I32(result) = result else {
        panic!("Expected I32 variant");
    };
    assert_eq!(result.get(0), Some(&50));
    assert_eq!(result.get(1), None); // Null index.
    assert_eq!(result.get(2), None); // data[2] is null.
    assert_eq!(result.get(3), Some(&10));
    assert_eq!(result.get(4), None); // Null index.
}
