// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_dtype::NativePType;
use vortex_mask::Mask;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::bool::BoolVector;
use vortex_vector::bool::BoolVectorMut;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PVectorMut;
use vortex_vector::primitive::PrimitiveVector;

use crate::take::Take;

/// Helper to collect a `PVector` into a `Vec<Option<T>>` for easy comparison.
fn collect_pvector<T: NativePType>(v: &PVector<T>) -> Vec<Option<T>> {
    (0..v.len()).map(|i| v.get(i).copied()).collect()
}

/// Helper to collect a `BoolVector` into a `Vec<Option<bool>>` for easy comparison.
fn collect_bool_vector(v: &BoolVector) -> Vec<Option<bool>> {
    (0..v.len()).map(|i| v.get(i)).collect()
}

#[test]
fn test_pvector_take_with_nullable_indices() {
    let data: PVector<i32> =
        PVectorMut::from_iter([Some(10), None, Some(30), Some(40), None, Some(60)]).freeze();
    let indices: PVector<u32> =
        PVectorMut::from_iter([Some(0), None, Some(2), Some(5), None]).freeze();

    let result = data.take(&indices);

    assert_eq!(
        collect_pvector(&result),
        vec![Some(10), None, Some(30), Some(60), None]
    );
}

#[test]
fn test_pvector_take_with_primitive_vector_indices() {
    let data: PVector<i64> =
        PVectorMut::from_iter([Some(100), Some(200), None, Some(400), Some(500)]).freeze();
    let indices: PrimitiveVector = PVectorMut::from_iter([4u16, 2, 0, 1]).freeze().into();

    let result: PVector<i64> = data.take(&indices);

    assert_eq!(
        collect_pvector(&result),
        vec![Some(500), None, Some(100), Some(200)]
    );
}

#[test]
fn test_bool_vector_take_with_nullable_indices() {
    let data: BoolVector =
        BoolVectorMut::from_iter([Some(true), None, Some(false), Some(true), None, Some(false)])
            .freeze();
    let indices: PVector<u32> =
        PVectorMut::from_iter([Some(5), None, Some(0), Some(3), None, Some(2)]).freeze();

    let result = data.take(&indices);

    assert_eq!(
        collect_bool_vector(&result),
        vec![Some(false), None, Some(true), Some(true), None, Some(false)]
    );
}

#[test]
fn test_bool_vector_take_with_primitive_vector_indices() {
    let data: BoolVector =
        BoolVectorMut::from_iter([Some(true), Some(false), None, Some(true), Some(false)]).freeze();
    let indices: PrimitiveVector = PVectorMut::from_iter([4u64, 2, 1, 0, 3]).freeze().into();

    let result: BoolVector = data.take(&indices);

    assert_eq!(
        collect_bool_vector(&result),
        vec![Some(false), None, Some(false), Some(true), Some(true)]
    );
}

#[test]
fn test_bit_buffer_take_small_and_large() {
    // Small buffer (uses take_byte_bool path).
    let small: BitBuffer = [true, false, true, true, false, true, false, false]
        .into_iter()
        .collect();
    let result = small.take(&[7u32, 0, 2, 5, 1][..]);

    let values: Vec<bool> = result.iter().collect();
    assert_eq!(values, vec![false, true, true, true, false]);

    // Large buffer (uses take_bool path, len > 4096).
    let large: BitBuffer = (0..5000).map(|i| i % 3 == 0).collect();
    let result = large.take(&[4999u32, 0, 1, 2, 3, 4998][..]);

    let values: Vec<bool> = result.iter().collect();
    assert_eq!(values, vec![false, true, false, false, true, true]);
}

#[test]
fn test_mask_take_all_variants() {
    // AllTrue with slice indices.
    let result = Mask::AllTrue(10).take(&[9u32, 0, 5][..]);
    assert!(result.all_true());
    assert_eq!(result.len(), 3);

    // AllFalse with slice indices.
    let result = Mask::AllFalse(10).take(&[9u32, 0, 5][..]);
    assert!(result.all_false());
    assert_eq!(result.len(), 3);

    // Values with slice indices.
    let values = Mask::from_iter([true, false, true, true, false, true]);
    let result = values.take(&[5u32, 1, 0, 4][..]);
    let bools: Vec<bool> = (0..result.len()).map(|i| result.value(i)).collect();
    assert_eq!(bools, vec![true, false, true, false]);

    // AllTrue with nullable PVector indices.
    let indices: PVector<u32> =
        PVectorMut::from_iter([Some(0), None, Some(5), None, Some(9)]).freeze();
    let result = Mask::AllTrue(10).take(&indices);
    let bools: Vec<bool> = (0..result.len()).map(|i| result.value(i)).collect();
    assert_eq!(bools, vec![true, false, true, false, true]);

    // AllFalse with nullable PVector indices.
    let result = Mask::AllFalse(10).take(&indices);
    assert!(result.all_false());
    assert_eq!(result.len(), 5);

    // Values with nullable PVector indices.
    let values = Mask::from_iter([true, false, true, false, true, false]);
    let indices: PVector<u32> =
        PVectorMut::from_iter([Some(0), None, Some(1), Some(4), None]).freeze();
    let result = values.take(&indices);
    let bools: Vec<bool> = (0..result.len()).map(|i| result.value(i)).collect();
    assert_eq!(bools, vec![true, false, false, true, false]);
}

#[test]
fn test_primitive_vector_take_with_pvector_indices() {
    let data: PrimitiveVector =
        PVectorMut::from_iter([Some(10i32), Some(20), None, Some(40), Some(50)])
            .freeze()
            .into();
    let indices: PVector<u16> =
        PVectorMut::from_iter([Some(4), None, Some(2), Some(0), None]).freeze();

    let result = data.take(&indices);

    let PrimitiveVector::I32(result) = result else {
        panic!("Expected I32 variant")
    };
    assert_eq!(
        collect_pvector(&result),
        vec![Some(50), None, None, Some(10), None]
    );
}
