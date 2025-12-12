// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativePType;
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

#[test]
fn test_null_vector_take() {
    use vortex_vector::VectorOps;
    use vortex_vector::null::NullVector;

    let data = NullVector::new(10);

    // Take with slice indices.
    let result = (&data).take(&[0u32, 5, 9, 2][..]);
    assert_eq!(result.len(), 4);
    assert!(result.validity().all_false());

    // Take with nullable PVector indices.
    let indices: PVector<u32> = PVectorMut::from_iter([Some(0), None, Some(5), None]).freeze();
    let result = (&data).take(&indices);
    assert_eq!(result.len(), 4);
    assert!(result.validity().all_false());
}

#[test]
fn test_dvector_take() {
    use vortex_buffer::buffer;
    use vortex_dtype::PrecisionScale;
    use vortex_mask::Mask;
    use vortex_vector::VectorOps;
    use vortex_vector::decimal::DVector;

    let ps = PrecisionScale::<i64>::new(10, 2);
    let data = DVector::new(
        ps,
        buffer![100i64, 200, 300, 400, 500],
        Mask::from_iter([true, true, false, true, true]),
    );

    // Take with slice indices.
    let result = (&data).take(&[4u32, 2, 0, 1][..]);
    assert_eq!(result.elements().as_slice(), &[500i64, 300, 100, 200]);
    let validity: Vec<bool> = (0..result.len())
        .map(|i| result.validity().value(i))
        .collect();
    assert_eq!(validity, vec![true, false, true, true]);
    assert_eq!(result.precision_scale(), ps);

    // Take with nullable indices.
    let indices: PVector<u32> = PVectorMut::from_iter([Some(0), None, Some(4), None]).freeze();
    let result = (&data).take(&indices);
    let validity: Vec<bool> = (0..result.len())
        .map(|i| result.validity().value(i))
        .collect();
    assert_eq!(validity, vec![true, false, true, false]);
}

#[test]
fn test_decimal_vector_take() {
    use vortex_buffer::buffer;
    use vortex_dtype::PrecisionScale;
    use vortex_mask::Mask;
    use vortex_vector::decimal::DVector;
    use vortex_vector::decimal::DecimalVector;

    let ps = PrecisionScale::<i32>::new(5, 1);
    let data: DecimalVector =
        DVector::new(ps, buffer![10i32, 20, 30, 40, 50], Mask::new_true(5)).into();

    let result = (&data).take(&[4u32, 0, 2][..]);

    let DecimalVector::D32(result) = result else {
        panic!("Expected D32 variant")
    };
    assert_eq!(result.elements().as_slice(), &[50i32, 10, 30]);
}

#[test]
fn test_struct_vector_take() {
    use std::sync::Arc;

    use vortex_mask::Mask;
    use vortex_vector::Vector;
    use vortex_vector::VectorOps;
    use vortex_vector::struct_::StructVector;

    let field1: Vector =
        PVectorMut::from_iter([Some(10i32), Some(20), Some(30), Some(40), Some(50)])
            .freeze()
            .into();
    let field2: Vector =
        BoolVectorMut::from_iter([Some(true), Some(false), Some(true), Some(false), Some(true)])
            .freeze()
            .into();
    let data = StructVector::new(
        Arc::new(vec![field1, field2].into()),
        Mask::from_iter([true, true, false, true, true]),
    );

    // Take with slice indices.
    let result = (&data).take(&[4u32, 2, 0][..]);
    let validity: Vec<bool> = (0..result.len())
        .map(|i| result.validity().value(i))
        .collect();
    assert_eq!(validity, vec![true, false, true]);

    let Vector::Primitive(PrimitiveVector::I32(f0)) = &result.fields()[0] else {
        panic!("Expected I32")
    };
    assert_eq!(f0.elements().as_slice(), &[50i32, 30, 10]);

    // Take with nullable indices.
    let indices: PVector<u32> = PVectorMut::from_iter([Some(0), None, Some(4)]).freeze();
    let result = (&data).take(&indices);
    let validity: Vec<bool> = (0..result.len())
        .map(|i| result.validity().value(i))
        .collect();
    assert_eq!(validity, vec![true, false, true]);
}

#[test]
fn test_fixed_size_list_vector_take() {
    use std::sync::Arc;

    use vortex_mask::Mask;
    use vortex_vector::Vector;
    use vortex_vector::VectorOps;
    use vortex_vector::fixed_size_list::FixedSizeListVector;

    // Elements: [1..12], grouped as 4 lists of size 3.
    let elements: Vector = PVectorMut::from_iter((1..=12).map(Some)).freeze().into();
    let data = FixedSizeListVector::new(
        Arc::new(elements),
        3,
        Mask::from_iter([true, true, false, true]),
    );

    // Take lists [3, 1, 0] -> elements [10,11,12], [4,5,6], [1,2,3].
    let result = (&data).take(&[3u32, 1, 0][..]);
    assert_eq!(result.len(), 3);
    assert_eq!(result.list_size(), 3);

    let validity: Vec<bool> = (0..result.len())
        .map(|i| result.validity().value(i))
        .collect();
    assert_eq!(validity, vec![true, true, true]);

    let Vector::Primitive(PrimitiveVector::I32(elems)) = result.elements().as_ref() else {
        panic!("Expected I32")
    };
    assert_eq!(elems.elements().as_slice(), &[10, 11, 12, 4, 5, 6, 1, 2, 3]);

    // Take with nullable indices.
    let indices: PVector<u32> = PVectorMut::from_iter([Some(2), None, Some(0)]).freeze();
    let result = (&data).take(&indices);
    let validity: Vec<bool> = (0..result.len())
        .map(|i| result.validity().value(i))
        .collect();
    assert_eq!(validity, vec![false, false, true]);
}

#[test]
fn test_fixed_size_list_vector_take_degenerate() {
    use std::sync::Arc;

    use vortex_mask::Mask;
    use vortex_vector::Vector;
    use vortex_vector::VectorOps;
    use vortex_vector::fixed_size_list::FixedSizeListVector;

    // Degenerate FSL with list_size=0.
    let elements: Vector = PVectorMut::<i32>::from_iter(std::iter::empty::<Option<i32>>())
        .freeze()
        .into();
    let data = FixedSizeListVector::new(Arc::new(elements), 0, Mask::new_true(5));

    let result = (&data).take(&[4u32, 0, 2][..]);
    assert_eq!(result.len(), 3);
    assert_eq!(result.list_size(), 0);
    assert!(result.elements().is_empty());
}

#[test]
fn test_vector_enum_take() {
    use vortex_vector::Vector;
    use vortex_vector::VectorOps;

    let data: Vector = PVectorMut::from_iter([Some(100i32), Some(200), None, Some(400), Some(500)])
        .freeze()
        .into();

    let result = (&data).take(&[4u32, 2, 0][..]);

    let Vector::Primitive(PrimitiveVector::I32(result)) = result else {
        panic!("Expected I32")
    };
    assert_eq!(result.elements().as_slice(), &[500i32, 0, 100]);
    let validity: Vec<bool> = (0..result.len())
        .map(|i| result.validity().value(i))
        .collect();
    assert_eq!(validity, vec![true, false, true]);
}
