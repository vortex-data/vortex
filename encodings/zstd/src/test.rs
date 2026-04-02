// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(clippy::cast_possible_truncation)]

use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_mask::Mask;

use crate::Zstd;

#[test]
fn test_zstd_compress_decompress() {
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::from_iter(data.clone());

    let compressed = Zstd::from_primitive(&array, 3, 0).unwrap();
    // this data should be compressible
    assert!(compressed.frames.len() < array.into_array().nbytes() as usize);
    assert!(compressed.dictionary.is_none());

    // check full decompression works
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let decompressed = compressed.decompress(&mut ctx).unwrap();
    assert_arrays_eq!(decompressed, PrimitiveArray::from_iter(data));

    // check slicing works
    let slice = compressed.slice(100..105).unwrap();
    for i in 0_i32..5 {
        assert_nth_scalar!(slice, i as usize, 100 + i);
    }
    assert_arrays_eq!(slice, PrimitiveArray::from_iter([100, 101, 102, 103, 104]));

    let slice = compressed.slice(200..200).unwrap();
    assert_arrays_eq!(slice, PrimitiveArray::from_iter(Vec::<i32>::new()));
}

#[test]
fn test_zstd_empty() {
    let data: Vec<i32> = vec![];
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let compressed = Zstd::from_primitive(&array, 3, 100).unwrap();

    assert_arrays_eq!(compressed, PrimitiveArray::from_iter(data));
}

#[test]
fn test_zstd_with_validity_and_multi_frame() {
    let data: Vec<i32> = (0..200).collect();
    let mut validity: Vec<bool> = vec![false; 200];
    validity[3] = true;
    validity[177] = true;
    let array = PrimitiveArray::new(
        Buffer::from(data),
        Validity::Array(BoolArray::from_iter(validity).into_array()),
    );

    let compressed = Zstd::from_primitive(&array, 0, 30).unwrap();
    assert!(compressed.dictionary.is_none());
    assert_nth_scalar!(compressed, 0, None::<i32>);
    assert_nth_scalar!(compressed, 3, 3);
    assert_nth_scalar!(compressed, 10, None::<i32>);
    assert_nth_scalar!(compressed, 177, 177);

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let decompressed = compressed.decompress(&mut ctx).unwrap().to_primitive();
    let decompressed_values = decompressed.as_slice::<i32>();
    assert_eq!(decompressed_values[3], 3);
    assert_eq!(decompressed_values[177], 177);
    assert!(
        decompressed
            .validity()
            .mask_eq(&array.validity(), &mut ctx)
            .unwrap()
    );

    // check slicing works
    let slice = compressed.slice(176..179).unwrap();
    let primitive = slice.to_primitive();
    assert_eq!(
        i32::try_from(&primitive.scalar_at(1).unwrap()).unwrap(),
        177
    );
    assert!(
        primitive
            .validity()
            .mask_eq(
                &Validity::Array(BoolArray::from_iter(vec![false, true, false]).into_array()),
                &mut ctx
            )
            .unwrap()
    );
}

#[test]
fn test_zstd_with_dict() {
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let compressed = Zstd::from_primitive(&array, 0, 16).unwrap();
    assert!(compressed.dictionary.is_some());
    assert_nth_scalar!(compressed, 0, 0);
    assert_nth_scalar!(compressed, 199, 199);

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let decompressed = compressed.decompress(&mut ctx).unwrap().to_primitive();
    assert_arrays_eq!(decompressed, PrimitiveArray::from_iter(data));

    // check slicing works
    let slice = compressed.slice(176..179).unwrap();
    let primitive = slice.to_primitive();
    assert_arrays_eq!(primitive, PrimitiveArray::from_iter([176, 177, 178]));
}

#[test]
fn test_validity_vtable() {
    let mask_bools = vec![false, true, true, false, true];
    let array = PrimitiveArray::new(
        (0..5).collect::<Buffer<_>>(),
        Validity::Array(BoolArray::from_iter(mask_bools.clone()).into_array()),
    );
    let compressed = Zstd::from_primitive(&array, 3, 0).unwrap();
    assert_eq!(
        compressed.as_array().validity_mask().unwrap(),
        Mask::from_iter(mask_bools)
    );
    assert_eq!(
        compressed.slice(1..4).unwrap().validity_mask().unwrap(),
        Mask::from_iter(vec![true, true, false])
    );
}

#[test]
fn test_zstd_var_bin_view() {
    let data: [Option<&'static [u8]>; 5] = [
        Some(b"foo"),
        Some(b"bar"),
        None,
        Some(b"Lorem ipsum dolor sit amet"),
        Some(b"baz"),
    ];
    let array = VarBinViewArray::from_iter(data, DType::Utf8(Nullability::Nullable));

    let compressed = Zstd::from_var_bin_view(&array, 0, 3).unwrap();
    assert!(compressed.dictionary.is_none());
    assert_nth_scalar!(compressed, 0, "foo");
    assert_nth_scalar!(compressed, 1, "bar");
    assert_nth_scalar!(compressed, 2, None::<String>);
    assert_nth_scalar!(compressed, 3, "Lorem ipsum dolor sit amet");
    assert_nth_scalar!(compressed, 4, "baz");

    let sliced = compressed.slice(1..4).unwrap();
    assert_nth_scalar!(sliced, 0, "bar");
    assert_nth_scalar!(sliced, 1, None::<String>);
    assert_nth_scalar!(sliced, 2, "Lorem ipsum dolor sit amet");
}

#[test]
fn test_zstd_decompress_var_bin_view() {
    let data: [Option<&'static [u8]>; 5] = [
        Some(b"foo"),
        Some(b"bar"),
        None,
        Some(b"Lorem ipsum dolor sit amet"),
        Some(b"baz"),
    ];
    let array = VarBinViewArray::from_iter(data, DType::Utf8(Nullability::Nullable));

    let compressed = Zstd::from_var_bin_view(&array, 0, 3).unwrap();
    assert!(compressed.dictionary.is_none());
    assert_nth_scalar!(compressed, 0, "foo");
    assert_nth_scalar!(compressed, 1, "bar");
    assert_nth_scalar!(compressed, 2, None::<String>);
    assert_nth_scalar!(compressed, 3, "Lorem ipsum dolor sit amet");
    assert_nth_scalar!(compressed, 4, "baz");

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let decompressed = compressed.decompress(&mut ctx).unwrap().to_varbinview();
    assert_nth_scalar!(decompressed, 0, "foo");
    assert_nth_scalar!(decompressed, 1, "bar");
    assert_nth_scalar!(decompressed, 2, None::<String>);
    assert_nth_scalar!(decompressed, 3, "Lorem ipsum dolor sit amet");
    assert_nth_scalar!(decompressed, 4, "baz");
}

#[test]
fn test_sliced_array_children() {
    let data: Vec<Option<i32>> = (0..10).map(|v| (v != 5).then_some(v)).collect();
    let compressed = Zstd::from_primitive(&PrimitiveArray::from_option_iter(data), 0, 100).unwrap();
    let sliced = compressed.slice(0..4).unwrap();
    sliced.children();
}

/// Tests that each beginning of a frame in ZSTD matches
/// the buffer alignment when compressing primitive arrays.
#[test]
fn test_zstd_frame_start_buffer_alignment() {
    let data = vec![0u8; 2];
    let aligned_buffer = Buffer::copy_from_aligned(&data, Alignment::new(8));
    // u8 array now has a 8-byte alignment.
    let array = PrimitiveArray::new(aligned_buffer, Validity::NonNullable);
    let compressed = Zstd::from_primitive(&array, 0, 1);

    assert!(compressed.is_ok());
}
