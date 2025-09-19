// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(clippy::cast_possible_truncation)]

use vortex_array::arrays::{BoolArray, PrimitiveArray};
use vortex_array::arrow::compute::to_arrow_preferred;
use vortex_array::serde::{ArrayParts, SerializeOptions};
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_array::{ArrayContext, ArrayRegistry, EncodingRef, ToCanonical};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{DType, Nullability, PType};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::{PcoArray, PcoEncoding};

macro_rules! assert_nth_scalar {
    ($arr:expr, $n:expr, $expected:expr) => {
        assert_eq!($arr.scalar_at($n), $expected.try_into().unwrap());
    };
}

#[test]
fn test_compress_decompress() {
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::from_iter(data.clone());
    let compressed = PcoArray::from_primitive(&array, 3, 0).unwrap();
    // this data should be compressible
    assert!(compressed.pages.len() < array.nbytes() as usize);

    // check full decompression works
    let decompressed = compressed.decompress();
    assert_eq!(decompressed.as_slice::<i32>(), &data);

    // check slicing works
    let slice = compressed.slice(100..105);
    for i in 0_i32..5 {
        assert_nth_scalar!(slice, i as usize, 100 + i);
    }
    let primitive = slice.to_primitive();
    assert_eq!(primitive.as_slice::<i32>(), &[100, 101, 102, 103, 104]);

    let slice = compressed.slice(200..200);
    let primitive = slice.to_primitive();
    assert_eq!(primitive.as_slice::<i32>(), &Vec::<i32>::new());
}

#[test]
fn test_compress_decompress_small() {
    let array = PrimitiveArray::from_option_iter([None, Some(1)]);
    let compressed = PcoArray::from_primitive(&array, 3, 0).unwrap();
    assert_eq!(compressed.scalar_at(0), Scalar::null_typed::<i32>());
    assert_eq!(compressed.scalar_at(1), Scalar::from(Some(1)));
    let decompressed = compressed.decompress();
    assert_eq!(decompressed.scalar_at(0), Scalar::null_typed::<i32>());
    assert_eq!(decompressed.scalar_at(1), Scalar::from(Some(1)));
}

#[test]
fn test_empty() {
    let data: Vec<i32> = vec![];
    let array = PrimitiveArray::from_iter(data.clone());
    let compressed = PcoArray::from_primitive(&array, 3, 100).unwrap();
    let primitive = compressed.decompress();
    assert_eq!(primitive.as_slice::<i32>(), &data);
}

#[test]
fn test_validity_and_multiple_chunks_and_pages() {
    let data: Vec<i32> = (0..200).collect();
    let mut validity: Vec<bool> = vec![true; 200];
    validity[7..15].fill(false);
    validity[101] = false;
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::Array(BoolArray::from_iter(validity).to_array()),
    );
    let compression_level = 3;
    let values_per_chunk = 33;
    let values_per_page = 10;
    let compressed = PcoArray::from_primitive_with_values_per_chunk(
        &array,
        compression_level,
        values_per_chunk,
        values_per_page,
    )
    .unwrap();

    assert_eq!(compressed.metadata.chunks.len(), 6); // 191 values / 33 rounds up to 6
    assert_eq!(compressed.metadata.chunks[0].pages.len(), 4); // 33 / 10 rounds up to 4
    assert_nth_scalar!(compressed, 0, 0);
    assert_nth_scalar!(compressed, 3, 3);
    assert_nth_scalar!(compressed, 7, None::<i32>);
    assert_nth_scalar!(compressed, 14, None::<i32>);
    assert_nth_scalar!(compressed, 15, 15);
    assert_nth_scalar!(compressed, 101, None::<i32>);
    assert_nth_scalar!(compressed, 199, 199);

    // check slicing works
    let slice = compressed.slice(100..103);
    assert_nth_scalar!(slice, 0, 100);
    assert_nth_scalar!(slice, 2, 102);
    let primitive = slice.to_primitive();
    assert_eq!(
        primitive.validity(),
        &Validity::Array(BoolArray::from_iter(vec![true, false, true]).to_array())
    );
}

#[test]
fn test_validity_vtable() {
    let data: Vec<i32> = (0..5).collect();
    let mask_bools = vec![false, true, true, false, true];
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::Array(BoolArray::from_iter(mask_bools.clone()).to_array()),
    );
    let compressed = PcoArray::from_primitive(&array, 3, 0).unwrap();
    assert_eq!(compressed.validity_mask(), Mask::from_iter(mask_bools));
    assert_eq!(
        compressed.slice(1..4).validity_mask(),
        Mask::from_iter(vec![true, true, false])
    );
}

#[test]
fn test_serde() {
    let data: BufferMut<i32> = (0..1_000_000).collect();
    let pco = PcoArray::from_primitive(&PrimitiveArray::new(data, Validity::NonNullable), 3, 100)
        .unwrap()
        .to_array();
    let context = ArrayContext::empty().with_many(
        ArrayRegistry::canonical_only()
            .vtables()
            .cloned()
            .chain([EncodingRef::new_ref(PcoEncoding.as_ref())]),
    );
    let bytes = pco
        .serialize(
            &context,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )
        .unwrap()
        .into_iter()
        .flat_map(|x| x.into_iter())
        .collect::<BufferMut<u8>>()
        .freeze();

    let parts = ArrayParts::try_from(bytes).unwrap();
    let decoded = parts
        .decode(
            &context,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
            1_000_000,
        )
        .unwrap();
    assert_eq!(
        &to_arrow_preferred(&pco).unwrap(),
        &to_arrow_preferred(&decoded).unwrap()
    );
}
