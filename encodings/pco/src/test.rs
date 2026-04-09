// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrow::ArrowArrayExecutor;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySession;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::PcoData;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<ArraySession>();
    session.arrays().register(Pco);
    session
});

use crate::Pco;
#[test]
fn test_compress_decompress() {
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::from_iter(data.clone());
    let compressed = Pco::from_primitive(&array, 3, 0).unwrap();
    // this data should be compressible
    assert!(compressed.pages.len() < array.into_array().nbytes() as usize);

    // check full decompression works
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let unsliced_validity = child_to_validity(
        &compressed.as_ref().slots()[0],
        compressed.dtype().nullability(),
    );
    let decompressed = compressed.decompress(&unsliced_validity, &mut ctx).unwrap();
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
fn test_compress_decompress_small() {
    let array = PrimitiveArray::from_option_iter([None, Some(1)]);
    let compressed = Pco::from_primitive(&array, 3, 0).unwrap();

    let expected = array.into_array();
    assert_arrays_eq!(compressed, expected);

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let unsliced_validity = child_to_validity(
        &compressed.as_ref().slots()[0],
        compressed.dtype().nullability(),
    );
    let decompressed = compressed.decompress(&unsliced_validity, &mut ctx).unwrap();
    assert_arrays_eq!(decompressed, expected);
}

#[test]
fn test_empty() {
    let data: Vec<i32> = vec![];
    let array = PrimitiveArray::from_iter(data.clone());
    let compressed = Pco::from_primitive(&array, 3, 100).unwrap();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let unsliced_validity = child_to_validity(
        &compressed.as_ref().slots()[0],
        compressed.dtype().nullability(),
    );
    let primitive = compressed.decompress(&unsliced_validity, &mut ctx).unwrap();
    assert_arrays_eq!(primitive, PrimitiveArray::from_iter(data));
}

#[test]
fn test_validity_and_multiple_chunks_and_pages() {
    let data: Vec<i32> = (0..200).collect();
    let mut validity: Vec<bool> = vec![true; 200];
    validity[7..15].fill(false);
    validity[101] = false;
    let array = PrimitiveArray::new(
        Buffer::from(data),
        Validity::Array(BoolArray::from_iter(validity).into_array()),
    );
    let compression_level = 3;
    let values_per_chunk = 33;
    let values_per_page = 10;
    let validity = array.validity().unwrap();
    let compressed = Pco::try_new(
        array.dtype().clone(),
        PcoData::from_primitive_with_values_per_chunk(
            &array,
            compression_level,
            values_per_chunk,
            values_per_page,
        )
        .unwrap(),
        validity,
    )
    .vortex_expect("PcoData is always valid");

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
    let slice = compressed.slice(100..103).unwrap();
    assert_nth_scalar!(slice, 0, 100);
    assert_nth_scalar!(slice, 2, 102);
    let primitive = slice.to_primitive();

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    assert!(
        primitive
            .validity()
            .unwrap()
            .mask_eq(
                &Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array()),
                &mut ctx,
            )
            .unwrap()
    );
}

#[test]
fn test_validity_vtable() {
    let data: Vec<i32> = (0..5).collect();
    let mask_bools = vec![false, true, true, false, true];
    let array = PrimitiveArray::new(
        Buffer::from(data),
        Validity::Array(BoolArray::from_iter(mask_bools.clone()).into_array()),
    );
    let compressed = Pco::from_primitive(&array, 3, 0).unwrap();
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
fn test_serde() -> VortexResult<()> {
    let data: PrimitiveArray = (0i32..1_000_000).collect();
    let pco = Pco::from_primitive(&data, 3, 100)?.into_array();

    let context = ArrayContext::empty();

    let bytes = pco
        .serialize(
            &context,
            &LEGACY_SESSION,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )?
        .into_iter()
        .flat_map(|x| x.into_iter())
        .collect::<BufferMut<u8>>()
        .freeze();

    let parts = SerializedArray::try_from(bytes)?;
    let decoded = parts.decode(
        &DType::Primitive(PType::I32, Nullability::NonNullable),
        1_000_000,
        &ReadContext::new(context.to_ids()),
        &SESSION,
    )?;
    let mut ctx = SESSION.create_execution_ctx();
    let data_type = data.dtype().to_arrow_dtype()?;
    let pco_arrow = pco.execute_arrow(Some(&data_type), &mut ctx)?;
    let decoded_arrow = decoded.execute_arrow(Some(&data_type), &mut ctx)?;
    assert!(pco_arrow == decoded_arrow);
    Ok(())
}
