// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use num_traits::NumCast;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_native_ptype;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::PcoArrayExt;
use crate::PcoData;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    session.arrays().register(Pco);
    session
});

use crate::Pco;
#[test]
fn test_compress_decompress() {
    let mut ctx = SESSION.create_execution_ctx();
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::from_iter(data.clone());
    let compressed = Pco::from_primitive(array.as_view(), 3, 0, &mut ctx).unwrap();
    // this data should be compressible
    assert!(compressed.pages.len() < array.into_array().nbytes() as usize);

    // check full decompression works
    let unsliced_validity = compressed.unsliced_validity();
    let decompressed = compressed.decompress(&unsliced_validity, &mut ctx).unwrap();
    assert_arrays_eq!(decompressed, PrimitiveArray::from_iter(data), &mut ctx);

    // check slicing works
    let slice = compressed.slice(100..105).unwrap();
    for i in 0_i32..5 {
        assert_nth_scalar!(slice, i as usize, 100 + i, &mut ctx);
    }
    assert_arrays_eq!(
        slice,
        PrimitiveArray::from_iter([100, 101, 102, 103, 104]),
        &mut ctx
    );

    let slice = compressed.slice(200..200).unwrap();
    assert_arrays_eq!(
        slice,
        PrimitiveArray::from_iter(Vec::<i32>::new()),
        &mut ctx
    );
}

#[test]
fn test_compress_decompress_small() {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::from_option_iter([None, Some(1)]);
    let compressed = Pco::from_primitive(array.as_view(), 3, 0, &mut ctx).unwrap();

    let expected = array.into_array();
    assert_arrays_eq!(compressed, expected, &mut ctx);

    let unsliced_validity = compressed.unsliced_validity();
    let decompressed = compressed.decompress(&unsliced_validity, &mut ctx).unwrap();
    assert_arrays_eq!(decompressed, expected, &mut ctx);
}

#[test]
fn test_empty() {
    let mut ctx = SESSION.create_execution_ctx();
    let data: Vec<i32> = vec![];
    let array = PrimitiveArray::from_iter(data.clone());
    let compressed = Pco::from_primitive(array.as_view(), 3, 100, &mut ctx).unwrap();
    let unsliced_validity = compressed.unsliced_validity();
    let primitive = compressed.decompress(&unsliced_validity, &mut ctx).unwrap();
    assert_arrays_eq!(primitive, PrimitiveArray::from_iter(data), &mut ctx);
}

#[test]
fn test_validity_and_multiple_chunks_and_pages() {
    let mut ctx = SESSION.create_execution_ctx();
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
            array.as_view(),
            compression_level,
            values_per_chunk,
            values_per_page,
            &mut ctx,
        )
        .unwrap(),
        validity,
    )
    .vortex_expect("PcoData is always valid");

    assert_eq!(compressed.metadata.chunks.len(), 6); // 191 values / 33 rounds up to 6
    assert_eq!(compressed.metadata.chunks[0].pages.len(), 4); // 33 / 10 rounds up to 4
    assert_nth_scalar!(compressed, 0, 0, &mut ctx);
    assert_nth_scalar!(compressed, 3, 3, &mut ctx);
    assert_nth_scalar!(compressed, 7, None::<i32>, &mut ctx);
    assert_nth_scalar!(compressed, 14, None::<i32>, &mut ctx);
    assert_nth_scalar!(compressed, 15, 15, &mut ctx);
    assert_nth_scalar!(compressed, 101, None::<i32>, &mut ctx);
    assert_nth_scalar!(compressed, 199, 199, &mut ctx);

    // check slicing works
    let slice = compressed.slice(100..103).unwrap();
    assert_nth_scalar!(slice, 0, 100, &mut ctx);
    assert_nth_scalar!(slice, 2, 102, &mut ctx);
    let primitive = slice.execute::<PrimitiveArray>(&mut ctx).unwrap();

    assert!(
        primitive
            .validity()
            .unwrap()
            .mask_eq(
                &Validity::Array(BoolArray::from_iter(vec![true, false, true]).into_array()),
                primitive.len(),
                &mut ctx,
            )
            .unwrap()
    );
}

#[test]
fn test_validity_vtable() {
    let mut ctx = SESSION.create_execution_ctx();
    let data: Vec<i32> = (0..5).collect();
    let mask_bools = vec![false, true, true, false, true];
    let array = PrimitiveArray::new(
        Buffer::from(data),
        Validity::Array(BoolArray::from_iter(mask_bools.clone()).into_array()),
    );
    let compressed = Pco::from_primitive(array.as_view(), 3, 0, &mut ctx).unwrap();
    let arr = compressed.as_array();
    assert_eq!(
        arr.validity()
            .unwrap()
            .execute_mask(arr.len(), &mut ctx)
            .unwrap(),
        Mask::from_iter(mask_bools)
    );
    let sliced = compressed.slice(1..4).unwrap();
    assert_eq!(
        sliced
            .validity()
            .unwrap()
            .execute_mask(sliced.len(), &mut ctx)
            .unwrap(),
        Mask::from_iter(vec![true, true, false])
    );
}

#[test]
fn test_serde() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let data: PrimitiveArray = (0i32..1_000_000).collect();
    let pco = Pco::from_primitive(data.as_view(), 3, 100, &mut ctx)?.into_array();

    let context = ArrayContext::empty();

    let bytes = pco
        .serialize(
            &context,
            &SESSION,
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
    let data_type = SESSION.arrow().to_arrow_field("", data.dtype())?;
    let pco_arrow = SESSION
        .arrow()
        .execute_arrow(pco, Some(&data_type), &mut ctx)?;
    let decoded_arrow = SESSION
        .arrow()
        .execute_arrow(decoded, Some(&data_type), &mut ctx)?;
    assert!(pco_arrow == decoded_arrow);
    Ok(())
}

/// Round-trip `values` through Pco compression, checking both full decompression and a slice.
fn assert_pco_roundtrip<T: NativePType>(values: Vec<T>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = PrimitiveArray::from_iter(values.clone());
    let compressed = Pco::from_primitive(array.as_view(), 3, 0, &mut ctx)?;

    let unsliced_validity = compressed.unsliced_validity();
    let decompressed = compressed.decompress(&unsliced_validity, &mut ctx)?;
    assert_arrays_eq!(
        decompressed,
        PrimitiveArray::from_iter(values.clone()),
        &mut ctx
    );

    let slice = compressed.slice(1..values.len() - 1)?;
    assert_arrays_eq!(
        slice,
        PrimitiveArray::from_iter(values[1..values.len() - 1].to_vec()),
        &mut ctx
    );
    Ok(())
}

/// Round-trip a nullable array of `values` with every third entry null.
fn assert_pco_nullable_roundtrip<T: NativePType>(values: Vec<T>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let options = values
        .iter()
        .enumerate()
        .map(|(i, v)| (i % 3 != 0).then_some(*v))
        .collect::<Vec<_>>();
    let array = PrimitiveArray::from_option_iter(options.clone());
    let compressed = Pco::from_primitive(array.as_view(), 3, 0, &mut ctx)?;

    assert_arrays_eq!(
        compressed,
        PrimitiveArray::from_option_iter(options),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_roundtrip_u8() -> VortexResult<()> {
    let values: Vec<u8> = (0..=u8::MAX).collect();
    assert_pco_roundtrip(values.clone())?;
    assert_pco_nullable_roundtrip(values)
}

#[test]
fn test_roundtrip_i8() -> VortexResult<()> {
    let values: Vec<i8> = (i8::MIN..=i8::MAX).collect();
    assert_pco_roundtrip(values.clone())?;
    assert_pco_nullable_roundtrip(values)
}

#[rstest]
#[case(PType::U8)]
#[case(PType::U16)]
#[case(PType::U32)]
#[case(PType::U64)]
#[case(PType::I8)]
#[case(PType::I16)]
#[case(PType::I32)]
#[case(PType::I64)]
#[case(PType::F16)]
#[case(PType::F32)]
#[case(PType::F64)]
fn test_roundtrip_each_ptype(#[case] ptype: PType) -> VortexResult<()> {
    match_each_native_ptype!(ptype, |T| {
        let values = (0..100_u8)
            .map(|i| <T as NumCast>::from(i).vortex_expect("0..100 fits in every native ptype"))
            .collect::<Vec<T>>();
        assert_pco_roundtrip(values)
    })
}
