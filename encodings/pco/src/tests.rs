// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![expect(clippy::cast_possible_truncation)]

use std::iter;
use std::ops::Range;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::runend::RunEnd;
use vortex_array::arrays::runend::RunEndArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
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
    let session = VortexSession::empty();
    session.arrays().register(Pco);
    session
});

use crate::Pco;

#[rstest]
#[case(0..0)]
#[case(0..1)]
#[case(1..2)]
#[case(0..3)]
#[case(3..6)]
#[case(2..4)]
#[case(379..389)]
#[case(1536..1539)]
#[case(1538..1539)]
#[case(1539..1542)]
#[case(0..3070)]
#[case(3069..3070)]
#[case(3070..3070)]
fn runend_slice_with_pco_ends(
    #[case] range: Range<usize>,
    #[values(false, true)] nested: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ends = PrimitiveArray::from_iter((1..=1024u32).map(|run| run * 3));
    let ends = Pco::from_primitive(ends.as_view(), 3, 128, &mut ctx)?.into_array();
    let array = RunEnd::try_new(
        ends,
        PrimitiveArray::from_iter(0..1024i32).into_array(),
        &mut ctx,
    )?;
    let offset = usize::from(nested);
    // Execute the outer slice to retain its nonzero run-end offset for the inner slice.
    let array = array
        .slice(offset..3072 - offset)?
        .execute_until::<RunEnd>(&mut ctx)?;
    assert_eq!(array.as_::<RunEnd>().offset(), offset);
    let expected = PrimitiveArray::from_iter((0..1024i32).flat_map(|run| iter::repeat_n(run, 3)));
    assert_arrays_eq!(
        array.slice(range.clone())?,
        expected.slice(offset + range.start..offset + range.end)?,
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_compress_decompress() {
    let mut ctx = SESSION.create_execution_ctx();
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::from_iter(data.clone());
    let compressed = Pco::from_primitive(array.as_view(), 3, 0, &mut ctx).unwrap();
    // this data should be compressible
    let compressed_bytes: usize = compressed.pages.iter().map(|page| page.len()).sum();
    assert!(compressed_bytes < array.into_array().nbytes() as usize);

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

    assert_arrays_eq!(compressed, array, &mut ctx);

    // check slicing works
    let slice = compressed.slice(100..103).unwrap();
    assert_nth_scalar!(slice, 0, 100, &mut ctx);
    assert_nth_scalar!(slice, 2, 102, &mut ctx);

    assert_arrays_eq!(
        slice,
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(102)]),
        &mut ctx
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
    let expected_arrow =
        SESSION
            .arrow()
            .execute_arrow(data.into_array(), Some(&data_type), &mut ctx)?;
    assert_eq!(pco_arrow.as_ref(), expected_arrow.as_ref());
    assert_eq!(decoded_arrow.as_ref(), expected_arrow.as_ref());
    Ok(())
}

#[test]
fn sliced_pco_cannot_be_serialized() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter(0..100_i32);
    let array = Pco::from_primitive(values.as_view(), 3, 0, &mut ctx)?.into_array();
    let context = ArrayContext::empty();
    let options = SerializeOptions::default();

    array.serialize(&context, &SESSION, &options)?;
    let slice = array.slice(10..20)?;
    assert!(slice.serialize(&context, &SESSION, &options).is_err());
    Ok(())
}
