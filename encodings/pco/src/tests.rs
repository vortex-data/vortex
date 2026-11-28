// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![expect(clippy::cast_possible_truncation)]

use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::compute::conformance::hegel::EncodingCheckOptions;
use vortex_array::compute::conformance::hegel::check_encoding_roundtrip_and_ops;
use vortex_array::compute::conformance::hegel::draw_validity_for;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
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
    crate::initialize(&session);
    session
});

use vortex_arrow::ExecuteArrowExt;

use crate::Pco;
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
    let pco_arrow = pco.execute_arrow(Some(&data_type), &mut ctx)?;
    let decoded_arrow = decoded.execute_arrow(Some(&data_type), &mut ctx)?;
    let expected_arrow = data
        .into_array()
        .execute_arrow(Some(&data_type), &mut ctx)?;
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

#[test]
fn test_f16_roundtrip_and_slice() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([
        f16::from_f32(0.0),
        f16::from_f32(-0.0),
        f16::from_f32(1.5),
        f16::from_f32(-42.25),
        f16::from_bits(1), // smallest positive subnormal
        f16::MAX,
        f16::MIN,
        f16::INFINITY,
        f16::NEG_INFINITY,
    ]);
    let compressed = Pco::from_primitive(values.as_view(), 3, 4, &mut ctx)?;
    assert_eq!(
        compressed.dtype(),
        &DType::Primitive(PType::F16, Nullability::NonNullable)
    );
    assert_arrays_eq!(&compressed, &values, &mut ctx);

    let sliced = compressed.slice(2..7)?;
    assert_arrays_eq!(sliced, values.clone().into_array().slice(2..7)?, &mut ctx);

    let mut probe = compressed.into_array().repeated_probe();
    for i in (0..values.len()).rev() {
        assert_eq!(
            probe.execute_scalar(i, &mut ctx)?,
            values.execute_scalar(i, &mut ctx)?,
            "f16 row {i}"
        );
    }
    Ok(())
}

#[test]
fn test_f16_nullable_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([
        Some(f16::from_f32(-12.5)),
        None,
        Some(f16::from_f32(0.25)),
        Some(f16::from_f32(100.0)),
        None,
        Some(f16::from_f32(3.5)),
    ]);
    let compressed = Pco::from_primitive(values.as_view(), 3, 2, &mut ctx)?;
    assert_arrays_eq!(&compressed, &values, &mut ctx);
    assert_arrays_eq!(
        compressed.slice(1..5)?,
        values.into_array().slice(1..5)?,
        &mut ctx
    );
    Ok(())
}

fn draw_pco_ints<T: NativePType + gs::Integer + Ord>(
    tc: &TestCase,
    len: usize,
    validity: Validity,
    sorted: bool,
) -> PrimitiveArray {
    let mut v = tc.draw(gs::vecs(gs::integers::<T>()).min_size(len).max_size(len));
    if sorted {
        v.sort_unstable();
    }
    PrimitiveArray::new(Buffer::from(v), validity)
}

fn draw_pco_floats<T: NativePType + Copy + Debug>(
    tc: &TestCase,
    len: usize,
    validity: Validity,
    sorted: bool,
    specials: &[T],
    draw_general: impl Fn(&TestCase) -> T,
    cmp: fn(&T, &T) -> Ordering,
) -> PrimitiveArray {
    let mut v: Vec<T> = (0..len)
        .map(|_| match tc.draw(gs::integers::<u8>().max_value(4)) {
            0 => tc.draw(gs::sampled_from(specials)),
            _ => draw_general(tc),
        })
        .collect();
    if sorted {
        v.sort_by(cmp);
    }
    PrimitiveArray::new(Buffer::from(v), validity)
}

fn draw_pco_primitive(tc: &TestCase) -> PrimitiveArray {
    const PCO_PTYPES: [PType; 9] = [
        PType::U16,
        PType::U32,
        PType::U64,
        PType::I16,
        PType::I32,
        PType::I64,
        PType::F16,
        PType::F32,
        PType::F64,
    ];
    let ptype = tc.draw(gs::sampled_from(&PCO_PTYPES));
    let len = if tc.draw(gs::integers::<u8>().max_value(4)) == 0 {
        tc.draw(gs::integers::<usize>().min_value(65).max_value(200))
    } else {
        tc.draw(gs::integers::<usize>().max_value(64))
    };
    let nullability = if tc.draw(gs::booleans()) {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let validity = draw_validity_for(tc, nullability, len);
    let sorted = tc.draw(gs::integers::<u8>().max_value(3)) == 0;

    match ptype {
        PType::U16 => draw_pco_ints::<u16>(tc, len, validity, sorted),
        PType::U32 => draw_pco_ints::<u32>(tc, len, validity, sorted),
        PType::U64 => draw_pco_ints::<u64>(tc, len, validity, sorted),
        PType::I16 => draw_pco_ints::<i16>(tc, len, validity, sorted),
        PType::I32 => draw_pco_ints::<i32>(tc, len, validity, sorted),
        PType::I64 => draw_pco_ints::<i64>(tc, len, validity, sorted),
        PType::F16 => draw_pco_floats(
            tc,
            len,
            validity,
            sorted,
            &[
                f16::from_f32(0.0),
                f16::from_f32(-0.0),
                f16::from_bits(1),
                f16::MIN,
                f16::MAX,
                f16::INFINITY,
                f16::NEG_INFINITY,
                f16::NAN,
            ],
            |tc| f16::from_bits(tc.draw(gs::integers::<u16>())),
            f16::total_cmp,
        ),
        PType::F32 => draw_pco_floats(
            tc,
            len,
            validity,
            sorted,
            &[
                0.0,
                -0.0,
                f32::from_bits(1),
                f32::MIN,
                f32::MAX,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
            ],
            |tc| tc.draw(gs::floats::<f32>()),
            f32::total_cmp,
        ),
        PType::F64 => draw_pco_floats(
            tc,
            len,
            validity,
            sorted,
            &[
                0.0,
                -0.0,
                f64::from_bits(1),
                f64::MIN,
                f64::MAX,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ],
            |tc| tc.draw(gs::floats::<f64>()),
            f64::total_cmp,
        ),
        _ => unreachable!(),
    }
}

#[hegel::test(test_cases = 30, print_blob = true)]
fn test_pco_hegel_encoding_harness(tc: TestCase) {
    let mut ctx = SESSION.create_execution_ctx();
    let prim = draw_pco_primitive(&tc);
    let level = tc.draw(gs::integers::<usize>().max_value(3));
    let values_per_chunk = tc.draw(gs::sampled_from(&[0usize, 16, 33]));
    let values_per_page = tc.draw(gs::sampled_from(&[0usize, 8, 17]));
    let encoded = if values_per_chunk == 0 {
        Pco::from_primitive(prim.as_view(), level, values_per_page, &mut ctx)
            .vortex_expect("Pco::from_primitive")
            .into_array()
    } else {
        let validity = prim.validity().vortex_expect("prim validity");
        Pco::try_new(
            prim.dtype().clone(),
            PcoData::from_primitive_with_values_per_chunk(
                prim.as_view(),
                level,
                values_per_chunk,
                values_per_page,
                &mut ctx,
            )
            .vortex_expect("PcoData::from_primitive_with_values_per_chunk"),
            validity,
        )
        .vortex_expect("Pco::try_new")
        .into_array()
    };
    let canonical = prim.into_array();

    check_encoding_roundtrip_and_ops(
        &tc,
        &encoded,
        &canonical,
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
    .vortex_expect("Pco harness check");
}
