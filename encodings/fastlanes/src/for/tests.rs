// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::dtype::NativePType;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::FoRArray;
use crate::FoRArrayExt;
use crate::FoRArraySlotsExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// Builds a FoR array over `encoded` with the given per-chunk references, and the values it
/// should decode to.
fn chunked<T: NativePType + num_traits::WrappingAdd>(
    encoded: PrimitiveArray,
    references: &[T],
) -> VortexResult<(FoRArray, PrimitiveArray)> {
    let mut ctx = SESSION.create_execution_ctx();
    let expected = PrimitiveArray::from_option_iter(
        (0..encoded.len())
            .map(|i| {
                let value = encoded.as_slice::<T>()[i];
                let valid = encoded.is_valid(i, &mut ctx)?;
                Ok(valid.then(|| value.wrapping_add(&references[i / FL_CHUNK_SIZE])))
            })
            .collect::<VortexResult<Vec<_>>>()?,
    );
    let expected = if encoded.dtype().is_nullable() {
        expected
    } else {
        PrimitiveArray::new(expected.to_buffer::<T>(), encoded.validity()?)
    };
    let references = Buffer::copy_from(references).into_array();
    Ok((
        FoR::try_new_chunked(encoded.into_array(), references, 0)?,
        expected,
    ))
}

fn unsigned() -> VortexResult<(FoRArray, PrimitiveArray)> {
    chunked(
        PrimitiveArray::from_iter((0..3000u32).map(|i| i % 100)),
        &[0u32, 1_000_000, 5],
    )
}

fn signed_wrapping() -> VortexResult<(FoRArray, PrimitiveArray)> {
    chunked(
        PrimitiveArray::from_iter((0..2100i16).map(|i| i % 7)),
        &[i16::MIN, -3, i16::MAX],
    )
}

fn nullable() -> VortexResult<(FoRArray, PrimitiveArray)> {
    chunked(
        PrimitiveArray::from_option_iter((0..2500i64).map(|i| (i % 5 != 0).then_some(i % 11))),
        &[-1_000i64, 0, 1 << 40],
    )
}

#[rstest]
#[case::unsigned(unsigned())]
#[case::signed_wrapping(signed_wrapping())]
#[case::nullable(nullable())]
fn decodes_per_chunk(#[case] arrays: VortexResult<(FoRArray, PrimitiveArray)>) -> VortexResult<()> {
    let (array, expected) = arrays?;
    assert!(array.constant_reference().is_none());
    assert_arrays_eq!(array, expected, &mut SESSION.create_execution_ctx());
    Ok(())
}

#[rstest]
#[case::unsigned(unsigned())]
#[case::signed_wrapping(signed_wrapping())]
#[case::nullable(nullable())]
fn consistency(#[case] arrays: VortexResult<(FoRArray, PrimitiveArray)>) -> VortexResult<()> {
    let (array, _) = arrays?;
    test_array_consistency(&array.into_array(), &mut SESSION.create_execution_ctx());
    Ok(())
}

#[rstest]
#[case::within_first_chunk(3, 900)]
#[case::across_chunks(1000, 2100)]
#[case::chunk_aligned(1024, 2048)]
#[case::last_chunk(2048, 3000)]
#[case::empty_unaligned(1500, 1500)]
fn slice_keeps_chunk_alignment(#[case] start: usize, #[case] end: usize) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (array, expected) = unsigned()?;
    let sliced = array.into_array().slice(start..end)?;
    assert_arrays_eq!(sliced, expected.into_array().slice(start..end)?, &mut ctx);

    if let Some(sliced) = sliced.as_opt::<FoR>() {
        assert_eq!(usize::from(sliced.offset()), start % FL_CHUNK_SIZE);
        // Slicing a slice composes offsets.
        let len = end - start;
        let inner: ArrayRef = sliced.array().slice(len / 3..len)?;
        let expected = unsigned()?.1.into_array().slice(start + len / 3..end)?;
        assert_arrays_eq!(inner, expected, &mut ctx);
    }
    Ok(())
}

#[test]
fn constant_references_keep_the_scalar_reference() -> VortexResult<()> {
    let array = FoR::try_new(
        PrimitiveArray::from_iter(0..3000u32).into_array(),
        7u32.into(),
    )?;
    assert_eq!(array.references().len(), 3);
    assert_eq!(array.constant_reference(), Some(7u32.into()));
    let sliced = array.into_array().slice(1500..1600)?;
    assert_eq!(sliced.as_::<FoR>().constant_reference(), Some(7u32.into()));
    Ok(())
}

fn drifting_u32(len: u32) -> PrimitiveArray {
    PrimitiveArray::from_iter((0..len).map(|i| (i / 1024) * 1_000_000 + i % 100))
}

#[rstest]
#[case::empty(PrimitiveArray::from_iter(Vec::<u32>::new()))]
#[case::one(PrimitiveArray::from_iter([7u32]))]
#[case::below_chunk(drifting_u32(1023))]
#[case::one_chunk(drifting_u32(1024))]
#[case::above_chunk(drifting_u32(1025))]
#[case::drifting(drifting_u32(5000))]
#[case::signed(PrimitiveArray::from_iter((0..3000i64).map(|i| (1 - (i / 1024) * 2) * 1_000_000_000 + i)))]
#[case::extremes(PrimitiveArray::from_iter((0..2048).map(|i| if i < 1024 { i8::MIN } else { i8::MAX })))]
#[case::nullable(PrimitiveArray::from_option_iter((0..3000i32).map(|i| (i % 3 != 0).then_some(i * 7))))]
#[case::nullable_whole_words(PrimitiveArray::from_option_iter((0..2048u32).map(|i| (i % 3 != 0).then_some(i))))]
#[case::null_chunk(PrimitiveArray::from_option_iter((0..3000u16).map(|i| (!(1024..2048).contains(&i)).then_some(i))))]
#[case::all_null(PrimitiveArray::from_option_iter((0..2000).map(|_| None::<u64>)))]
fn encode_chunked_roundtrip(#[case] array: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = FoR::encode_chunked(array.clone(), &mut ctx)?;
    assert_eq!(encoded.offset(), 0);
    assert_arrays_eq!(encoded, array, &mut ctx);
    test_array_consistency(&encoded.into_array(), &mut ctx);
    Ok(())
}

#[test]
fn encode_chunked_uses_chunk_minimums() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter((0..4000u32).map(|i| match i / 1024 {
        0 => Some(500 + i % 10),
        1 => None,
        // Null slots hold 0, which must not count towards the minimum.
        2 => (i % 7 != 0).then_some(9_000 + i % 10),
        _ => Some(3 + i % 10),
    }));
    let encoded = FoR::encode_chunked(values, &mut ctx)?;
    // The all-null chunk 1 reuses chunk 0's reference.
    assert_arrays_eq!(
        encoded.references(),
        PrimitiveArray::from_iter([500u32, 500, 9_000, 3]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn encode_chunked_all_null_is_constant() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter((0..2000).map(|_| None::<u64>));
    let encoded = FoR::encode_chunked(values.clone(), &mut ctx)?;
    assert!(encoded.encoded().is::<Constant>());
    assert!(encoded.constant_reference().is_some());
    // A constant reference serializes in the single-reference format.
    assert!(SESSION.array_serialize(encoded.as_array()).is_ok());
    assert_arrays_eq!(encoded, values, &mut ctx);
    Ok(())
}

/// A chunked FoR over a BitPacked child, which takes the fused decode path.
fn fused(bit_width: u8) -> VortexResult<(FoRArray, PrimitiveArray)> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_option_iter(
        (0..5000u32).map(|i| (i % 13 != 0).then_some((i / 1024) * 1_000_000 + i % 7)),
    );
    let for_array = FoR::encode_chunked(values.clone(), &mut ctx)?;
    let bp = crate::BitPacked::encode(for_array.encoded(), bit_width, &mut ctx)?;
    let array = FoR::try_new_chunked(bp.into_array(), for_array.references().clone(), 0)?;
    Ok((array, values))
}

#[rstest]
#[case::no_patches(3)]
#[case::patches(2)]
fn fused_decode(#[case] bit_width: u8) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (array, expected) = fused(bit_width)?;
    assert!(array.encoded().is::<crate::BitPacked>());
    assert_arrays_eq!(array, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::within_chunk(10, 900)]
#[case::across_chunks(1000, 3100)]
#[case::chunk_aligned(1024, 4096)]
#[case::tail(4000, 5000)]
fn fused_decode_sliced(
    #[case] start: usize,
    #[case] end: usize,
    #[values(2, 3)] bit_width: u8,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (array, expected) = fused(bit_width)?;
    let sliced = array.into_array().slice(start..end)?;
    let sliced_for = sliced.as_::<FoR>();
    // A BitPacked child with patches slices lazily, which takes the unfused path.
    assert_eq!(
        sliced_for.encoded().is::<crate::BitPacked>(),
        bit_width == 3
    );
    assert_eq!(usize::from(sliced_for.offset()), start % FL_CHUNK_SIZE);
    assert_arrays_eq!(sliced, expected.into_array().slice(start..end)?, &mut ctx);
    Ok(())
}
