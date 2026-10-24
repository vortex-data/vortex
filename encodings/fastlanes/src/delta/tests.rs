// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Delta roundtrips and compute operations against primitive arrays.

use fastlanes::FastLanes;
use prost::Message;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_aggregates::min_max::min_max;
use vortex_array::ArrayContext;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::SliceArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;
use crate::Delta;
use crate::DeltaArray;
use crate::DeltaArrayExt;
use crate::DeltaArraySlotsExt;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_compress::bitpack_encode;
use crate::delta::array::delta_decompress::PackedDeltas;
use crate::delta::vtable::DeltaMetadata;

fn session() -> VortexSession {
    let session = VortexSession::empty();
    crate::initialize(&session);
    vortex_aggregates::initialize(&session);
    session
}

fn serde_roundtrip(array: &ArrayRef, session: &VortexSession) -> VortexResult<ArrayRef> {
    let write_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in array.serialize(&write_ctx, session, &SerializeOptions::default())? {
        bytes.extend_from_slice(&buffer);
    }
    SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(write_ctx.to_ids()),
        session,
    )
}

fn check_roundtrip(source: &PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<DeltaArray> {
    let delta = Delta::try_from_primitive_array(source, ctx)?;
    assert!(matches!(delta.bases().validity()?, Validity::NonNullable));
    assert!(matches!(delta.deltas().validity()?, Validity::NonNullable));
    assert!(!delta.bases().dtype().is_nullable());
    assert!(!delta.deltas().dtype().is_nullable());
    if let Some(validity) = delta.validity_child() {
        assert_eq!(validity.len(), source.len());
        assert_eq!(validity.dtype(), &Validity::DTYPE);
    }
    let decoded = delta.as_array().clone().execute::<PrimitiveArray>(ctx)?;
    assert_arrays_eq!(decoded, source, ctx);
    assert_eq!(
        min_max(delta.as_array(), ctx, Default::default())?,
        min_max(source.as_array(), ctx, Default::default())?,
    );

    let valid = source.validity()?.execute_mask(source.len(), ctx)?;
    match_each_integer_ptype!(source.ptype(), |T| {
        check_filled_values(
            source.as_slice::<T>(),
            decoded.to_buffer::<T>(),
            &valid,
            ctx,
        );
    });
    Ok(delta)
}

fn check_filled_values<T: NativePType>(
    values: &[T],
    decoded: Buffer<T>,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
) {
    // Nulls repeat the last valid value, except that nulls before a chunk's first valid value
    // take that value instead.
    let mut filled = Vec::with_capacity(values.len());
    let mut previous = T::default();
    for (chunk, chunk_valid) in values.chunks(FL_CHUNK_SIZE).zip(
        valid
            .to_bit_buffer()
            .iter()
            .collect::<Vec<_>>()
            .chunks(FL_CHUNK_SIZE),
    ) {
        if let Some(first_valid) = chunk_valid.iter().position(|&is_valid| is_valid) {
            previous = chunk[first_valid];
        }
        for (&value, &is_valid) in chunk.iter().zip(chunk_valid) {
            if is_valid {
                previous = value;
            }
            filled.push(previous);
        }
    }
    assert_arrays_eq!(
        PrimitiveArray::new(decoded, Validity::NonNullable),
        PrimitiveArray::from_iter(filled),
        ctx
    );
}

#[rstest]
#[case::all_null(0)]
#[case::alternating(1)]
#[case::leading_nulls(2)]
fn delta_null_patterns(#[case] pattern: usize) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    for len in [0, 1, 63, 1023, 1024, 1025, 2049, 3072] {
        let validity = Validity::from_iter((0..len).map(|index| match pattern {
            0 => false,
            1 => index % 2 == 1,
            _ => index % FL_CHUNK_SIZE >= 63,
        }));
        let source = PrimitiveArray::new(
            (0..len)
                .map(|index| 1000 + index as u32)
                .collect::<Buffer<_>>(),
            validity,
        );
        let delta = check_roundtrip(&source, &mut ctx)?;
        let deltas = delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
        let packed = Delta::try_new(
            delta.bases().clone(),
            bitpack_encode(&deltas, 1, None, &mut ctx)?.into_array(),
            delta.validity()?,
            0,
            len,
        )?;
        assert_arrays_eq!(packed, source, &mut ctx);

        for array in [delta.into_array(), packed.into_array()] {
            let restored = serde_roundtrip(&array, &session)?;
            let restored_delta = restored
                .as_opt::<Delta>()
                .ok_or_else(|| vortex_err!("expected deserialized Delta"))?;
            assert!(!restored_delta.deltas().dtype().is_nullable());
            assert_arrays_eq!(restored, source, &mut ctx);
        }
    }
    Ok(())
}

#[rstest]
#[case::u8(PType::U8)]
#[case::u16(PType::U16)]
#[case::u32(PType::U32)]
#[case::u64(PType::U64)]
#[case::i8(PType::I8)]
#[case::i16(PType::I16)]
#[case::i32(PType::I32)]
#[case::i64(PType::I64)]
fn delta_randomized_model(#[case] ptype: PType) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(0xde17a);
    for _ in 0..64 {
        let len = rng.random_range(1..=3073);
        let valid_probability = rng.random_range(0.0..=1.0);
        let validity = Validity::from_iter((0..len).map(|_| rng.random_bool(valid_probability)));
        let source = match_each_integer_ptype!(ptype, |T| {
            PrimitiveArray::new(
                (0..len).map(|_| rng.random::<T>()).collect::<Buffer<_>>(),
                validity,
            )
        });
        let delta = check_roundtrip(&source, &mut ctx)?.into_array();
        let start = rng.random_range(0..=len);
        let end = rng.random_range(start..=len);
        let sliced = delta.slice(start..end)?;
        assert_arrays_eq!(sliced, source.slice(start..end)?, &mut ctx);
        let nested_start = rng.random_range(0..=end - start);
        assert_arrays_eq!(
            sliced.slice(nested_start..end - start)?,
            source.slice(start + nested_start..end)?,
            &mut ctx
        );

        let indices = PrimitiveArray::from_option_iter((0..32).map(|_| {
            rng.random_bool(0.8)
                .then(|| rng.random_range(0..len) as u64)
        }))
        .into_array();
        assert_arrays_eq!(
            delta.take(indices.clone())?,
            source.take(indices)?,
            &mut ctx
        );
        let mask = Mask::from_iter((0..len).map(|_| rng.random_bool(0.5)));
        assert_arrays_eq!(delta.filter(mask.clone())?, source.filter(mask)?, &mut ctx);
    }
    Ok(())
}

#[test]
fn delta_stats_include_null_slot_residuals() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let source = PrimitiveArray::from_option_iter(
        (0u32..1024).map(|index| (index % 2 == 1).then_some(1000 + index)),
    );
    let delta = check_roundtrip(&source, &mut ctx)?;
    let deltas = delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(deltas.statistics().compute_min::<u32>(&mut ctx), Some(0));
    // The leading null takes the first valid value, so no residual jumps from zero to 1001.
    assert_eq!(deltas.statistics().compute_max::<u32>(&mut ctx), Some(2));
    assert_eq!(delta.statistics().compute_min::<u32>(&mut ctx), Some(1001));
    assert_eq!(delta.statistics().compute_max::<u32>(&mut ctx), Some(2023));
    Ok(())
}

#[rstest]
#[case::all_valid(Validity::AllValid)]
#[case::all_invalid(Validity::AllInvalid)]
#[case::bitmap(Validity::from_iter((0..FL_CHUNK_SIZE).map(|index| index % 2 == 0)))]
fn delta_rejects_nullable_deltas(#[case] validity: Validity) -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let source = PrimitiveArray::from_iter(0u32..1024);
    let delta = Delta::try_from_primitive_array(&source, &mut ctx)?;
    let deltas = delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let nullable = PrimitiveArray::new(deltas.to_buffer::<u32>(), validity);
    assert!(
        Delta::try_new(
            delta.bases().clone(),
            nullable.into_array(),
            Validity::AllValid,
            0,
            source.len()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn delta_rejects_invalid_top_validity() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let source = PrimitiveArray::from_iter(0u32..100);
    let delta = Delta::try_from_primitive_array(&source, &mut ctx)?;
    for validity in [
        BoolArray::from_iter([true, false]).into_array(),
        PrimitiveArray::from_iter(0u32..100).into_array(),
    ] {
        assert!(
            Delta::try_new(
                delta.bases().clone(),
                delta.deltas().clone(),
                Validity::Array(validity),
                0,
                source.len()
            )
            .is_err()
        );
    }
    Ok(())
}

/// A nullable signed array sliced to a non-zero offset round-trips through serde.
#[test]
fn delta_serde_sliced_signed() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let len = 3000;
    let source = PrimitiveArray::from_option_iter(
        (0..len).map(|index| (index % 7 != 0).then_some(index as i64 * -3 + 500)),
    );
    let delta = Delta::try_from_primitive_array(&source, &mut ctx)?;
    let (offset, sliced_len) = (1000, 2000);
    let sliced = Delta::try_new(
        delta.bases().clone(),
        delta.deltas().clone(),
        Validity::from_iter((offset..offset + sliced_len).map(|index| index % 7 != 0)),
        offset,
        sliced_len,
    )?
    .into_array();

    let restored = serde_roundtrip(&sliced, &session)?;
    let restored_delta = restored
        .as_opt::<Delta>()
        .ok_or_else(|| vortex_err!("expected deserialized Delta"))?;
    assert_eq!(restored_delta.offset(), offset);
    assert_arrays_eq!(
        restored,
        source.slice(offset..offset + sliced_len)?,
        &mut ctx
    );
    Ok(())
}

/// Serializing rejects an array whose deltas extend past the chunk holding its last value.
#[test]
fn delta_serialize_rejects_trailing_chunks() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let source = PrimitiveArray::from_iter(0..2048u32);
    let delta = Delta::try_from_primitive_array(&source, &mut ctx)?;
    let truncated = Delta::try_new(
        delta.bases().clone(),
        delta.deltas().clone(),
        Validity::NonNullable,
        0,
        10,
    )?
    .into_array();
    assert!(serde_roundtrip(&truncated, &session).is_err());
    Ok(())
}

#[test]
fn delta_deserialize_rejects_malformed_parts() -> VortexResult<()> {
    let session = session();
    let mut ctx = session.create_execution_ctx();
    let len = 1500;
    let source = PrimitiveArray::from_option_iter(
        (0..len).map(|index| (index % 3 != 0).then_some(index as u32)),
    );
    let delta = Delta::try_from_primitive_array(&source, &mut ctx)?;
    let bases = delta.bases().clone();
    let deltas = delta.deltas().clone();
    let validity = delta
        .validity_child()
        .ok_or_else(|| vortex_err!("expected a validity child"))?
        .clone();
    let metadata = DeltaMetadata {
        deltas_len: 2048,
        offset: 0,
        has_validity: true,
    };
    let dtype = source.dtype().clone();
    let deserialize = |dtype: &DType, metadata: &DeltaMetadata, children: &[ArrayRef]| {
        Delta.deserialize(
            ArrayDeserialization::new(
                ArrayPlugin::id(&Delta),
                dtype,
                len,
                &metadata.encode_to_vec(),
                &[],
                &children,
            ),
            &session,
        )
    };
    let all = [bases.clone(), deltas.clone(), validity.clone()];

    let restored = deserialize(&dtype, &metadata, &all)?;
    assert_arrays_eq!(restored, source, &mut ctx);

    // Readers ignore children that follow the ones the metadata declares.
    let with_trailing = [bases.clone(), deltas.clone(), validity, bases.clone()];
    let restored = deserialize(&dtype, &metadata, &with_trailing)?;
    assert_arrays_eq!(restored, source, &mut ctx);

    // The metadata declares a validity child that is missing.
    assert!(deserialize(&dtype, &metadata, &[bases, deltas]).is_err());

    // A validity child on a non-nullable dtype.
    assert!(deserialize(&dtype.as_nonnullable(), &metadata, &all).is_err());

    // A non-integer dtype.
    let float = DType::Primitive(PType::F32, Nullability::Nullable);
    assert!(deserialize(&float, &metadata, &all).is_err());

    // The offset must lie in the first chunk.
    let offset_past_chunk = DeltaMetadata {
        offset: 1024,
        ..metadata
    };
    assert!(deserialize(&dtype, &offset_past_chunk, &all).is_err());

    // The deltas length must be exactly (offset + len) rounded up to a whole chunk.
    for deltas_len in [0, 1024, 2000, 3072, u64::MAX] {
        let wrong_len = DeltaMetadata {
            deltas_len,
            ..metadata
        };
        assert!(deserialize(&dtype, &wrong_len, &all).is_err());
    }
    Ok(())
}

/// Replace the deltas of `source` encoded as Delta with deltas bit-packed at `bit_widths`: one
/// global width, or one width per chunk.
fn delta_bitpacked(
    source: &PrimitiveArray,
    bit_widths: &[u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<DeltaArray> {
    let delta = Delta::try_from_primitive_array(source, ctx)?;
    let deltas = delta.deltas().clone();
    let packed = match bit_widths {
        [bit_width] => BitPacked::encode(&deltas, *bit_width, ctx)?,
        _ => BitPackedData::encode_blocked(&deltas, bit_widths, ctx)?,
    };
    Delta::try_new(
        delta.bases().clone(),
        packed.into_array(),
        delta.validity()?,
        0,
        source.len(),
    )
}

/// Monotone values whose deltas fit in a few bits, across signed and unsigned widths.
fn monotone(ptype: PType, len: usize) -> PrimitiveArray {
    match_each_integer_ptype!(ptype, |T| {
        PrimitiveArray::from_iter(
            (0..len).map(|i| <T as num_traits::NumCast>::from(i / 32).unwrap()),
        )
    })
}

/// Whether the deltas of `array` take the fused decode.
fn fused(array: &ArrayRef) -> bool {
    PackedDeltas::try_new(array.as_::<Delta>().deltas()).is_some()
}

#[rstest]
fn delta_bitpacked_fused_global(
    #[values(PType::U8, PType::U16, PType::U32, PType::U64, PType::I32, PType::I64)] ptype: PType,
    #[values(1, 1023, 1024, 3000)] len: usize,
) -> VortexResult<()> {
    let mut ctx = session().create_execution_ctx();
    let source = monotone(ptype, len);
    let array = delta_bitpacked(&source, &[1], &mut ctx)?.into_array();
    assert!(fused(&array));
    assert_arrays_eq!(array, source, &mut ctx);
    Ok(())
}

#[rstest]
#[case::nullable(PrimitiveArray::from_option_iter(
    (0..5000u32).map(|i| (i % 7 != 0).then_some(i / 2))
))]
#[case::signed_crossing_zero(PrimitiveArray::from_iter((-2500..2500i32).map(|i| i / 2)))]
fn delta_bitpacked_fused_values(#[case] source: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = session().create_execution_ctx();
    let array = delta_bitpacked(&source, &[1], &mut ctx)?.into_array();
    assert!(fused(&array));
    assert_arrays_eq!(array, source, &mut ctx);
    Ok(())
}

/// Each chunk's deltas grow by 4 bits, so each block has a different best width.
fn blocked_source() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..4 * FL_CHUNK_SIZE as u64).scan(0u64, |acc, i| {
        *acc += (i % 3) << (4 * (i as usize / FL_CHUNK_SIZE));
        Some(*acc)
    }))
}

#[rstest]
#[case::fitting(&[2, 6, 10, 14], false)]
// Width 0 of the third chunk does not fit its deltas, so that chunk is patched.
#[case::patched_block(&[2, 6, 0, 14], true)]
fn delta_bitpacked_fused_blocked(
    #[case] bit_widths: &[u8],
    #[case] patched: bool,
) -> VortexResult<()> {
    let mut ctx = session().create_execution_ctx();
    let source = blocked_source();
    let array = delta_bitpacked(&source, bit_widths, &mut ctx)?;
    let bp = array.deltas().as_::<BitPacked>();
    assert!(!bp.bit_widths().is_global());
    assert_eq!(bp.patches().is_some(), patched);
    let array = array.into_array();
    assert!(fused(&array));
    assert_arrays_eq!(array, source, &mut ctx);
    Ok(())
}

/// Deltas that do not fit the bit width become patches. Chunks with patches are unpacked and
/// patched before undelta, the others take the fused kernel.
#[test]
fn delta_bitpacked_patched() -> VortexResult<()> {
    let mut ctx = session().create_execution_ctx();
    // Jumps in the first and third chunks only.
    let source = PrimitiveArray::from_iter(
        (0..4000u32).map(|i| if i == 100 || i == 2500 { i * 7 } else { i }),
    );
    let array = delta_bitpacked(&source, &[3], &mut ctx)?;
    assert!(array.deltas().as_::<BitPacked>().patches().is_some());
    let array = array.into_array();
    assert!(fused(&array));
    assert_arrays_eq!(array, source, &mut ctx);
    Ok(())
}

#[rstest]
#[case::within_chunk(10, 900)]
#[case::across_chunks(1000, 3100)]
#[case::chunk_aligned(1024, 3072)]
#[case::past_patches(2100, 3999)]
#[case::tail(3000, 4000)]
fn delta_bitpacked_fused_sliced(
    #[case] start: usize,
    #[case] end: usize,
    #[values(false, true)] patched: bool,
) -> VortexResult<()> {
    let mut ctx = session().create_execution_ctx();
    let source = if patched {
        PrimitiveArray::from_iter((0..4000u32).map(|i| if i % 1000 == 7 { i * 7 } else { i / 32 }))
    } else {
        monotone(PType::U32, 4000)
    };
    let array = delta_bitpacked(&source, &[1], &mut ctx)?;
    assert_eq!(
        array.deltas().as_::<BitPacked>().patches().is_some(),
        patched
    );
    let sliced = array.into_array().slice(start..end)?.optimize_ctx(&ctx)?;
    assert_eq!(sliced.as_::<Delta>().offset(), start % FL_CHUNK_SIZE);
    assert!(fused(&sliced));
    assert_arrays_eq!(sliced, source.into_array().slice(start..end)?, &mut ctx);
    Ok(())
}

/// A lazy slice of patched deltas that starts on a block decodes whole blocks of its child.
#[test]
fn delta_lazy_sliced_bitpacked_fused() -> VortexResult<()> {
    let mut ctx = session().create_execution_ctx();
    let source =
        PrimitiveArray::from_iter((0..4000u32).map(|i| if i % 1000 == 7 { i * 7 } else { i / 32 }));
    let array = delta_bitpacked(&source, &[1], &mut ctx)?;
    assert!(array.deltas().as_::<BitPacked>().patches().is_some());
    let lanes = u32::LANES;
    let deltas = SliceArray::new(array.deltas().clone(), 1024..3072).into_array();
    let sliced = Delta::try_new(
        array.bases().slice(lanes..3 * lanes)?,
        deltas,
        Validity::NonNullable,
        5,
        2000,
    )?
    .into_array();
    assert!(sliced.as_::<Delta>().deltas().is::<Slice>());
    assert!(fused(&sliced));
    assert_arrays_eq!(sliced, source.into_array().slice(1029..3029)?, &mut ctx);
    Ok(())
}

/// Deltas that start inside a packed block straddle two blocks per chunk, so they are unpacked
/// before undelta.
#[test]
fn delta_bitpacked_misaligned_unfused() -> VortexResult<()> {
    let mut ctx = session().create_execution_ctx();
    let source = monotone(PType::U32, 3 * FL_CHUNK_SIZE);
    let delta = Delta::try_from_primitive_array(&source, &mut ctx)?;
    let deltas = delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
    // Shift the deltas half a block into the packed buffer.
    let mut shifted = vec![0u32; 512];
    shifted.extend(deltas.as_slice::<u32>());
    let packed = BitPacked::encode(
        &PrimitiveArray::from_iter(shifted).into_array(),
        1,
        &mut ctx,
    )?;
    let misaligned = packed
        .into_array()
        .slice(512..512 + deltas.len())?
        .optimize_ctx(&ctx)?;
    assert_eq!(misaligned.as_::<BitPacked>().offset(), 512);
    let array = Delta::try_new(
        delta.bases().clone(),
        misaligned,
        Validity::NonNullable,
        0,
        source.len(),
    )?
    .into_array();
    assert!(!fused(&array));
    assert_arrays_eq!(array, source, &mut ctx);
    Ok(())
}
