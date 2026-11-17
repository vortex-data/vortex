// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayProbe;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::RepeatedArrayProbe;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::Delta;
use crate::DeltaArrayExt;
use crate::DeltaArraySlotsExt;
use crate::delta::array::lane_count;

/// A one-off or retained probe over `array`, the retained one living in `retained`.
fn probe_for<'a>(
    array: &'a ArrayRef,
    retained: &'a mut Option<RepeatedArrayProbe>,
    repeated: bool,
) -> ArrayProbe<'a> {
    if repeated {
        retained.insert(array.repeated_probe()).as_probe()
    } else {
        array.probe()
    }
}

#[derive(Clone, Copy, Debug)]
enum Deltas {
    Primitive,
    BitPacked,
    /// Bit-packed at a width that leaves some deltas as patches.
    Patched,
}

#[rstest]
fn random_access_across_chunks_and_nulls(
    #[values(false, true)] repeated: bool,
    #[values(false, true)] sliced: bool,
    #[values(Deltas::Primitive, Deltas::BitPacked, Deltas::Patched)] deltas: Deltas,
) -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    // Jumps make deltas too wide for 2 bits, so the patched case packs them as patches.
    let jumps = matches!(deltas, Deltas::Patched);
    let input = PrimitiveArray::from_option_iter(
        (0..5000u32)
            .map(|i| (i % 11 != 0).then_some(if jumps && i % 997 == 5 { i * 7 } else { i / 3 })),
    );
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let delta = match deltas {
        Deltas::Primitive => delta,
        Deltas::BitPacked | Deltas::Patched => {
            let packed = BitPacked::encode(delta.deltas(), 2, &mut ctx)?;
            assert_eq!(packed.patches().is_some(), jumps);
            Delta::try_new(
                delta.bases().clone(),
                packed.into_array(),
                delta.validity()?,
                0,
                input.len(),
            )?
        }
    };
    let range = if sliced { 1777..4333 } else { 0..5000 };
    let source = if sliced {
        delta
            .into_array()
            .slice(range.clone())?
            .optimize_ctx(&ctx)?
    } else {
        delta.into_array()
    };
    assert!(source.is::<Delta>());
    let input = input.slice(range)?;
    // Revisits earlier chunks, so a retained probe must replace its cached chunk.
    let indices = [0u32, 1, 1023, 1024, 2048, 2047, 11, 33, 2500, 1500, 0];
    let mut actual = builder_with_capacity_in(source.dtype(), indices.len(), ctx.allocator());
    let mut retained = None;
    let mut probe = probe_for(&source, &mut retained, repeated);
    for index in indices {
        actual.append_scalar(&probe.execute_scalar(index as usize, &mut ctx)?)?;
    }
    assert_arrays_eq!(
        actual.finish(),
        input.take(PrimitiveArray::from_iter(indices).into_array())?,
        &mut ctx
    );
    assert!(probe.execute_scalar(source.len(), &mut ctx).is_err());
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    PrimitiveDeltas,
    /// Bit-packed deltas at one width.
    GlobalBitPacked,
    /// Bit-packed deltas at a width per 1024-value block.
    BlockedBitPacked,
    /// Bit-packed deltas one bit narrower than the widest, so the widest become patches.
    PatchedBitPacked,
    /// Primitive deltas over bit-packed bases, which the probe reads through the bases array.
    PackedBases,
}

/// Re-encodes the deltas or bases of `delta` as `layout` asks.
fn with_layout(
    delta: crate::DeltaArray,
    layout: Layout,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (bases, deltas) = (delta.bases().clone(), delta.deltas().clone());
    let (bases, deltas) = match layout {
        Layout::PrimitiveDeltas => (bases, deltas),
        Layout::GlobalBitPacked | Layout::PatchedBitPacked => {
            let max = deltas
                .clone()
                .execute::<PrimitiveArray>(ctx)?
                .into_array()
                .statistics()
                .compute_max::<u64>(ctx)
                .unwrap_or(0);
            let width = u8::try_from(u64::BITS - max.leading_zeros())?.max(2);
            let width = if matches!(layout, Layout::PatchedBitPacked) {
                width - 1
            } else {
                width
            };
            let packed = BitPacked::encode(&deltas, width, ctx)?;
            assert_eq!(
                packed.patches().is_some(),
                matches!(layout, Layout::PatchedBitPacked)
            );
            (bases, packed.into_array())
        }
        Layout::BlockedBitPacked => {
            let deltas = deltas.execute::<PrimitiveArray>(ctx)?;
            (
                bases,
                crate::bitpack_compress::bitpack_blocked_to_best_bit_widths(&deltas, ctx)?
                    .into_array(),
            )
        }
        Layout::PackedBases => {
            let width = u8::try_from(8 * bases.dtype().as_ptype().byte_width() - 1)?;
            (BitPacked::encode(&bases, width, ctx)?.into_array(), deltas)
        }
    };
    Ok(Delta::try_new(bases, deltas, delta.validity()?, 0, delta.len())?.into_array())
}

/// Reads every row of `source` through a one-off and a repeated probe, and checks both against
/// `expected`. With `lanes`, also checks that both kinds of read take the lane path rather than
/// decoding the chunk; without, that they fall back.
fn assert_every_row(
    source: &ArrayRef,
    expected: &PrimitiveArray,
    lanes: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let delta = source.as_::<Delta>();
    let resolved = super::ResolvedLanes::resolve(delta);
    assert_eq!(resolved.is_some(), lanes);
    for resolved in [None, resolved.as_ref()] {
        for chunk in 0..source.len().div_ceil(1024) {
            let within = (517 + delta.offset()).min(source.len() - 1 + delta.offset()) % 1024;
            let read = super::lane_value(delta, resolved, chunk, within, ctx)?;
            assert_eq!(read.is_some(), lanes, "chunk {chunk}");
        }
    }
    for repeated in [false, true] {
        let mut actual = builder_with_capacity_in(source.dtype(), source.len(), ctx.allocator());
        let mut retained = None;
        let mut probe = probe_for(source, &mut retained, repeated);
        for index in 0..source.len() {
            actual.append_scalar(&probe.execute_scalar(index, ctx)?)?;
        }
        assert_arrays_eq!(actual.finish(), expected.clone(), ctx);
    }
    Ok(())
}

/// Every row of a three-chunk array, read through one-off and repeated probes, matches the
/// decoded array, for each lane layout of each integer width.
///
/// The values increase in small steps from zero, so the deltas, and for [`Layout::PackedBases`]
/// the bases, are non-negative and bit-pack at every width.
#[rstest]
// A `u8` lane holds 8 consecutive values, so the 8-bit steps fall off multiples of 8.
#[case::u8(PrimitiveArray::from_iter((0..3000u32).map(|i| u8::try_from(i / 41 * 3).unwrap())))]
#[case::i8(PrimitiveArray::from_iter((0..3000i32).map(|i| i8::try_from(i / 49 * 2).unwrap())))]
#[case::u16(PrimitiveArray::from_iter((0..3000u16).map(|i| i * 3 + i % 4)))]
#[case::i16(PrimitiveArray::from_iter((0..3000i16).map(|i| i * 5 + i % 3)))]
#[case::u32(PrimitiveArray::from_iter((0..3000u32).map(|i| i * 7 + i % 5)))]
#[case::i32(PrimitiveArray::from_iter((0..3000i32).map(|i| i * 3 + i % 2)))]
#[case::u64(PrimitiveArray::from_iter((0..3000u64).map(|i| i * 11 + 1_000_000_000_000)))]
#[case::i64(PrimitiveArray::from_iter((0..3000i64).map(|i| i * 13 + i % 7)))]
fn every_row_matches_decode(
    #[case] input: PrimitiveArray,
    #[values(
        Layout::PrimitiveDeltas,
        Layout::GlobalBitPacked,
        Layout::BlockedBitPacked,
        Layout::PatchedBitPacked,
        Layout::PackedBases
    )]
    layout: Layout,
) -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let source = with_layout(delta, layout, &mut ctx)?;
    assert_every_row(&source, &input, true, &mut ctx)
}

/// Negative signed values, whose increasing deltas still bit-pack. Bit-packed bases cannot hold
/// negative values, so [`Layout::PackedBases`] is not among the layouts.
#[rstest]
#[case::i32(PrimitiveArray::from_iter((0..3000i32).map(|i| i * 3 + i % 2 - 4_000)))]
#[case::i64(PrimitiveArray::from_iter((0..3000i64).map(|i| i * 13 - 1_000_000_000_000)))]
fn negative_signed_values_match_decode(
    #[case] input: PrimitiveArray,
    #[values(
        Layout::PrimitiveDeltas,
        Layout::GlobalBitPacked,
        Layout::BlockedBitPacked,
        Layout::PatchedBitPacked
    )]
    layout: Layout,
) -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let source = with_layout(delta, layout, &mut ctx)?;
    assert_every_row(&source, &input, true, &mut ctx)
}

/// Increasing `u32` values whose deltas vary from 1 to 15, five chunks long.
fn varied_u32() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..5000u32).scan(0u32, |value, i| {
        *value += 1 + i * 7919 % 15;
        Some(*value)
    }))
}

/// A Delta array sliced off chunk boundaries keeps an offset, and its deltas become a slice of
/// the bit-packed array that starts at a later block. Patched and per-block widths alike.
#[rstest]
fn sliced_bit_packed_deltas(#[values(false, true)] blocked: bool) -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    let input = varied_u32();
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let deltas = delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let packed = if blocked {
        let widths = [1, 3, 0, 2, 5];
        crate::bitpack_compress::bitpack_encode_blocked(&deltas, &widths, None, &mut ctx)?
    } else {
        BitPacked::encode(&deltas.into_array(), 3, &mut ctx)?
    };
    assert!(packed.patches().is_some());
    let source = Delta::try_new(
        delta.bases().clone(),
        packed.into_array(),
        delta.validity()?,
        0,
        input.len(),
    )?
    .into_array()
    .slice(1500..4500)?
    .optimize_ctx(&ctx)?;
    assert!(source.is::<Delta>());
    assert_ne!(source.as_::<Delta>().offset(), 0);
    let input = input
        .slice(1500..4500)?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_every_row(&source, &input, true, &mut ctx)
}

/// A bit-packed array that itself starts partway into its first block, under a lazy slice that
/// brings the deltas back to a block boundary, with patches.
#[test]
fn bit_packed_deltas_with_an_offset() -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    let input = varied_u32();
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let packed = BitPacked::encode(delta.deltas(), 3, &mut ctx)?.into_array();
    // The slice kernel keeps the packed blocks and slices the patches, so the bit-packed array
    // starts 500 into its first block. The lazy slice over it starts 524 further on, at the
    // second block, so the deltas start on a block boundary.
    let offset_packed =
        <BitPacked as SliceKernel>::slice(packed.as_::<BitPacked>(), 500..packed.len(), &mut ctx)?
            .vortex_expect("bit-packed arrays slice");
    assert_eq!(offset_packed.as_::<BitPacked>().offset(), 500);
    assert!(offset_packed.as_::<BitPacked>().patches().is_some());
    let deltas = offset_packed.slice(524..offset_packed.len())?;
    let lanes = lane_count(input.ptype());
    let source = Delta::try_new(
        delta.bases().slice(lanes..delta.bases().len())?,
        deltas,
        delta.validity()?.slice(1024..input.len())?,
        0,
        input.len() - 1024,
    )?
    .into_array();
    let input = input
        .slice(1024..input.len())?
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_every_row(&source, &input, true, &mut ctx)
}

/// Deltas in an encoding lane reads do not support, frame of reference, fall back to decoding the
/// chunk, through both kinds of probe.
#[test]
fn other_delta_encodings_fall_back() -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    let input = varied_u32();
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let deltas = crate::FoR::encode(
        delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?,
        &mut ctx,
    )?;
    let source = Delta::try_new(
        delta.bases().clone(),
        deltas.into_array(),
        delta.validity()?,
        0,
        input.len(),
    )?
    .into_array();
    assert_every_row(&source, &input, false, &mut ctx)
}

/// Block boundaries from a file are only validated in debug builds. A block whose boundaries give
/// more bits than the type holds falls back to the decode, which reports it, rather than
/// panicking.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "debug builds reject the boundaries when the array is built"
)]
fn malformed_block_offsets_do_not_panic() -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_iter(0..1024u32);
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    // 33 bits per value, one more than `u32` holds.
    let bytes = 33 * 128;
    let packed = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(ByteBuffer::from(vec![0u8; bytes])),
        PType::U32,
        Validity::NonNullable,
        None,
        PrimitiveArray::from_iter([0u32, u32::try_from(bytes)?]).into_array(),
        1024,
        0,
    )?;
    let source = Delta::try_new(
        delta.bases().clone(),
        packed.into_array(),
        Validity::NonNullable,
        0,
        1024,
    )?
    .into_array();
    assert!(super::ResolvedLanes::resolve(source.as_::<Delta>()).is_some());
    assert!(super::lane_value(source.as_::<Delta>(), None, 0, 5, &mut ctx)?.is_none());
    for repeated in [false, true] {
        let mut retained = None;
        let mut probe = probe_for(&source, &mut retained, repeated);
        assert!(probe.execute_scalar(5, &mut ctx).is_err());
    }
    Ok(())
}

/// A repeated probe that alternates between chunks reads a lane on its first visit to each and
/// decodes a chunk on its second, and stays correct while the decoded chunk changes.
#[test]
fn repeated_probe_alternating_chunks() -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    let input = varied_u32();
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let source = with_layout(delta, Layout::PatchedBitPacked, &mut ctx)?;
    let indices = [5u32, 1030, 7, 2100, 1031, 9, 2101, 4999, 6, 4998, 1032];
    let mut actual = builder_with_capacity_in(source.dtype(), indices.len(), ctx.allocator());
    let mut probe = source.repeated_probe();
    for index in indices {
        actual.append_scalar(&probe.execute_scalar(index as usize, &mut ctx)?)?;
    }
    assert_arrays_eq!(
        actual.finish(),
        input.take(PrimitiveArray::from_iter(indices).into_array())?,
        &mut ctx
    );
    Ok(())
}
