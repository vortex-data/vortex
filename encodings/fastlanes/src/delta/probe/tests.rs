// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayProbe;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RepeatedArrayProbe;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::Delta;
use crate::DeltaArraySlotsExt;

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
    ctx: &mut vortex_array::ExecutionCtx,
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

/// Every row of a three-chunk array, read through a one-off and a retained probe, matches the
/// decoded array, for each lane layout of each integer width.
#[rstest]
#[case::u8(PrimitiveArray::from_iter((0..3000u32).map(|i| u8::try_from(i * 5 % 256).unwrap())))]
#[case::u16(PrimitiveArray::from_iter((0..3000u16).map(|i| i.wrapping_mul(3))))]
#[case::u32(PrimitiveArray::from_iter((0..3000u32).map(|i| i * 7 + i % 5)))]
#[case::u64(PrimitiveArray::from_iter((0..3000u64).map(|i| i * 11 + 1_000_000_000_000)))]
#[case::i32(PrimitiveArray::from_iter((0..3000i32).map(|i| i * 3 - 4000)))]
#[case::i64(PrimitiveArray::from_iter((0..3000i64).map(|i| (i % 7) * 9 - i)))]
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
    #[values(false, true)] repeated: bool,
) -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    // Bit-packing needs non-negative values, so signed inputs only use primitive deltas and bases,
    // and the `u8` deltas, which need all 8 bits, are not bit-packed.
    let packs_deltas = matches!(
        layout,
        Layout::GlobalBitPacked | Layout::BlockedBitPacked | Layout::PatchedBitPacked
    );
    if !matches!(layout, Layout::PrimitiveDeltas) && input.ptype().is_signed_int()
        || packs_deltas && input.ptype() == PType::U8
    {
        return Ok(());
    }
    let delta = Delta::try_from_primitive_array(&input, &mut ctx)?;
    let source = with_layout(delta, layout, &mut ctx)?;
    // Every layout here reads single lanes rather than falling back to decoding the chunk.
    assert!(super::lane_value(source.as_::<Delta>(), 1, 517, &mut ctx)?.is_some());
    let mut actual = builder_with_capacity_in(source.dtype(), source.len(), ctx.allocator());
    let mut retained = None;
    let mut probe = probe_for(&source, &mut retained, repeated);
    for index in 0..source.len() {
        actual.append_scalar(&probe.execute_scalar(index, &mut ctx)?)?;
    }
    assert_arrays_eq!(actual.finish(), input, &mut ctx);
    Ok(())
}
