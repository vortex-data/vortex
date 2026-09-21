// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::BitPackedData;
use crate::BlockedFoR;
use crate::BlockedFoRArrayExt;
use crate::BlockedFoRArraySlotsExt;
use crate::BlockedFoRData;
use crate::FoRArraySlotsExt;
use crate::FoRData;
use crate::bitpack_compress::bit_width_histogram;
use crate::bitpack_compress::find_best_bit_width;
use crate::blocked_for::array::BLOCK_SIZE;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
    crate::initialize(&session);
    session
});

/// A staircase: values drift far apart globally but stay tightly clustered inside each block,
/// which is exactly the shape blocked FoR is meant to exploit.
fn staircase(len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter(
        (0..len).map(|i| (i / BLOCK_SIZE) as i64 * 1_000_000 + (i % 97) as i64),
    )
}

#[rstest]
#[case(0)]
#[case(1)]
#[case(BLOCK_SIZE - 1)]
#[case(BLOCK_SIZE)]
#[case(BLOCK_SIZE + 1)]
#[case(3 * BLOCK_SIZE + 7)]
fn round_trip(#[case] len: usize) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = staircase(len);
    let encoded = BlockedFoRData::encode(array.clone(), &mut ctx)?;

    assert_eq!(encoded.num_blocks(), len.div_ceil(BLOCK_SIZE));
    assert_eq!(encoded.references().len(), len.div_ceil(BLOCK_SIZE));
    assert_arrays_eq!(encoded, array, &mut ctx);
    Ok(())
}

#[test]
fn references_are_block_minima() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 2 * BLOCK_SIZE + 5;
    let encoded = BlockedFoRData::encode(staircase(len), &mut ctx)?;

    let references = encoded
        .references()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(
        references.as_slice::<i64>(),
        &[0, 1_000_000, 2_000_000 + (2 * BLOCK_SIZE % 97) as i64]
    );
    Ok(())
}

#[rstest]
#[case::unsigned(PrimitiveArray::from_iter((0u32..3000).map(|v| v / 100 * 50_000 + v % 13)))]
#[case::signed_negative(PrimitiveArray::from_iter((0i32..3000).map(|v| -v * 1000)))]
#[case::i8_full_range(PrimitiveArray::from_iter(i8::MIN..=i8::MAX))]
#[case::constant(PrimitiveArray::from_iter(std::iter::repeat_n(7u64, 2500)))]
fn round_trip_ptypes(#[case] array: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = BlockedFoRData::encode(array.clone(), &mut ctx)?;
    assert_arrays_eq!(encoded, array, &mut ctx);
    Ok(())
}

#[test]
fn round_trip_with_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 2 * BLOCK_SIZE + 3;
    // Leave the whole second block null so that its reference has no valid value to derive from.
    let mut bits = BitBufferMut::new_set(len);
    for i in 0..len {
        if (BLOCK_SIZE..2 * BLOCK_SIZE).contains(&i) || i % 7 == 0 {
            bits.unset(i);
        }
    }
    let values = (0..len)
        .map(|i| (i / BLOCK_SIZE) as i64 * 1_000_000 + (i % 97) as i64)
        .collect::<Buffer<_>>();
    let array = PrimitiveArray::new(values, Validity::from(bits.freeze()));

    let encoded = BlockedFoRData::encode(array.clone(), &mut ctx)?;
    assert_arrays_eq!(encoded, array, &mut ctx);
    Ok(())
}

#[rstest]
#[case(0..0)]
#[case(0..10)]
#[case(5..5)]
#[case(1..BLOCK_SIZE)]
#[case(BLOCK_SIZE..BLOCK_SIZE + 1)]
#[case(7..2 * BLOCK_SIZE + 9)]
#[case(2 * BLOCK_SIZE..3 * BLOCK_SIZE + 7)]
fn slice_round_trip(#[case] range: std::ops::Range<usize>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 3 * BLOCK_SIZE + 7;
    let array = staircase(len);
    let encoded = BlockedFoRData::encode(array.clone(), &mut ctx)?;

    let sliced = encoded.as_ref().slice(range.clone())?;
    let expected = array.as_ref().slice(range)?;
    assert_arrays_eq!(sliced, expected, &mut ctx);
    Ok(())
}

#[test]
fn slice_of_slice_round_trip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = staircase(3 * BLOCK_SIZE + 7);
    let encoded = BlockedFoRData::encode(array.clone(), &mut ctx)?;

    let sliced = encoded.as_ref().slice(500..2500)?.slice(600..1700)?;
    let expected = array.as_ref().slice(500..2500)?.slice(600..1700)?;
    assert_arrays_eq!(sliced, expected, &mut ctx);
    Ok(())
}

#[test]
fn scalar_at_matches_canonical() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 2 * BLOCK_SIZE + 5;
    let array = staircase(len);
    let encoded = BlockedFoRData::encode(array.clone(), &mut ctx)?;
    // Include a sliced array so the block offset is exercised too.
    let sliced = encoded.as_ref().slice(37..len)?;

    for i in [0, 1, BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1, len - 38] {
        assert_eq!(
            sliced.execute_scalar(i, &mut ctx)?,
            array.as_ref().slice(37..len)?.execute_scalar(i, &mut ctx)?,
        );
    }
    Ok(())
}

#[test]
fn bitpacked_child_round_trip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 2 * BLOCK_SIZE + 11;
    let array = PrimitiveArray::from_iter(
        (0..len).map(|i| (i / BLOCK_SIZE) as u32 * 1_000_000 + (i % 97) as u32),
    );
    let encoded = BlockedFoRData::encode(array.clone(), &mut ctx)?;

    // The block-local residuals are all < 128, so they bit-pack to 7 bits.
    let packed = BitPackedData::encode(encoded.encoded(), 7, &mut ctx)?;
    let blocked = BlockedFoR::try_new(packed.into_array(), encoded.references().clone(), 0)?;
    assert_arrays_eq!(blocked, array, &mut ctx);
    Ok(())
}

#[test]
fn rejects_mismatched_reference_count() {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = BlockedFoRData::encode(staircase(2 * BLOCK_SIZE), &mut ctx).unwrap();
    let too_few = encoded.references().slice(0..1).unwrap();
    assert!(BlockedFoR::try_new(encoded.encoded().clone(), too_few, 0).is_err());
}

#[rstest]
#[case(0)]
#[case(3 * BLOCK_SIZE + 7)]
fn serde_round_trip(#[case] len: usize) -> VortexResult<()> {
    let array = staircase(len);
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = BlockedFoRData::encode(array.clone(), &mut ctx)?
        // Slice so a non-zero block offset survives the round trip.
        .as_ref()
        .slice(len.min(37)..len)?;
    let expected = array.as_ref().slice(len.min(37)..len)?;

    let dtype = encoded.dtype().clone();
    let encoded_len = encoded.len();
    let array_ctx = ArrayContext::empty();
    let serialized = encoded.serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;

    let mut concat = ByteBufferMut::empty();
    for buffer in serialized {
        concat.extend_from_slice(buffer.as_ref());
    }
    let decoded = SerializedArray::try_from(concat.freeze())?.decode(
        &dtype,
        encoded_len,
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;

    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

/// The point of per-block references: the single bit width that [`crate::BitPacked`] picks for
/// the residuals is driven by the widest block, not by the spread of the whole array.
#[test]
fn blocked_references_narrow_the_bit_width() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = staircase(16 * BLOCK_SIZE);

    let blocked = BlockedFoRData::encode(array.clone(), &mut ctx)?;
    let global = FoRData::encode(array, &mut ctx)?;

    let bit_width = |residuals: &ArrayRef, ctx: &mut _| -> VortexResult<u8> {
        let residuals = residuals.clone().execute::<PrimitiveArray>(ctx)?;
        let histogram = bit_width_histogram(residuals.as_view(), ctx)?;
        find_best_bit_width(residuals.ptype(), &histogram)
    };

    // Blocks span `i % 97`, the array spans 16 steps of a million.
    assert_eq!(bit_width(blocked.encoded(), &mut ctx)?, 7);
    assert_eq!(bit_width(global.encoded(), &mut ctx)?, 24);
    Ok(())
}
