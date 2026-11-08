// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedData;
use crate::Delta;
use crate::DeltaArraySlotsExt;
use crate::FL_CHUNK_SIZE;
use crate::RLE;
use crate::RLEArray;
use crate::rle::RLEArrayExt;
use crate::rle::RLEArraySlotsExt;
use crate::rle::array::gather::U8_LANES;
use crate::rle::array::rle_decompress::max_lane_index;
use crate::rle::array::rle_decompress::rle_decompress;
use crate::test::SESSION;

fn indices_with_oob(oob: u16) -> ArrayRef {
    let mut indices = [0u16, 1]
        .iter()
        .cycle()
        .take(FL_CHUNK_SIZE)
        .copied()
        .collect::<Vec<_>>();
    indices[100] = oob;
    PrimitiveArray::from_iter(indices).into_array()
}

#[test]
fn test_decode_rejects_out_of_bounds_index() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
    let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

    // Index 999 points far beyond the 2 values of the only chunk.
    let rle = RLE::try_new(
        values,
        indices_with_oob(999),
        values_idx_offsets,
        0,
        FL_CHUNK_SIZE,
    )?;
    assert!(rle_decompress(&rle, &mut ctx).is_err());
    Ok(())
}

#[test]
fn test_decode_rejects_index_into_next_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Two chunks with 2 values each: an index of 2 in chunk 0 is within `values`
    // but out of bounds for the chunk, and must be rejected rather than silently
    // reading chunk 1's values.
    let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
    let values_idx_offsets = PrimitiveArray::from_iter([0u64, 2]).into_array();
    let mut indices = [0u16, 1].repeat(FL_CHUNK_SIZE);
    indices[100] = 2;
    let indices = PrimitiveArray::from_iter(indices).into_array();

    let rle = RLE::try_new(values, indices, values_idx_offsets, 0, 2 * FL_CHUNK_SIZE)?;
    assert!(rle_decompress(&rle, &mut ctx).is_err());
    Ok(())
}

#[test]
fn test_decode_rejects_non_monotonic_offsets() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
    let values_idx_offsets = PrimitiveArray::from_iter([2u64, 0]).into_array();
    let indices = PrimitiveArray::from_iter([0u16, 1].repeat(FL_CHUNK_SIZE)).into_array();

    let rle = RLE::try_new(values, indices, values_idx_offsets, 0, 2 * FL_CHUNK_SIZE)?;
    assert!(rle_decompress(&rle, &mut ctx).is_err());
    Ok(())
}

#[test]
fn test_decode_rejects_offsets_beyond_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([10u32, 20, 30, 40]).into_array();
    // The offset span (100) exceeds the number of values present (4).
    let values_idx_offsets = PrimitiveArray::from_iter([0u64, 100]).into_array();
    let indices = PrimitiveArray::from_iter([0u16, 1].repeat(FL_CHUNK_SIZE)).into_array();

    let rle = RLE::try_new(values, indices, values_idx_offsets, 0, 2 * FL_CHUNK_SIZE)?;
    assert!(rle_decompress(&rle, &mut ctx).is_err());
    Ok(())
}

#[test]
fn test_decode_rejects_empty_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
    // Chunk 0 references no values: offsets [0, 0].
    let values_idx_offsets = PrimitiveArray::from_iter([0u64, 0]).into_array();
    let indices = PrimitiveArray::from_iter([0u16, 1].repeat(FL_CHUNK_SIZE)).into_array();

    let rle = RLE::try_new(values, indices, values_idx_offsets, 0, 2 * FL_CHUNK_SIZE)?;
    assert!(rle_decompress(&rle, &mut ctx).is_err());
    Ok(())
}

#[test]
fn test_decode_rejects_out_of_bounds_index_at_valid_position() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
    let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

    // Position 100 is VALID but holds an out-of-bounds index: this is corruption
    // and must fail, even though other positions are null.
    let mut indices = [0u16, 1].repeat(FL_CHUNK_SIZE / 2);
    indices[100] = 999;
    let mut validity = vec![true; FL_CHUNK_SIZE];
    validity[200] = false;
    let indices = PrimitiveArray::new(
        indices.into_iter().collect::<Buffer<u16>>(),
        Validity::from_iter(validity),
    )
    .into_array();

    let rle = RLE::try_new(values, indices, values_idx_offsets, 0, FL_CHUNK_SIZE)?;
    assert!(rle_decompress(&rle, &mut ctx).is_err());
    Ok(())
}

#[test]
fn test_decode_tolerates_garbage_index_at_null_position() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
    let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();

    // Position 100 is NULL and holds a garbage out-of-bounds index (e.g. after
    // further compression of the indices): its value is masked, so decoding
    // must succeed and every valid position must decode correctly.
    let mut indices = [0u16, 1].repeat(FL_CHUNK_SIZE / 2);
    indices[100] = 999;
    let mut validity = vec![true; FL_CHUNK_SIZE];
    validity[100] = false;
    let indices = PrimitiveArray::new(
        indices.into_iter().collect::<Buffer<u16>>(),
        Validity::from_iter(validity),
    )
    .into_array();

    let rle = RLE::try_new(values, indices, values_idx_offsets, 0, FL_CHUNK_SIZE)?;
    let decoded = rle_decompress(&rle, &mut ctx)?;
    let expected = PrimitiveArray::from_option_iter(
        (0..FL_CHUNK_SIZE).map(|i| (i != 100).then_some(if i % 2 == 0 { 10u32 } else { 20 })),
    );
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_decode_in_bounds_indices_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter([10u32, 20]).into_array();
    let values_idx_offsets = PrimitiveArray::from_iter([0u64]).into_array();
    let indices = PrimitiveArray::from_iter([0u16, 1].iter().cycle().take(FL_CHUNK_SIZE).copied())
        .into_array();

    let rle = RLE::try_new(values, indices, values_idx_offsets, 0, FL_CHUNK_SIZE)?;
    let decoded = rle_decompress(&rle, &mut ctx)?;
    let expected =
        PrimitiveArray::from_iter([10u32, 20].iter().cycle().take(FL_CHUNK_SIZE).copied());
    assert_arrays_eq!(decoded, expected, &mut ctx);
    Ok(())
}

/// How the RLE indices are stored.
#[derive(Clone, Copy, Debug)]
enum Indices {
    Primitive,
    BitPacked,
    BlockedBitPacked,
    Delta,
}

/// `len` `i32` values in runs of `1..=max_run` (or exactly `max_run` when `fixed`), every
/// seventh run null when `nullable`.
fn runs(len: usize, max_run: usize, fixed: bool, nullable: bool) -> PrimitiveArray {
    let mut rng = StdRng::seed_from_u64(max_run as u64);
    let mut values = Vec::with_capacity(len);
    let mut run_idx = 0;
    while values.len() < len {
        let run = if fixed {
            max_run
        } else {
            rng.random_range(1..=max_run)
        };
        let value = (!nullable || run_idx % 7 != 3).then(|| rng.random_range(0..1_000_000i32));
        values.extend(std::iter::repeat_n(value, run));
        run_idx += 1;
    }
    values.truncate(len);
    if nullable {
        PrimitiveArray::from_option_iter(values)
    } else {
        PrimitiveArray::from_iter(values.into_iter().map(Option::unwrap))
    }
}

fn bits_needed(max: u64) -> u8 {
    u8::try_from(u64::BITS - max.leading_zeros()).unwrap()
}

/// Bit-pack `array` at the width its maximum needs, or keep it as is when that width is its
/// native width, which BitPacked cannot represent.
fn bitpack(array: PrimitiveArray, max: u64, blocked: bool) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let width = bits_needed(max);
    if usize::from(width) >= array.ptype().bit_width() {
        return Ok(array.into_array());
    }
    Ok(if blocked {
        let widths = vec![width; array.len().div_ceil(FL_CHUNK_SIZE)];
        BitPackedData::encode_blocked(&array.into_array(), &widths, &mut ctx)?.into_array()
    } else {
        BitPacked::encode(&array.into_array(), width, &mut ctx)?.into_array()
    })
}

/// RLE-encode `array` with narrowed indices stored as `encoding`.
fn rle_with_indices(array: &PrimitiveArray, encoding: Indices) -> VortexResult<RLEArray> {
    let mut ctx = SESSION.create_execution_ctx();
    let rle = RLE::encode(array.as_view(), &mut ctx)?;
    let indices = rle
        .indices()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?
        .narrow(&mut ctx)?;
    let max = indices
        .statistics()
        .compute_max::<u64>(&mut ctx)
        .unwrap_or_default();
    let indices: ArrayRef = match encoding {
        Indices::Primitive => indices.into_array(),
        Indices::BitPacked => bitpack(indices, max, false)?,
        Indices::BlockedBitPacked => bitpack(indices, max, true)?,
        Indices::Delta => {
            let delta = Delta::try_from_primitive_array(&indices, &mut ctx)?;
            let deltas = delta.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
            let max_delta = deltas
                .statistics()
                .compute_max::<u64>(&mut ctx)
                .unwrap_or_default();
            let deltas = if max_delta == 0 {
                ConstantArray::new(0u8, deltas.len()).into_array()
            } else {
                bitpack(deltas, max_delta, false)?
            };
            Delta::try_new(
                delta.bases().clone(),
                deltas,
                indices.validity()?,
                0,
                indices.len(),
            )?
            .into_array()
        }
    };
    RLE::try_new(
        rle.values().clone(),
        indices,
        rle.values_idx_offsets().clone(),
        rle.offset(),
        rle.len(),
    )
}

#[rstest]
fn decode_any_index_encoding(
    #[values(
        Indices::Primitive,
        Indices::BitPacked,
        Indices::BlockedBitPacked,
        Indices::Delta
    )]
    encoding: Indices,
    #[values((1, false), (8, true), (16, true), (8, false), (15, false), (64, false))] runs_of: (
        usize,
        bool,
    ),
    #[values(false, true)] nullable: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let (max_run, fixed) = runs_of;
    let len = 10 * FL_CHUNK_SIZE + 300;
    let expected = runs(len, max_run, fixed, nullable);
    let rle = rle_with_indices(&expected, encoding)?.into_array();

    assert_arrays_eq!(rle, expected, &mut ctx);
    for range in [0..len, 1024..3072, 1000..5000, 5..17, 4096..len] {
        assert_arrays_eq!(
            rle.slice(range.clone())?,
            expected.clone().into_array().slice(range)?,
            &mut ctx
        );
    }
    Ok(())
}

#[rstest]
#[case::small(10, 0b0000_0111, 13)]
#[case::exactly_254(254, 0, 254)]
#[case::exactly_255(255, 0, 255)]
#[case::saturates_to_255(247, 0xFF, 255)]
#[case::past_255(250, 0xFF, 258)]
fn max_lane_index_is_exact(#[case] base: u8, #[case] bits: u8, #[case] expected: usize) {
    let mut bases = [0u8; U8_LANES];
    let mut deltas = [0u8; U8_LANES];
    bases[77] = base;
    deltas[77] = bits;
    assert_eq!(max_lane_index(&bases, Some(&deltas)), expected);
    assert_eq!(max_lane_index(&bases, None), usize::from(base));
}
