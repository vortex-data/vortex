// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the chunk layout children of uniformly bit-packed arrays.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::bitpack_compress::bitpack_to_best_bit_width;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// Four full chunks plus a partial trailer, each chunk with a distinctly different magnitude:
/// 3-bit values, 12-bit values with a few 20-bit outliers, all zeros, 20-bit values, and a
/// 5-bit tail.
fn varied(len_tail: usize) -> Vec<u32> {
    (0..4 * FL_CHUNK_SIZE + len_tail)
        .map(|i| {
            let chunk = i / FL_CHUNK_SIZE;
            let pos = (i % FL_CHUNK_SIZE) as u32;
            match chunk {
                0 => pos % 8,
                1 if pos % 300 == 7 => 1 << 20 | pos,
                1 => pos % 4096,
                2 => 0,
                3 => (pos * 977) % (1 << 20),
                _ => pos % 32,
            }
        })
        .collect()
}

fn encode(values: &[u32]) -> VortexResult<BitPackedArray> {
    let mut ctx = SESSION.create_execution_ctx();
    bitpack_to_best_bit_width(&PrimitiveArray::from_iter(values.iter().copied()), &mut ctx)
}

#[rstest]
#[case::decreasing(buffer![0u64, 128, 256, 128])]
#[case::wrong_width(buffer![0u64, 128, 256, 385])]
#[case::out_of_bounds(buffer![0u64, 128, 256, u64::MAX])]
fn invalid_offsets_rejected_before_unpacking(#[case] offsets: Buffer<u64>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..3072u32).map(|i| i % 2));
    let packed = bitpack_to_best_bit_width(&values, &mut ctx)?;
    let widths = packed.width_table().clone();
    let offsets = offsets.into_array();
    assert!(BitPacked::with_chunk_layout(packed, widths, offsets).is_err());
    Ok(())
}

#[test]
fn offset_child_shape_is_validated() -> VortexResult<()> {
    let packed = encode(&varied(100))?;
    let widths = packed.width_table().clone();
    assert!(
        BitPacked::with_chunk_layout(packed.clone(), widths.clone(), buffer![0u64].into_array())
            .is_err()
    );
    let wrong_dtype =
        PrimitiveArray::from_iter(vec![0u32; packed.chunk_offsets().len()]).into_array();
    assert!(BitPacked::with_chunk_layout(packed, widths, wrong_dtype).is_err());
    Ok(())
}

/// Every array carries a non-nullable `u8` width per chunk, including uniform arrays.
#[test]
fn width_table_is_validated() -> VortexResult<()> {
    let packed = encode(&varied(100))?;
    let num_chunks = packed
        .chunk_widths(&mut SESSION.create_execution_ctx())?
        .len();
    let short = PrimitiveArray::from_iter(vec![3u8; num_chunks - 1]).into_array();
    assert!(BitPacked::with_width_table(packed.clone(), short).is_err());
    let wide = PrimitiveArray::from_iter(vec![3u16; num_chunks]).into_array();
    assert!(BitPacked::with_width_table(packed, wide).is_err());

    let uniform = encode(&(0..3000u32).map(|i| i % 128).collect::<Vec<_>>())?;
    assert!(uniform.width_table().is::<Constant>());
    assert_eq!(uniform.bit_width(), 7);
    let mismatched = PrimitiveArray::from_iter([7u8, 6, 7]).into_array();
    assert!(BitPacked::with_width_table(uniform.clone(), mismatched).is_err());
    let table = PrimitiveArray::from_iter(vec![
        7u8;
        uniform
            .chunk_widths(&mut SESSION.create_execution_ctx())?
            .len()
    ])
    .into_array();
    let replaced = BitPacked::with_width_table(uniform.clone(), table)?;
    assert_arrays_eq!(uniform, replaced, &mut SESSION.create_execution_ctx());
    Ok(())
}
