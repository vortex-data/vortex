// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Behavioural tests for bit-packed arrays whose chunks are packed at different widths.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::scalar::Scalar;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
use crate::ChunkWidths;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::bitpack_compress::bitpack_encode_with_widths;
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
    let widths = ChunkWidths::new(Buffer::from_iter(values.chunks(FL_CHUNK_SIZE).map(
        |chunk| {
            chunk
                .iter()
                .map(|v| (u32::BITS - v.leading_zeros()) as u8)
                .max()
                .unwrap_or(0)
        },
    )));
    bitpack_encode_with_widths(
        &PrimitiveArray::from_iter(values.iter().copied()),
        widths,
        &mut ctx,
    )
}

#[test]
fn explicit_widths_including_full_width_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<u16> = (0..2 * FL_CHUNK_SIZE + 10)
        .map(|i| {
            if i < FL_CHUNK_SIZE {
                (i % 16) as u16
            } else {
                u16::MAX - i as u16
            }
        })
        .collect();
    let array = PrimitiveArray::from_iter(values.iter().copied());
    // Chunk 1 and the tail use the full 16 bits, which a single global width could never pick.
    let widths = ChunkWidths::new(buffer![4u8, 16, 16]);
    let packed = bitpack_encode_with_widths(&array, widths, &mut ctx)?;
    assert!(packed.patches().is_none());
    assert_eq!(
        packed
            .chunk_widths(&mut SESSION.create_execution_ctx())?
            .max_width(),
        16
    );
    assert_arrays_eq!(packed, array, &mut ctx);
    Ok(())
}

#[test]
fn slice_preserves_offset_origin() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..3072u32).map(|i| match i / 1024 {
        0 => i % 8,
        1 => i % 32,
        _ => i % 4,
    }));
    let packed =
        bitpack_encode_with_widths(&values, ChunkWidths::new(buffer![3u8, 5, 2]), &mut ctx)?
            .into_array();
    let slice = SliceArray::new(packed.clone(), 1100..2300).into_array();
    let sliced = packed
        .reduce_parent(&slice, 0)?
        .ok_or_else(|| vortex_err!("expected bitpacked slice"))?;
    let bp = sliced.as_::<BitPacked>();
    assert_eq!(bp.offset(), 76);
    assert_eq!(bp.packed().len(), 896);
    assert_arrays_eq!(
        bp.chunk_offsets(),
        buffer![384u64, 1024, 1280].into_array(),
        &mut ctx
    );
    assert_arrays_eq!(bp.width_table(), buffer![5u8, 2].into_array(), &mut ctx);
    // Both primitive children and the packed bytes remain views into the original buffers.
    let original = packed.as_::<BitPacked>();
    assert_eq!(
        bp.packed().as_host().as_ptr(),
        original.packed().as_host()[384..].as_ptr()
    );
    assert_eq!(
        bp.chunk_offsets()
            .as_::<Primitive>()
            .as_slice::<u64>()
            .as_ptr(),
        original
            .chunk_offsets()
            .as_::<Primitive>()
            .as_slice::<u64>()[1..]
            .as_ptr()
    );
    let read = sliced.clone();
    assert_arrays_eq!(
        read,
        values.clone().into_array().slice(1100..2300)?,
        &mut ctx
    );
    let nested = sliced.slice(1000..1200)?;
    assert_arrays_eq!(nested, values.into_array().slice(2100..2300)?, &mut ctx);
    Ok(())
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
    assert!(BitPacked::with_chunk_layout(packed.clone(), widths.clone(), offsets.clone()).is_err());
    let offsets = offsets.execute::<PrimitiveArray>(&mut ctx)?;
    let offsets = bitpack_to_best_bit_width(&offsets, &mut ctx)?.into_array();
    let packed = BitPacked::with_chunk_layout(packed, widths, offsets)?.into_array();
    // An isolated scalar checks only its own chunk; bulk unpacking validates the whole layout.
    assert_eq!(packed.execute_scalar(1, &mut ctx)?, Scalar::from(1u32));
    assert!(packed.execute_scalar(2048, &mut ctx).is_err());
    assert!(packed.execute::<PrimitiveArray>(&mut ctx).is_err());
    Ok(())
}

#[test]
fn slice_rejects_unaligned_offsets() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::from_iter((0..3072u32).map(|i| i % 2));
    let packed = bitpack_to_best_bit_width(&values, &mut ctx)?;
    let widths = packed.width_table().clone();
    let offsets = PrimitiveArray::from_iter([0u64, 127, 255, 383]);
    let offsets = bitpack_to_best_bit_width(&offsets, &mut ctx)?.into_array();
    let packed = BitPacked::with_chunk_layout(packed, widths, offsets)?;
    assert!(<BitPacked as SliceKernel>::slice(packed.as_view(), 1024..2048, &mut ctx).is_err());
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
