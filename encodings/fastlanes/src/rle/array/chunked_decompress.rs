// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Streaming chunked decompression for RLE arrays.
//!
//! Bit-packed indices, the common case, are unpacked one block at a time; indices in any other
//! streaming encoding stream up and are regrouped into blocks. Either way each 1024-index block is
//! gathered from its chunk's run values while L1-resident, so neither the indices nor the output
//! are materialized.

use fastlanes::BitPacking;
use fastlanes::RLE as FastLanesRLE;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::chunk_iter::BlockDecodeSink;
use vortex_array::chunk_iter::ChunkSink;
use vortex_array::chunk_iter::ScratchChunk;
use vortex_array::chunk_iter::emit_block;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::FL_CHUNK_SIZE;
use crate::RLE;
use crate::rle::RLEArrayExt;
use crate::rle::RLEArraySlotsExt;
use crate::rle::array::rle_decompress::ChunkDecoder;
use crate::rle::array::rle_decompress::load_values_idx_offsets;
use crate::unpack_iter::for_each_packed_chunk;

pub(crate) fn supports_decompress_chunks(array: ArrayView<'_, RLE>) -> bool {
    matches!(array.indices().dtype().as_ptype(), PType::U8 | PType::U16)
        && array.indices().supports_decompress_chunks()
}

pub(crate) fn decompress_chunks(
    array: ArrayView<'_, RLE>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()> {
    if array.is_empty() {
        return Ok(());
    }
    let values_idx_offsets = load_values_idx_offsets(array.values_idx_offsets(), ctx)?;
    let values = array.values().clone().execute::<PrimitiveArray>(ctx)?;
    let indices = array.indices();
    vortex_ensure!(
        indices.len().is_multiple_of(FL_CHUNK_SIZE),
        "RLE indices must hold whole chunks, got {}",
        indices.len()
    );
    let indices_validity = indices.validity()?.execute_mask(indices.len(), ctx)?;
    // `None` means every index is valid.
    let validity_bits = (!indices_validity.all_true()).then(|| indices_validity.to_bit_buffer());
    let num_chunks = (array.offset() + array.len()).div_ceil(FL_CHUNK_SIZE);

    match_each_native_ptype!(values.ptype(), |V| {
        let decoder = ChunkDecoder::try_new(
            values.as_slice::<V>(),
            &values_idx_offsets,
            num_chunks,
            validity_bits.as_ref(),
        )?;
        match indices.dtype().as_ptype() {
            PType::U8 => stream_chunks::<V, u8>(array, decoder, ctx, sink),
            PType::U16 => stream_chunks::<V, u16>(array, decoder, ctx, sink),
            ptype => vortex_bail!("Unsupported index type for RLE decoding: {ptype}"),
        }
    })
}

fn stream_chunks<V, I>(
    array: ArrayView<'_, RLE>,
    decoder: ChunkDecoder<'_, V>,
    ctx: &mut ExecutionCtx,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()>
where
    V: NativePType + FastLanesRLE,
    I: NativePType + Ord + Into<usize> + PhysicalPType<Physical = I> + BitPacking,
{
    // Bit-packed indices, the common case, unpack block by block right here rather than
    // streaming up through their own encoding and being regrouped into blocks.
    if let Some(bp) = array.indices().as_opt::<BitPacked>()
        && bp.patches().is_none()
        && bp.offset() == 0
    {
        let decoder = decoder.with_index_bound(1 << bp.bit_width());
        return stream_bitpacked_indices::<V, I>(array, &decoder, bp, sink);
    }
    let mut adapter = BlockDecodeSink::new(
        array.offset(),
        array.len(),
        V::PTYPE,
        |block, indices: &[I; FL_CHUNK_SIZE], out: &mut [V; FL_CHUNK_SIZE]| {
            decoder.decode(block, indices, out)
        },
        sink,
    );
    array.indices().decompress_chunks(ctx, &mut adapter)
}

/// Decode each block of `array` from its bit-packed indices, unpacked one block at a time.
fn stream_bitpacked_indices<V, I>(
    array: ArrayView<'_, RLE>,
    decoder: &ChunkDecoder<'_, V>,
    bp: ArrayView<'_, BitPacked>,
    sink: &mut dyn ChunkSink,
) -> VortexResult<()>
where
    V: NativePType + FastLanesRLE,
    I: NativePType + Ord + Into<usize> + PhysicalPType<Physical = I> + BitPacking,
{
    let window = array.offset()..array.offset() + array.len();
    let bit_width = bp.bit_width() as usize;
    let mut indices = [I::default(); FL_CHUNK_SIZE];
    let mut output = ScratchChunk::new();
    let mut decode = |block, indices: &[I; FL_CHUNK_SIZE], out: &mut [V; FL_CHUNK_SIZE]| {
        decoder.decode(block, indices, out)
    };
    let mut result = Ok(());
    for_each_packed_chunk::<I, _>(
        bp.packed_slice::<I>(),
        bit_width,
        0,
        bp.len(),
        |packed, rows| {
            let block = rows.start / FL_CHUNK_SIZE;
            if result.is_err() || rows.end <= window.start || rows.start >= window.end {
                return;
            }
            // SAFETY: `packed` holds one chunk at `bit_width`, and `indices` holds a full chunk.
            unsafe { I::unchecked_unpack(bit_width, packed, &mut indices) };
            result = emit_block(
                block,
                &indices,
                &mut decode,
                &mut output,
                window.clone(),
                V::PTYPE,
                &mut *sink,
            );
        },
    )?;
    result
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::test_harness::assert_streams_like_execute;
    use vortex_error::VortexResult;

    use crate::BitPackedData;
    use crate::RLE;
    use crate::RLEData;
    use crate::rle::RLEArraySlotsExt;
    use crate::test::SESSION;

    fn values() -> PrimitiveArray {
        PrimitiveArray::from_option_iter(
            (0..5000u32).map(|i| (i % 37 != 0).then_some(i / 40 * 1_000)),
        )
    }

    /// Rebuild `rle` with its indices re-encoded by `reencode`.
    fn with_indices(rle: &ArrayRef, reencode: impl FnOnce(ArrayRef) -> ArrayRef) -> ArrayRef {
        let rle = rle.as_::<RLE>();
        RLE::try_new(
            rle.values().clone(),
            reencode(rle.indices().clone()),
            rle.values_idx_offsets().clone(),
            0,
            rle.len(),
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    #[case::primitive_indices(0)]
    #[case::bitpacked_indices(1)]
    #[case::unaligned_indices(2)]
    fn rle_streams_like_execute(#[case] indices: u8) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let rle = RLEData::encode(values().as_view(), &mut ctx)?.into_array();
        let array = match indices {
            0 => rle,
            1 => with_indices(&rle, |indices| {
                let indices = indices.execute::<PrimitiveArray>(&mut ctx).unwrap();
                BitPackedData::encode(&indices.into_array(), 6, &mut ctx)
                    .unwrap()
                    .into_array()
            }),
            // Chunks of indices that straddle the RLE chunks are buffered into whole chunks.
            _ => with_indices(&rle, |indices| {
                let pieces = [0, 700, 1500, 2600, 4000, indices.len()]
                    .windows(2)
                    .map(|w| indices.slice(w[0]..w[1]).unwrap())
                    .collect::<Vec<_>>();
                ChunkedArray::try_new(pieces, indices.dtype().clone())
                    .unwrap()
                    .into_array()
            }),
        };
        assert_streams_like_execute(&array, &mut ctx)?;
        assert_streams_like_execute(&array.slice(517..4013)?, &mut ctx)
    }
}
