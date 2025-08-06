// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::{Compressor, Symbol};
use std::task::{Poll, ready};
use vortex_array::arrays::BinaryView;
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::buffers::BufferHandle;
use vortex_array::pipeline::vector::PrimitiveVector;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{Pipeline, PipelineContext};
use vortex_buffer::ByteBufferMut;
use vortex_error::{VortexExpect, VortexResult};

pub struct FSSTPipeline {
    len: usize,

    symbols_buffer: BufferHandle<Symbol>,
    symbols_lens_buffer: BufferHandle<u8>,

    compressor: Option<Compressor>,

    codes_offsets: Box<dyn Pipeline>,
    codes_offsets_vec: PrimitiveVector<u32>,
    codes_buffer: BufferHandle<u8>,

    uncompressed_lens: Box<dyn Pipeline>,
    uncompressed_lens_vec: PrimitiveVector<u32>,
}

impl Pipeline for FSSTPipeline {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.codes_offsets.seek(chunk_idx)?;
        self.uncompressed_lens.seek(chunk_idx)
    }

    fn step(
        &mut self,
        ctx: &dyn PipelineContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        // TODO(ngates): use a join! macro here to ensure we try to fetch all buffers before
        //  returning pending.
        let symbols = ready!(self.symbols_buffer.get_or_load(ctx))?;
        let symbol_lens = ready!(self.symbols_lens_buffer.get_or_load(ctx))?;
        let codes = ready!(self.codes_buffer.get_or_load(ctx))?;

        let compressor = self
            .compressor
            .get_or_insert_with(|| Compressor::rebuild_from(symbols, symbol_lens));

        // We do not push down the selection mask for offsets, since we need adjacent offsets.
        let mut codes_offsets_mut = self.codes_offsets_vec.as_view_mut();
        ready!(
            self.codes_offsets
                .step(ctx, BitView::all_true(), &mut codes_offsets_mut)
        )?;
        let codes_offsets = codes_offsets_mut.as_ref::<u32>();

        // But we do push down the selection mask for uncompressed lengths.
        // These lengths are only used to size the output buffer.
        let mut uncompressed_lens_mut = self.uncompressed_lens_vec.as_view_mut();
        ready!(
            self.uncompressed_lens
                .step(ctx, selected, &mut uncompressed_lens_mut)
        )?;
        let uncompressed_lens = uncompressed_lens_mut.as_ref::<u32>();

        // TODO(ngates): this is probably slow :(
        let mut output_size = 0;
        selected.iter_ones(|idx| {
            output_size += uncompressed_lens[idx] as usize;
        });

        let mut uncompressed = ByteBufferMut::with_capacity(output_size + 7);
        let mut views_out = out.elements::<BinaryView>();

        // TODO(ngates): iterate the mask as indices, slices, or all-true.
        let view_idx = 0;
        selected.iter_ones(|idx| {
            let codes_range = codes_offsets[idx] as usize..codes_offsets[idx + 1] as usize;
            let len = compressor
                .decompressor()
                .decompress_into(&codes[codes_range], uncompressed.spare_capacity_mut());
            let offset = uncompressed.len();
            unsafe { uncompressed.set_len(offset + len) };
            views_out[view_idx] = BinaryView::make_view(
                &uncompressed.as_slice()[offset..],
                0,
                u32::try_from(offset).vortex_expect("FSSTPipeline: offset overflow"),
            )
        });
        unsafe { uncompressed.set_len(output_size) };

        out.add_buffer(uncompressed.freeze());
        out.set_selection_len(selected.true_count());

        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod test {
    use crate::tests::build_fsst_array;
    use vortex_array::Array;

    #[test]
    fn test_fsst_pipeline() {
        let fsst_array = build_fsst_array();
        let pipeline = fsst_array.to_pipeline().unwrap();
    }
}
