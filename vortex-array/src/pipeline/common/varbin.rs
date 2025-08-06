// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::BinaryView;
use crate::pipeline::bits::BitView;
use crate::pipeline::buffers::BufferHandle;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{N, Pipeline, PipelineContext};
use std::task::Poll;
use vortex_error::VortexResult;

pub struct VarBinPipeline {
    views_buffer: BufferHandle<BinaryView>,
    data_buffers: Vec<BufferHandle<u8>>,

    len: usize,
    offset: usize,
}

impl Pipeline for VarBinPipeline {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.offset = chunk_idx * N;
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn PipelineContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        todo!()
    }
}
