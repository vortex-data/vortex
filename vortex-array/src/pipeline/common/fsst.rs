// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::bits::BitView;
use crate::pipeline::buffers::BufferHandle;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Pipeline, PipelineContext};
use std::task::Poll;
use vortex_error::VortexResult;

pub struct FSSTPipeline {
    symbols_buffer: BufferHandle<Symbol>,
}

impl Pipeline for FSSTPipeline {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        // Implement seeking logic here
        todo!()
    }

    fn step(
        &mut self,
        ctx: &dyn PipelineContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        // Implement stepping logic here
        todo!()
    }
}
