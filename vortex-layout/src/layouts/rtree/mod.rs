// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::segments::SegmentSinkRef;
use crate::sequence::{SendableSequentialStream, SequencePointer};
use crate::{LayoutRef, LayoutStrategy};
use async_trait::async_trait;
use vortex_array::ArrayContext;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

pub struct RTreeStrategy {}

#[async_trait]
impl LayoutStrategy for RTreeStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        todo!()
    }
}
