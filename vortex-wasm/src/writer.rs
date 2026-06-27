// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`WasmLayoutStrategy`] writes a [`WasmLayout`]: it delegates the input arrays to a child
//! strategy and appends the embedded kernel as a segment at the end of the file.
//!
//! The kernel is written with a sequence id taken from the end-of-file pointer, so the segment
//! sink flushes it only after every data segment — placing the `.wasm` blob at the end of the
//! file. The child strategy and the kernel write are awaited concurrently to avoid the
//! end-of-file deadlock described on [`LayoutStrategy`].

use std::sync::Arc;

use async_trait::async_trait;
use futures::try_join;
use vortex_array::ArrayContext;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::IntoLayout;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::segments::SegmentSinkRef;
use vortex_layout::sequence::SendableSequentialStream;
use vortex_layout::sequence::SequencePointer;
use vortex_session::VortexSession;

use crate::layout::WasmLayout;

/// A layout strategy that decodes its arrays with an embedded WebAssembly kernel.
///
/// The kernel reconstructs the output from the child layout(s) written by `data`. In the first
/// implementation `data` writes the input stream verbatim and the kernel is an identity decoder;
/// real encodings supply a compressing child stream and a matching kernel.
pub struct WasmLayoutStrategy {
    kernel: ByteBuffer,
    encoding_id: String,
    data: Arc<dyn LayoutStrategy>,
}

impl WasmLayoutStrategy {
    /// Create a strategy from the kernel `.wasm` bytes, a guest encoding id, and the child
    /// strategy used to write the kernel's inputs.
    pub fn new(
        kernel: impl Into<ByteBuffer>,
        encoding_id: impl Into<String>,
        data: Arc<dyn LayoutStrategy>,
    ) -> Self {
        Self {
            kernel: kernel.into(),
            encoding_id: encoding_id.into(),
            data,
        }
    }
}

#[async_trait]
impl LayoutStrategy for WasmLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        // Reserve a sequence position for the data (sorts before EOF) and one for the kernel (the
        // remaining EOF pointer, which sorts after all data).
        let data_eof = eof.split_off();
        let kernel_seq = eof.downgrade();

        let kernel = self.kernel.clone();
        let kernel_sink = Arc::clone(&segment_sink);

        let data_fut =
            self.data
                .write_stream(ctx, Arc::clone(&segment_sink), stream, data_eof, session);
        let kernel_fut = async move { kernel_sink.write(kernel_seq, vec![kernel]).await };

        let (data_layout, kernel_segment) = try_join!(data_fut, kernel_fut)?;

        let row_count = data_layout.row_count();
        let children = crate::layout::wasm_layout_children(vec![data_layout]);
        Ok(WasmLayout::new(
            dtype,
            row_count,
            self.encoding_id.clone(),
            kernel_segment,
            None,
            children,
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.data.buffered_bytes()
    }
}
