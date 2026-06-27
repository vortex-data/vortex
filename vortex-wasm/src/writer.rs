// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`WasmLayoutStrategy`] writes a [`WasmLayout`]: a [`WasmEncoder`] turns each input chunk into a
//! payload (parsed by the guest) plus one child input array, the child is written through a child
//! strategy, and the embedded kernel is appended as a segment at the end of the file.
//!
//! The kernel is written with a sequence id taken from the end-of-file pointer, so the segment
//! sink flushes it only after every data segment. The child writes and the kernel write are driven
//! concurrently to avoid the end-of-file deadlock described on [`LayoutStrategy`].

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::once;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_layout::IntoLayout;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSinkRef;
use vortex_layout::sequence::SendableSequentialStream;
use vortex_layout::sequence::SequencePointer;
use vortex_layout::sequence::SequentialStream;
use vortex_layout::sequence::SequentialStreamAdapter;
use vortex_layout::sequence::SequentialStreamExt;
use vortex_session::VortexSession;

use crate::layout::WasmLayout;
use crate::layout::same_dtype_children;

/// The encoder's output for a single input chunk: a payload the guest parses, plus the single
/// child input array the kernel decodes.
pub struct WasmEncoded {
    /// Encoding-specific header bytes the guest parses (empty for an identity encoding).
    pub payload: ByteBuffer,
    /// The single child input array (e.g. Frame-of-Reference deltas).
    pub child: ArrayRef,
}

/// Write-side counterpart of a WASM decoder kernel: transforms an input chunk into the kernel's
/// payload and child input.
pub trait WasmEncoder: 'static + Send + Sync {
    /// Encode `chunk` into a payload and a single child input array.
    fn encode(&self, chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<WasmEncoded>;
}

/// The trivial encoder: an empty payload and the chunk itself as the child. Pairs with an identity
/// kernel.
pub struct IdentityEncoder;

impl WasmEncoder for IdentityEncoder {
    fn encode(&self, chunk: ArrayRef, _ctx: &mut ExecutionCtx) -> VortexResult<WasmEncoded> {
        Ok(WasmEncoded {
            payload: ByteBuffer::empty(),
            child: chunk,
        })
    }
}

/// A layout strategy that decodes its arrays with an embedded WebAssembly kernel.
pub struct WasmLayoutStrategy {
    kernel: ByteBuffer,
    encoding_id: String,
    encoder: Arc<dyn WasmEncoder>,
    child: Arc<dyn LayoutStrategy>,
}

impl WasmLayoutStrategy {
    /// Create a strategy from the kernel `.wasm` bytes, a guest encoding id, the [`WasmEncoder`]
    /// that produces the kernel's inputs, and the child strategy used to write them.
    pub fn new(
        kernel: impl Into<ByteBuffer>,
        encoding_id: impl Into<String>,
        encoder: Arc<dyn WasmEncoder>,
        child: Arc<dyn LayoutStrategy>,
    ) -> Self {
        Self {
            kernel: kernel.into(),
            encoding_id: encoding_id.into(),
            encoder,
            child,
        }
    }
}

/// Per-chunk results gathered before the kernel segment id is known.
struct ChunkParts {
    row_count: u64,
    dtype: vortex_array::dtype::DType,
    payload_segment: Option<SegmentId>,
    child: LayoutRef,
}

#[async_trait]
impl LayoutStrategy for WasmLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        // The kernel uses the very last sequence position, so its segment is flushed after all data.
        let kernel_seq = eof.downgrade();
        let kernel = self.kernel.clone();
        let kernel_sink = Arc::clone(&segment_sink);
        let kernel_fut = async move { kernel_sink.write(kernel_seq, vec![kernel]).await };

        // Process each chunk into (payload segment, child layout), without yet knowing the kernel
        // segment id (which is only assigned once the kernel write collapses, after all data).
        let process_fut = async {
            let mut parts: Vec<ChunkParts> = Vec::new();
            while let Some(item) = stream.next().await {
                let (seq_id, chunk) = item?;
                let row_count = chunk.len() as u64;
                let out_dtype = chunk.dtype().clone();

                let mut exec = session.create_execution_ctx();
                let WasmEncoded { payload, child } = self.encoder.encode(chunk, &mut exec)?;

                // Sub-sequence positions for this chunk: payload, then the child input.
                let mut ptr = seq_id.descend();
                let payload_segment = if payload.is_empty() {
                    None
                } else {
                    let payload_seq = ptr.advance();
                    Some(segment_sink.write(payload_seq, vec![payload]).await?)
                };

                let child_dtype = child.dtype().clone();
                let child_seq = ptr.advance();
                let child_eof = ptr;
                let child_stream = SequentialStreamAdapter::new(
                    child_dtype,
                    once(std::future::ready(Ok((child_seq, child)))),
                )
                .sendable();
                let child_layout = self
                    .child
                    .write_stream(
                        ctx.clone(),
                        Arc::clone(&segment_sink),
                        child_stream,
                        child_eof,
                        session,
                    )
                    .await?;

                parts.push(ChunkParts {
                    row_count,
                    dtype: out_dtype,
                    payload_segment,
                    child: child_layout,
                });
            }
            Ok::<_, VortexError>(parts)
        };

        let (parts, kernel_segment) = futures::try_join!(process_fut, kernel_fut)?;

        let mut chunks: Vec<LayoutRef> = parts
            .into_iter()
            .map(|p| {
                WasmLayout::new(
                    p.dtype,
                    p.row_count,
                    self.encoding_id.clone(),
                    kernel_segment,
                    p.payload_segment,
                    vec![p.child],
                )
                .into_layout()
            })
            .collect();

        match chunks.len() {
            1 => Ok(chunks.remove(0)),
            _ => {
                let row_count = chunks.iter().map(|c| c.row_count()).sum();
                Ok(ChunkedLayout::new(row_count, dtype, same_dtype_children(chunks)).into_layout())
            }
        }
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes()
    }
}
