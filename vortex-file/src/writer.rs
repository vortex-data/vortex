// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_stream::try_stream;
use futures::future::ready;
use futures::{Stream, StreamExt, TryStreamExt, pin_mut};
use vortex_array::ArrayContext;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamExt, SendableArrayStream};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::VortexWrite;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::Handle;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::{SequenceId, SequentialStreamAdapter, SequentialStreamExt};
use vortex_layout::{LayoutContext, LayoutStrategy};

use crate::footer::{FileStatistics, FooterFlatBufferWriter, Postscript, PostscriptSegment};
use crate::segments::writer::BufferedSegmentSink;
use crate::{EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION, WriteStrategyBuilder};

/// Configure a new writer, which can eventually be used to write an [`ArrayStream`] into a sink that implements [`VortexWrite`].
///
/// Unless overridden, the default [write strategy][crate::WriteStrategyBuilder] will be used with no
/// additional configuration.
pub struct VortexWriteOptions {
    strategy: Arc<dyn LayoutStrategy>,
    exclude_dtype: bool,
    max_variable_length_statistics_size: usize,
    file_statistics: Vec<Stat>,
}

impl Default for VortexWriteOptions {
    fn default() -> Self {
        Self {
            strategy: WriteStrategyBuilder::new().build(),
            exclude_dtype: false,
            file_statistics: PRUNING_STATS.to_vec(),
            max_variable_length_statistics_size: 64,
        }
    }
}

impl VortexWriteOptions {
    /// Replace the default layout strategy with the provided one.
    pub fn with_strategy(mut self, strategy: Arc<dyn LayoutStrategy>) -> Self {
        self.strategy = strategy;
        self
    }

    /// Exclude the DType from the Vortex file. You must provide the DType to the reader.
    // TODO(ngates): Should we store some sort of DType checksum to make sure the one passed at
    //  read-time is sane? I guess most layouts will have some reasonable validation.
    pub fn exclude_dtype(mut self) -> Self {
        self.exclude_dtype = true;
        self
    }

    /// Configure which statistics to compute at the file-level.
    pub fn with_file_statistics(mut self, file_statistics: Vec<Stat>) -> Self {
        self.file_statistics = file_statistics;
        self
    }
}

impl VortexWriteOptions {
    /// Write to an `ObjectStore` using the provided `VortexWrite` implementation.
    #[cfg(feature = "object_store")]
    pub async fn write_object_store<S: ArrayStream + Unpin + Send + 'static>(
        self,
        object_store: &Arc<dyn object_store::ObjectStore>,
        path: &object_store::path::Path,
        stream: S,
    ) -> VortexResult<()> {
        use futures::future::FutureExt;
        use vortex_io::ObjectStoreWriter;

        self.write_tokio(
            ObjectStoreWriter::new(object_store.clone(), path).await?,
            stream,
        )
        .boxed()
        .await?
        .shutdown()
        .await?;
        Ok(())
    }

    #[cfg(feature = "tokio")]
    pub async fn write_tokio<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<W> {
        self.write(
            write,
            stream,
            vortex_io::runtime::tokio::TokioRuntime::handle(),
        )
        .await
    }

    /// Perform a blocking single-threaded write of the provided stream of `Array`.
    // TODO(ngates): we may just want to not have these APIs to avoid all the feature flags.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_blocking<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<W> {
        vortex_io::runtime::single::SingleThreadRuntime::block_on(|handle| {
            self.write(write, ArrayStreamExt::boxed(stream), handle)
        })
    }

    /// Perform an async write of the provided stream of `Array`.
    pub async fn write<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        mut write: W,
        stream: S,
        handle: Handle,
    ) -> VortexResult<W> {
        let stream = self.write_stream(ArrayStreamExt::boxed(stream), handle);
        pin_mut!(stream);

        while let Some(buffer) = stream.next().await {
            write.write_all(buffer?).await?;
        }
        write.flush().await?;
        Ok(write)
    }

    /// Write an [`ArrayStream`] as a Vortex file by returning a stream of [`ByteBuffer`] that
    /// should be written contiguously.
    pub fn write_stream(
        self,
        stream: SendableArrayStream,
        handle: Handle,
    ) -> impl Stream<Item = VortexResult<ByteBuffer>> {
        // Set up a Context to capture the encodings used in the file.
        let ctx = ArrayContext::empty();

        let dtype = stream.dtype().clone();

        let (mut ptr, eof) = SequenceId::root().split();

        let stream = SequentialStreamAdapter::new(
            dtype.clone(),
            stream
                .try_filter(|chunk| ready(!chunk.is_empty()))
                .map(move |result| result.map(|chunk| (ptr.advance(), chunk))),
        )
        .sendable();
        let (file_stats, stream) = accumulate_stats(
            stream,
            self.file_statistics.clone().into(),
            self.max_variable_length_statistics_size,
        );

        try_stream! {
            // First, write the magic bytes.
            yield ByteBuffer::copy_from(MAGIC_BYTES);
            let mut position = MAGIC_BYTES.len() as u64;

            // Create a channel to send buffers from the segment sink to the output stream.
            let (send, recv) = kanal::bounded_async(16);

            let segments = BufferedSegmentSink::new(send, position);

            // We spawn the layout future so it is driven in the background while we yield the
            // buffer stream, so we don't need to poll it until all buffers have been drained.
            let handle2 = handle.clone();
            let ctx2 = ctx.clone();
            let layout_fut = handle.spawn(async move {
                let layout = self.strategy
                    .write_stream(&ctx2, &segments, stream, eof, handle2)
                    .await?;
                Ok::<_, VortexError>((layout, segments.to_specs()))
            });

            // Yield buffers as they arrive
            let recv_stream = recv.into_stream();
            pin_mut!(recv_stream);
            while let Some(buffer) = recv_stream.next().await {
                let buffer = buffer?;
                if buffer.is_empty() {
                    continue;
                }
                position += buffer.len() as u64;
                yield buffer;
            }

            let (layout, segment_specs) = layout_fut.await?;

            let dtype_segment = if self.exclude_dtype {
                None
            } else {
                let (buffer, dtype_segment) = write_flatbuffer(&mut position, &dtype)?;
                yield buffer;
                Some(dtype_segment)
            };

            let layout_ctx = LayoutContext::empty();
            let (buffer, layout_segment) = write_flatbuffer(
                &mut position,
                &layout.flatbuffer_writer(&layout_ctx),
            )?;
            yield buffer;

            let statistics_segment = if self.file_statistics.is_empty() {
                None
            } else {
                let file_statistics = FileStatistics(file_stats.stats_sets().into());
                let (buffer, stats_segment) = write_flatbuffer(&mut position, &file_statistics)?;
                yield buffer;
                Some(stats_segment)
            };

            let (buffer, footer_segment) = write_flatbuffer(
                &mut position,
                &FooterFlatBufferWriter {
                    ctx: ctx.clone(),
                    layout_ctx,
                    segment_specs: segment_specs.into(),
                },
            )?;
            yield buffer;

            // Assemble the postscript, and write it manually to avoid any framing.
            let postscript = Postscript {
                dtype: dtype_segment,
                layout: layout_segment,
                statistics: statistics_segment,
                footer: footer_segment,
            };
            let postscript_buffer = postscript.write_flatbuffer_bytes();
            if postscript_buffer.len() > MAX_FOOTER_SIZE as usize {
                Err(vortex_err!(
                    "Postscript is too large ({} bytes); max postscript size is {}",
                    postscript_buffer.len(),
                    MAX_FOOTER_SIZE
                ))?;
            }

            let postscript_len = u16::try_from(postscript_buffer.len())
                .vortex_expect("Postscript already verified to fit into u16");
            yield postscript_buffer.into_inner();

            // And finally, the EOF 8-byte footer.
            let mut eof = [0u8; EOF_SIZE];
            eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
            eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
            eof[4..8].copy_from_slice(&MAGIC_BYTES);
            yield ByteBuffer::copy_from(eof);
        }
    }
}

fn write_flatbuffer<F: FlatBufferRoot + WriteFlatBuffer>(
    offset: &mut u64,
    flatbuffer: &F,
) -> VortexResult<(ByteBuffer, PostscriptSegment)> {
    let buffer = flatbuffer.write_flatbuffer_bytes();
    let length = u32::try_from(buffer.len())
        .map_err(|_| vortex_err!("flatbuffer length exceeds maximum u32"))?;

    let segment = PostscriptSegment {
        offset: *offset,
        length,
        alignment: FlatBuffer::alignment(),
    };

    *offset += u64::from(length);

    Ok((buffer.into_inner(), segment))
}
