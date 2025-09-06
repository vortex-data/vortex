// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;
use std::task::Poll;

use async_stream::try_stream;
use futures::{pin_mut, poll, Stream, StreamExt, TryStreamExt};
use vortex_array::stats::{Stat, PRUNING_STATS};
use vortex_array::stream::ArrayStream;
use vortex_array::ArrayContext;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_io::runtime::Handle;
use vortex_io::VortexWrite;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::{SequenceId, SequentialStreamAdapter, SequentialStreamExt};
use vortex_layout::{LayoutContext, LayoutRef, LayoutStrategy, LocalExecutor};

use crate::footer::{FileStatistics, FooterFlatBufferWriter, Postscript, PostscriptSegment};
use crate::segments::writer::BufferedSegmentSink;
use crate::{WriteStrategyBuilder, EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION};

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
            strategy: WriteStrategyBuilder::new()
                .with_executor(Arc::new(LocalExecutor))
                .build(),
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
        use vortex_io::runtime::tokio::TokioRuntime;

        self.write(
            ObjectStoreWriter::new(object_store.clone(), path).await?,
            stream,
            TokioRuntime::handle(),
        )
        .boxed()
        .await?
        .shutdown()
        .await?;
        Ok(())
    }

    /// Perform a blocking single-threaded write of the provided stream of `Array`.
    pub fn write_blocking<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<W> {
        SingleThreadRuntime::block_on(|handle| self.write(write, stream, handle))
    }

    /// Perform an async write of the provided stream of `Array`.
    pub async fn write<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        mut write: W,
        stream: S,
        handle: Handle<'_>,
    ) -> VortexResult<W> {
        let stream = self.write_stream(stream, handle);
        pin_mut!(stream);

        while let Some(buffer) = stream.next().await {
            write.write_all(buffer?).await?;
        }
        write.flush().await?;
        Ok(write)
    }

    pub fn write_stream<S: ArrayStream + Unpin + Send + 'static>(
        &self,
        stream: S,
        handle: Handle,
    ) -> impl Stream<Item = VortexResult<ByteBuffer>> {
        try_stream! {
            // Set up a Context to capture the encodings used in the file.
            let ctx = ArrayContext::empty();

            let dtype = stream.dtype().clone();

            let (mut ptr, eof) = SequenceId::root().split();
            let stream = SequentialStreamAdapter::new(
                dtype.clone(),
                stream
                    .try_filter(|chunk| future::ready(!chunk.is_empty()))
                    .map(move |result| result.map(|chunk| (ptr.advance(), chunk))),
            )
            .sendable();

            let (file_stats, stream) = accumulate_stats(
                stream,
                self.file_statistics.clone().into(),
                self.max_variable_length_statistics_size,
            );

            // Create a segment writer with the initial MAGIC BYTES already written.
            let segments = BufferedSegmentSink::new([ByteBuffer::copy_from(MAGIC_BYTES)]);

            let layout = {
                let layout_fut = self.strategy.write_stream(&ctx, &segments, stream, eof, handle);
                pin_mut!(layout_fut);

                // Now, we sit in a loop polling the layout future and draining the segment writer.
                let layout: LayoutRef;
                loop {
                    // On each iteration, attempt to drain the segment writer to send buffers.
                    for buffer in segments.drain_to_vec() {
                        yield buffer;
                    }

                    // Then we poll the layout future once.
                    if let Poll::Ready(result) = poll!(&mut layout_fut) {
                        layout = result?;
                        // Drain the buffers one last time.
                        for buffer in segments.drain_to_vec() {
                            yield buffer;
                        }
                        break layout;
                    }
                }
            };

            // Once we finish writing our layout, we need to extract the segment specs.
            let mut position = segments.byte_offset();
            let segment_specs = segments.into_specs();

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
