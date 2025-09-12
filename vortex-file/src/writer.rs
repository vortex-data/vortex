// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::sync::Arc;

use async_stream::try_stream;
use futures::channel::oneshot;
use futures::future::{ready, Fuse, LocalBoxFuture};
use futures::io::AllowStdIo;
use futures::stream::BoxStream;
use futures::{StreamExt, TryFutureExt, TryStreamExt, pin_mut, AsyncWrite, AsyncWriteExt, FutureExt, select, select_biased};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_array::iter::{ArrayIterator, ArrayIteratorExt};
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamExt, SendableArrayStream};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err, vortex_bail, vortex_panic};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::{BlockingRuntime, Handle, Task};
use vortex_io::{AsyncWriteAdapter, VortexWrite};
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::{SequenceId, SequentialStreamAdapter, SequentialStreamExt};
use vortex_layout::{LayoutContext, LayoutStrategy};

use crate::footer::{FileStatistics, FooterFlatBufferWriter, Postscript, PostscriptSegment};
use crate::segments::writer::BufferedSegmentSink;
use crate::{EOF_SIZE, Footer, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION, WriteStrategyBuilder};

/// Configure a new writer, which can eventually be used to write an [`ArrayStream`] into a sink that implements [`VortexWrite`].
///
/// Unless overridden, the default [write strategy][crate::WriteStrategyBuilder] will be used with no
/// additional configuration.
pub struct VortexWriteOptions {
    strategy: Arc<dyn LayoutStrategy>,
    exclude_dtype: bool,
    max_variable_length_statistics_size: usize,
    file_statistics: Vec<Stat>,
    handle: Option<Handle>
}

impl Default for VortexWriteOptions {
    fn default() -> Self {
        Self {
            strategy: WriteStrategyBuilder::new().build(),
            exclude_dtype: false,
            file_statistics: PRUNING_STATS.to_vec(),
            max_variable_length_statistics_size: 64,
            handle: Handle::find()
        }
    }
}

impl VortexWriteOptions {
    /// Configure a [`Handle`] for driving async tasks.
    ///
    /// If not provided, a handle will try to be inferred from [`Handle::find`].
    pub fn with_handle(mut self, handle: Handle) -> Self {
        self.handle = Some(handle);
        self
    }

    /// See [`VortexWriteOptions::with_handle`].
    pub fn with_some_handle(mut self, handle: Option<Handle>) -> Self {
        self.handle = handle.or(self.handle);
        self
    }

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
    ) -> VortexResult<Footer> {
        use vortex_io::ObjectStoreWriter;

        let mut writer = ObjectStoreWriter::new(object_store.clone(), path).await?;
        let footer = self.write_tokio(&mut writer, stream).await?;
        writer.shutdown().await?;

        Ok(footer)
    }

    /// Drop into the blocking writer API using the given runtime.
    pub fn blocking<B: BlockingRuntime + Default>(self) -> BlockingWriter<B> {
        self.with_blocking(B::default())
    }

    /// Drop into the blocking writer API using the given runtime.
    pub fn with_blocking<B: BlockingRuntime>(self, runtime: B) -> BlockingWriter<B> {
        if self.handle.is_some() {
            vortex_panic!("Must not provide or infer a Handle when using the blocking writer API")
        }
        BlockingWriter {
            options: self,
            runtime,
        }
    }

    #[cfg(feature = "tokio")]
    pub async fn write_tokio<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: &mut W,
        stream: S,
    ) -> VortexResult<Footer> {
        self.write(
            write,
            stream,
            vortex_io::runtime::tokio::TokioRuntime::handle(),
        )
        .await
    }

    /// Write the given [`ArrayStream`] into the provided `AsyncWrite` sink.
    pub async fn write<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: &mut W,
        stream: S,
    ) -> VortexResult<Footer> {
        let dtype = stream.dtype().clone();
        let mut writer = self.writer(dtype, write);
        let mut stream = stream.sendable();

        let Some(handle) = self.handle.clone() else {
            vortex_panic!("Must provide a Handle to use the async writer API");
        };

        async move {
            // We spawn a task to push the stream into the writer.
            pin_mut!(stream);
            while let Some(chunk) = stream.next().await {
                writer.push(chunk?).await?;
            }
            writer.finish().await
        }
    }

    /// Create a [`Writer`] that can be used to incrementally write arrays to the file.
    pub fn writer<W: AsyncWrite + Unpin>(
        self,
        dtype: DType,
        write: &mut W,
    ) -> Writer {
        let Some(handle) = self.handle else {
            vortex_panic!("Must provide a Handle to use the async writer API");
        };

        // Create a channel for sending arrays to the layout task.
        let (arrays_send, arrays_recv) = kanal::bounded_async(1);

        // Create a channel for sending byte buffers back from the layout task.
        let (buffers_send, buffers_recv) = kanal::bounded(1);

        // Create a future that writes buffers to the output.
        let write_fut = async move {
            let buffers = buffers_recv.to_async();
            while let Ok(buffer) = buffers.recv().await {
                write.write_all(&buffer?).await?;
            }
            write.flush().await?;
            Ok::<_, VortexError>(())
        }.boxed_local();

        // Set up a Context to capture the encodings used in the file.
        let ctx = ArrayContext::empty();

        let (mut ptr, eof) = SequenceId::root().split();

        let stream = SequentialStreamAdapter::new(
            dtype.clone(),
            arrays_recv
                .into_stream()
                .try_filter(|chunk| ready(!chunk.is_empty()))
                .map(move |result| result.map(|chunk| (ptr.advance(), chunk))),
        )
        .sendable();
        let (file_stats, stream) = accumulate_stats(
            stream,
            self.file_statistics.clone().into(),
            self.max_variable_length_statistics_size,
        );

        // First, write the magic bytes.
        let _ = buffers_send.send(ByteBuffer::copy_from(MAGIC_BYTES));
        let position = MAGIC_BYTES.len() as u64;

        let segments = Arc::new(BufferedSegmentSink::new(buffers_send.clone().to_async(), position));

        // We spawn the layout future so it is driven in the background while we yield the
        // buffer stream, so we don't need to poll it until all buffers have been drained.
        let handle2 = handle.clone();
        let ctx2 = ctx.clone();
        let task = handle.spawn(async move {
            let buffers_send = buffers_send.to_async();

            let layout = self.strategy
                .write_stream(ctx2, segments.clone(), stream, eof, handle2)
                .await?;

            // Close the segment sink, getting the final segment specs and position.
            let (segment_specs, mut position) = segments.close();

            let dtype_segment = if self.exclude_dtype {
                None
            } else {
                let (buffer, dtype_segment) = write_flatbuffer(&mut position, &dtype)?;
                buffers_send.send(buffer).await
                    .map_err(|_| vortex_err!("Buffer sink closed"))?;
                Some(dtype_segment)
            };

            let layout_ctx = LayoutContext::empty();
            let (buffer, layout_segment) = write_flatbuffer(
                &mut position,
                &layout.flatbuffer_writer(&layout_ctx),
            )?;
            buffers_send.send(buffer).await
                .map_err(|_| vortex_err!("Buffer sink closed"))?;

            let (statistics_segment, file_statistics) = if self.file_statistics.is_empty() {
                (None, None)
            } else {
                let file_statistics = FileStatistics(file_stats.stats_sets().into());
                let (buffer, stats_segment) = write_flatbuffer(&mut position, &file_statistics)?;
                buffers_send.send(buffer).await
                    .map_err(|_| vortex_err!("Buffer sink closed"))?;
                (Some(stats_segment), Some(file_statistics))
            };

            let footer = Footer::new(
                layout.clone(),
                segment_specs.clone(),
                file_statistics,
            );

            let (buffer, footer_segment) = write_flatbuffer(
                &mut position,
                &FooterFlatBufferWriter {
                    ctx: ctx.clone(),
                    layout_ctx,
                    segment_specs,
                },
            )?;
            buffers_send.send(buffer).await
                .map_err(|_| vortex_err!("Buffer sink closed"))?;

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
            buffers_send.send(postscript_buffer.into_inner()).await
                .map_err(|_| vortex_err!("Buffer sink closed"))?;

            // And finally, the EOF 8-byte footer.
            let mut eof = [0u8; EOF_SIZE];
            eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
            eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
            eof[4..8].copy_from_slice(&MAGIC_BYTES);
            buffers_send.send(ByteBuffer::copy_from(eof)).await
                .map_err(|_| vortex_err!("Buffer sink closed"))?;

            Ok::<_, VortexError>(footer)
        });

        Writer {
            write: write_fut,
            arrays: arrays_send,
            task: Some(task),
            handle,
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

/// An async API for writing Vortex files.
pub struct Writer<'w> {
    // A future that writes buffers to the output.
    write: Fuse<LocalBoxFuture<'w, VortexResult<()>>>,
    // The input channel for sending arrays to the writer.
    arrays: kanal::AsyncSender<VortexResult<ArrayRef>>,
    // The writer task that ultimately produces the footer.
    task: Option<Task<VortexResult<Footer>>>,
    handle: Handle,
}

impl<'w> Writer<'w> {
    /// Push a new chunk into the writer.
    ///
    /// This function works by first writing enough buffers to ensure the
    pub async fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        let mut send_fut = self.arrays.send(Ok(chunk)).fuse();

        // We select over the write future and sending on the input channel.
        select! {
            res = &mut self.write => {
                // If writing fails, propagate the error.
                res?;
                // If writing has _finished_, we have a problem in the layout task.
                return self.handle_failed_task().await;
            },

            res = send_fut => {
                if let Err(e) = res {
                    // If sending fails, the receiver has been dropped, which means the layout
                    // future will have failed or panicked.
                    return self.handle_failed_task().await;
                }
            }
        }

        Ok(())
    }

    /// Push an entire [`ArrayStream`] into the writer, consuming it.
    ///
    /// A task is spawned to consume the stream and push it into the writer, with the current
    /// thread being used to write buffers to the output.
    pub async fn push_stream(&mut self, mut stream: SendableArrayStream) -> VortexResult<()> {
        let arrays = self.arrays.clone();
        self.handle.spawn(async move {
            while let Some(chunk) = stream.next().await {
                if let Err(e) = arrays.send(chunk).await {
                    // If the arrays channel is closed, the layout task has failed or panicked.
                    break;
                }
            }
        });

        while let Some(buffer) = self.buffers.next().await {
            self.write.write_all(&buffer?).await?;
        }

        Ok(())
    }

    /// Finish writing the Vortex file, flushing any remaining buffers and returning the
    /// new file's footer.
    pub async fn finish(mut self) -> VortexResult<Footer> {
        // Close the input channel to signal EOF.
        if let Err(e) = self.arrays.close() {
            vortex_bail!("Error closing writer channel: {}", e);
        }

        // Write any remaining buffers.
        while let Some(buffer) = self.buffers.next().await {
            self.write.write_all(&buffer?).await?;
        }

        // Flush the output.
        self.write.flush().await?;

        // Await the layout task.
        let Some(task) = self.task.take() else {
            vortex_bail!("Internal error: writer task already consumed");
        };
        task.await
    }

    async fn handle_failed_task(&mut self) -> VortexResult<()> {
        let Some(task) = self.task.take() else {
            vortex_bail!("Internal error: writer task already consumed");
        };
        task.await?;
        vortex_bail!("Internal error: writer task completed successfully but write future finished early")
    }
}

/// A blocking API for writing Vortex files.
pub struct BlockingWriter<B: BlockingRuntime> {
    options: VortexWriteOptions,
    runtime: B,
}

impl<B: BlockingRuntime> BlockingWriter<B> {
    /// Write a Vortex file into the given `Write` sink.
    pub fn write(
        self,
        write: &mut impl Write,
        iter: impl ArrayIterator + Send + 'static,
    ) -> VortexResult<Footer> {
        self.runtime.block_on(|handle| async move {
            let writer = self.options
                .writer(
                    iter.dtype().clone(),
                    &mut AsyncWriteAdapter(AllowStdIo::new(write))
                );
                .await
        })
    }
}
