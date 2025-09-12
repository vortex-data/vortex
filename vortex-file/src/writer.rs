// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use async_stream::__private::AsyncStream;
use async_stream::try_stream;
use futures::channel::oneshot;
use futures::future::{ready, Fuse, FusedFuture, LocalBoxFuture};
use futures::io::{AllowStdIo};
use futures::{StreamExt, TryFutureExt, TryStreamExt, pin_mut, AsyncWrite, AsyncWriteExt, FutureExt, select};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_array::iter::{ArrayIterator, ArrayIteratorExt};
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt, SendableArrayStream};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_err, vortex_bail, vortex_panic};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::{BlockingRuntime, Handle};
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::{SequenceId, SequentialStreamAdapter, SequentialStreamExt};
use vortex_layout::{LayoutContext, LayoutStrategy};

use crate::footer::{FileStatistics, FooterFlatBufferWriter, Postscript, PostscriptSegment};
use crate::segments::writer::BufferedSegmentSink;
use crate::{EOF_SIZE, Footer, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION, WriteStrategyBuilder};
use crate::counting::CountingAsyncWrite;

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

    /// Write an [`ArrayStream`] as a Vortex file.
    ///
    /// Note that buffers are flushed as soon as they are available with no buffering, the caller
    /// is responsible for deciding how to configure buffering on the underlying `Write` sink.
    pub async fn write<W: AsyncWrite + Unpin, S: ArrayStream + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<Footer> {
        self.write_internal(write, ArrayStreamExt::boxed(stream)).await
    }

    async fn write_internal<W: AsyncWrite + Unpin>(self,
                                                   mut write: W,
                                                   stream: SendableArrayStream,
    ) -> VortexResult<Footer> {
    let Some(handle) = self.handle else {
            vortex_panic!("Must provide a Handle to use the async writer API");
        };

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

        let (footer_send, footer_recv) = oneshot::channel();

        let byte_stream = try_stream! {
            // First, write the magic bytes.
            yield ByteBuffer::copy_from(MAGIC_BYTES);
            let mut position = MAGIC_BYTES.len() as u64;

            // Create a channel to send buffers from the segment sink to the output stream.
            let (send, recv) = kanal::bounded_async(16);

            let segments = Arc::new(BufferedSegmentSink::new(send, position));

            // We spawn the layout future so it is driven in the background while we yield the
            // buffer stream, so we don't need to poll it until all buffers have been drained.
            let handle2 = handle.clone();
            let ctx2 = ctx.clone();
            let layout_fut = handle.spawn(async move {
                let layout = self.strategy
                    .write_stream(ctx2, segments.clone(), stream, eof, handle2)
                    .await?;
                Ok::<_, VortexError>((layout, segments.segment_specs()))
            });

            // Yield buffers as they arrive
            let recv_stream = recv.into_stream();
            pin_mut!(recv_stream);
            while let Some(buffer) = recv_stream.next().await {
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

            let (statistics_segment, file_statistics) = if self.file_statistics.is_empty() {
                (None, None)
            } else {
                let file_statistics = FileStatistics(file_stats.stats_sets().into());
                let (buffer, stats_segment) = write_flatbuffer(&mut position, &file_statistics)?;
                yield buffer;
                (Some(stats_segment), Some(file_statistics))
            };

            // Return a Footer object via the oneshot channel.
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

            // Emit the footer to the caller.
            let _ = footer_send.send(footer);

            Ok::<_, VortexError>(())
        };

        pin_mut!(byte_stream);
        while let Some(buffer) = byte_stream.next().await {
            write.write_all(&buffer?).await?;
        }
        write.flush().await?;

        footer_recv
            .map_err(|_canceled| vortex_err!("Cannot return Footer from failed write"))
            .await
    }

    /// Create a push-based [`Writer`] that can be used to incrementally write arrays to the file.
    pub fn writer<'w, W: AsyncWrite + Unpin + 'w>(
        self,
        write: W,
        dtype: DType,
    ) -> Writer<'w> {
        // Create a channel for sending arrays to the layout task.
        let (arrays_send, arrays_recv) = kanal::bounded_async(1);
        let arrays = ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays_recv.into_stream()));

        let write = CountingAsyncWrite::new(write);
        let bytes_written = write.counter();
        let future = self.write(write, arrays).boxed_local().fuse();

        Writer {
            arrays: arrays_send,
            future,
            bytes_written,
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
    // The input channel for sending arrays to the writer.
    arrays: kanal::AsyncSender<VortexResult<ArrayRef>>,
    // The writer task that ultimately produces the footer.
    future: Fuse<LocalBoxFuture<'w, VortexResult<Footer>>>,
    // The bytes written so far.
    bytes_written: Arc<AtomicU64>,
}

impl Writer<'_> {
    /// Push a new chunk into the writer.
    pub async fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        let arrays = self.arrays.clone();
        let send_fut = async move { arrays.send(Ok(chunk)).await }.fuse();
        pin_mut!(send_fut);

        // We poll the writer future to continue writing bytes to the output, while waiting for
        // enough room to push the next chunk into the channel.
        select! {
            result = send_fut => {
                // If the send future failed, the writer has failed or panicked.
                if result.is_err() {
                    return Err(self.handle_failed_task().await);
                }
            },
            result = &mut self.future => {
                // Under normal operation, the writer future should never complete until
                // finish() is called. Therefore, we can assume the writer has failed.
                // The writer future has failed, we need to propagate the error.
                if result.is_ok() {
                    vortex_bail!("Internal error: writer future completed early");
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
        let stream_fut = async move {
            while let Some(chunk) = stream.next().await {
                arrays.send(chunk).await?;
            }
            Ok::<_, kanal::SendError>(())
        }.fuse();
        pin_mut!(stream_fut);

        // We poll the writer future to continue writing bytes to the output, while waiting for
        // enough room to push the stream into the channel.
        select! {
            result = stream_fut => {
                if let Err(_send_err) = result {
                    // If the send future failed, the writer has failed or panicked.
                    return Err(self.handle_failed_task().await);
                }
            }

            result = &mut self.future => {
                // Under normal operation, the writer future should never complete until
                // finish() is called. Therefore, we can assume the writer has failed.
                // The writer future has failed, we need to propagate the error.
                if result.is_ok() {
                    vortex_bail!("Internal error: writer future completed early");
                }
            }
        }

        Ok(())
    }

    /// Returns the number of bytes written to the file so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Finish writing the Vortex file, flushing any remaining buffers and returning the
    /// new file's footer.
    pub async fn finish(self) -> VortexResult<Footer> {
        // Close the input channel to signal EOF.
        if let Err(e) = self.arrays.close() {
            vortex_bail!("Error closing writer channel: {}", e);
        }

        // Await the future task.
        if self.future.is_terminated() {
            vortex_bail!("Internal error: writer task already consumed");
        }
        self.future.await
    }

    /// Assuming the writer task has failed, await it to get the error.
    async fn handle_failed_task(&mut self) -> VortexError {
        if self.future.is_terminated() {
            return vortex_err!("Internal error: writer task already consumed");
        }
        match (&mut self.future).await {
            Ok(_) => vortex_err!("Internal error: writer task completed successfully but write future finished early"),
            Err(e) => e,
        }
    }
}

/// A blocking API for writing Vortex files.
pub struct BlockingWriter<B: BlockingRuntime> {
    options: VortexWriteOptions,
    runtime: B,
}

impl<B: BlockingRuntime> BlockingWriter<B> {
    /// Write a Vortex file into the given `Write` sink.
    pub fn write<W: Write>(
        self,
        write: W,
        iter: impl ArrayIterator + Send + 'static,
    ) -> VortexResult<Footer> {
        self.runtime.block_on(|handle| async move {
            self.options
                .with_handle(handle)
                .write(AllowStdIo::new(write), iter.into_array_stream())
                .await
        })
    }
}
