// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use futures::future::{Fuse, LocalBoxFuture, ready};
use futures::{FutureExt, StreamExt, TryStreamExt, pin_mut, select};
use vortex_array::iter::{ArrayIterator, ArrayIteratorExt};
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt, SendableArrayStream};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{
    VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic,
};
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::{BlockingRuntime, Handle};
use vortex_io::{IoBuf, VortexWrite};
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::{SequenceId, SequentialStreamAdapter, SequentialStreamExt};

use crate::counting::CountingVortexWrite;
use crate::footer::FileStatistics;
use crate::segments::writer::BufferedSegmentSink;
use crate::{Footer, MAGIC_BYTES, WriteStrategyBuilder};

/// Configure a new writer, which can eventually be used to write an [`ArrayStream`] into a sink that implements [`VortexWrite`].
///
/// Unless overridden, the default [write strategy][crate::WriteStrategyBuilder] will be used with no
/// additional configuration.
pub struct VortexWriteOptions {
    strategy: Arc<dyn LayoutStrategy>,
    exclude_dtype: bool,
    max_variable_length_statistics_size: usize,
    file_statistics: Vec<Stat>,
    handle: Option<Handle>,
}

impl Default for VortexWriteOptions {
    fn default() -> Self {
        Self {
            strategy: WriteStrategyBuilder::new().build(),
            exclude_dtype: false,
            file_statistics: PRUNING_STATS.to_vec(),
            max_variable_length_statistics_size: 64,
            handle: Handle::find(),
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
    pub fn blocking<B: BlockingRuntime + Default>(self) -> BlockingWrite<B> {
        self.with_blocking(B::default())
    }

    /// Drop into the blocking writer API using the given runtime.
    pub fn with_blocking<B: BlockingRuntime>(self, runtime: B) -> BlockingWrite<B> {
        if self.handle.is_some() {
            vortex_panic!("Must not provide or infer a Handle when using the blocking writer API")
        }
        BlockingWrite {
            options: self,
            runtime,
        }
    }

    /// Write an [`ArrayStream`] as a Vortex file.
    ///
    /// Note that buffers are flushed as soon as they are available with no buffering, the caller
    /// is responsible for deciding how to configure buffering on the underlying `Write` sink.
    pub async fn write<W: VortexWrite + Unpin, S: ArrayStream + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<WriteSummary> {
        self.write_internal(write, ArrayStreamExt::boxed(stream))
            .await
    }

    async fn write_internal<W: VortexWrite + Unpin>(
        self,
        mut write: W,
        stream: SendableArrayStream,
    ) -> VortexResult<WriteSummary> {
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

        // First, write the magic bytes.
        write.write_all(ByteBuffer::copy_from(MAGIC_BYTES)).await?;
        let mut position = MAGIC_BYTES.len() as u64;

        // Create a channel to send buffers from the segment sink to the output stream.
        let (send, recv) = kanal::bounded_async(1);

        let segments = Arc::new(BufferedSegmentSink::new(send, position));

        // We spawn the layout future so it is driven in the background while we write the
        // buffer stream, so we don't need to poll it until all buffers have been drained.
        let ctx2 = ctx.clone();
        let layout_fut = handle.spawn_nested(|h| async move {
            let layout = self
                .strategy
                .write_stream(ctx2, segments.clone(), stream, eof, h)
                .await?;
            Ok::<_, VortexError>((layout, segments.segment_specs()))
        });

        // Flush buffers as they arrive
        let recv_stream = recv.into_stream();
        pin_mut!(recv_stream);
        while let Some(buffer) = recv_stream.next().await {
            if buffer.is_empty() {
                continue;
            }
            position += buffer.len() as u64;
            write.write_all(buffer).await?;
        }

        let (layout, segment_specs) = layout_fut.await?;

        // Assemble the Footer object now that we have all the segments.
        let footer = Footer::new(
            layout.clone(),
            segment_specs,
            if self.file_statistics.is_empty() {
                None
            } else {
                Some(FileStatistics(file_stats.stats_sets().into()))
            },
            ctx,
        );

        // Emit the footer buffers and EOF.
        let footer_buffers = footer
            .clone()
            .into_serializer()
            .with_offset(position)
            .with_exclude_dtype(self.exclude_dtype)
            .serialize()?;
        for buffer in footer_buffers {
            position += buffer.len() as u64;
            write.write_all(buffer).await?;
        }

        Ok(WriteSummary {
            footer,
            size: position,
        })
    }

    /// Create a push-based [`Writer`] that can be used to incrementally write arrays to the file.
    pub fn writer<'w, W: VortexWrite + Unpin + 'w>(self, write: W, dtype: DType) -> Writer<'w> {
        // Create a channel for sending arrays to the layout task.
        let (arrays_send, arrays_recv) = kanal::bounded_async(1);

        let arrays =
            ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays_recv.into_stream()));

        let write = CountingVortexWrite::new(write);
        let bytes_written = write.counter();
        let future = self.write(write, arrays).boxed_local().fuse();

        Writer {
            arrays: Some(arrays_send),
            future,
            bytes_written,
        }
    }
}

/// An async API for writing Vortex files.
pub struct Writer<'w> {
    // The input channel for sending arrays to the writer.
    arrays: Option<kanal::AsyncSender<VortexResult<ArrayRef>>>,
    // The writer task that ultimately produces the footer.
    future: Fuse<LocalBoxFuture<'w, VortexResult<WriteSummary>>>,
    // The bytes written so far.
    bytes_written: Arc<AtomicU64>,
}

impl Writer<'_> {
    /// Push a new chunk into the writer.
    pub async fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        let arrays = self.arrays.clone().vortex_expect("missing arrays sender");
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
                match result {
                    Ok(_) => vortex_bail!("Internal error: writer future completed early"),
                    Err(e) => return Err(e),
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
        let arrays = self.arrays.clone().vortex_expect("missing arrays sender");
        let stream_fut = async move {
            while let Some(chunk) = stream.next().await {
                arrays.send(chunk).await?;
            }
            Ok::<_, kanal::SendError>(())
        }
        .fuse();
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
                match result {
                    Ok(_) => vortex_bail!("Internal error: writer future completed early"),
                    Err(e) => return Err(e),
                }
            }
        }

        Ok(())
    }

    /// Returns the number of bytes written to the file so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Finish writing the Vortex file, flushing any remaining buffers and returning the
    /// new file's footer.
    pub async fn finish(mut self) -> VortexResult<WriteSummary> {
        // Drop the input channel to signal EOF.
        drop(self.arrays.take());

        // Await the future task.
        self.future.await
    }

    /// Assuming the writer task has failed, await it to get the error.
    async fn handle_failed_task(&mut self) -> VortexError {
        match (&mut self.future).await {
            Ok(_) => vortex_err!(
                "Internal error: writer task completed successfully but write future finished early"
            ),
            Err(e) => e,
        }
    }
}

/// A blocking API for writing Vortex files.
pub struct BlockingWrite<B: BlockingRuntime> {
    options: VortexWriteOptions,
    runtime: B,
}

impl<B: BlockingRuntime> BlockingWrite<B> {
    /// Write a Vortex file into the given `Write` sink.
    pub fn write<W: Write + Unpin>(
        self,
        write: W,
        iter: impl ArrayIterator + Send + 'static,
    ) -> VortexResult<WriteSummary> {
        let handle = self.runtime.handle();
        self.runtime.block_on(async move {
            self.options
                .with_handle(handle)
                .write(BlockingWriteAdapter(write), iter.into_array_stream())
                .await
        })
    }

    pub fn writer<'w, W: Write + Unpin + 'w>(
        self,
        write: W,
        dtype: DType,
    ) -> BlockingWriter<'w, B> {
        BlockingWriter {
            writer: self
                .options
                .with_handle(self.runtime.handle())
                .writer(BlockingWriteAdapter(write), dtype),
            runtime: self.runtime,
        }
    }
}

/// A blocking adapter around a [`Writer`], allowing incremental writing of arrays to a Vortex file.
pub struct BlockingWriter<'w, B: BlockingRuntime> {
    runtime: B,
    writer: Writer<'w>,
}

impl<B: BlockingRuntime> BlockingWriter<'_, B> {
    pub fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        self.runtime.block_on(self.writer.push(chunk))
    }

    pub fn bytes_written(&self) -> u64 {
        self.writer.bytes_written()
    }

    pub fn finish(self) -> VortexResult<WriteSummary> {
        self.runtime.block_on(self.writer.finish())
    }
}

// TODO(ngates): this blocking API may change, for now we just run blocking I/O inline.
struct BlockingWriteAdapter<W>(W);

impl<W: Write + Unpin> VortexWrite for BlockingWriteAdapter<W> {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.0.write_all(buffer.as_slice())?;
        Ok(buffer)
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(self.0.flush())
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Ok(()))
    }
}

pub struct WriteSummary {
    footer: Footer,
    size: u64,
    // TODO(ngates): add a checksum
}

impl WriteSummary {
    /// The footer of the written Vortex file.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    /// The total size of the written Vortex file in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The footer of the written Vortex file.
    pub fn row_count(&self) -> u64 {
        self.footer.row_count()
    }
}
