// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::Fuse;
use futures::future::LocalBoxFuture;
use futures::future::ready;
use futures::pin_mut;
use futures::select;
use itertools::Itertools;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_array::iter::ArrayIterator;
use vortex_array::iter::ArrayIteratorExt;
use vortex_array::session::ArraySessionExt;
use vortex_array::stats::PRUNING_STATS;
use vortex_array::stream::ArrayStream;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::IoBuf;
use vortex_io::VortexWrite;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialStreamAdapter;
use vortex_layout::sequence::SequentialStreamExt;
use vortex_session::SessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::ALLOWED_ENCODINGS;
use crate::Footer;
use crate::MAGIC_BYTES;
use crate::WriteStrategyBuilder;
use crate::counting::CountingVortexWrite;
use crate::footer::FileStatistics;
use crate::segments::writer::BufferedSegmentSink;

/// Configure a new writer, which can eventually be used to write an [`ArrayStream`] into a sink
/// that implements [`VortexWrite`].
///
/// Unless overridden, the default [write strategy][crate::WriteStrategyBuilder] will be used with no
/// additional configuration.
///
/// Construct with [`WriteOptionsSessionExt::write_options`] for normal use so the writer inherits
/// the session's runtime, array registry, and memory configuration.
pub struct VortexWriteOptions {
    session: VortexSession,
    strategy: Arc<dyn LayoutStrategy>,
    exclude_dtype: bool,
    max_variable_length_statistics_size: usize,
    file_statistics: Vec<Stat>,
}

/// Extension trait for constructing [`VortexWriteOptions`] from a session.
pub trait WriteOptionsSessionExt: SessionExt {
    /// Create [`VortexWriteOptions`] for writing to a Vortex file.
    fn write_options(&self) -> VortexWriteOptions {
        let session = self.session();
        VortexWriteOptions {
            strategy: WriteStrategyBuilder::default().build(),
            session,
            exclude_dtype: false,
            file_statistics: PRUNING_STATS.to_vec(),
            max_variable_length_statistics_size: 64,
        }
    }
}
impl<S: SessionExt> WriteOptionsSessionExt for S {}

impl VortexWriteOptions {
    /// Create a new [`VortexWriteOptions`] with the given session.
    pub fn new(session: VortexSession) -> Self {
        VortexWriteOptions {
            strategy: WriteStrategyBuilder::default().build(),
            session,
            exclude_dtype: false,
            file_statistics: PRUNING_STATS.to_vec(),
            max_variable_length_statistics_size: 64,
        }
    }

    /// Replace the default layout strategy with the provided one.
    ///
    /// The strategy controls repartitioning, statistics layout, compression, and leaf segment
    /// emission. Use [`WriteStrategyBuilder`] when only a small part of the default strategy needs
    /// customization.
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

    /// Configure which statistics to compute at the file level.
    ///
    /// Pass an empty vector to omit file-level statistics.
    pub fn with_file_statistics(mut self, file_statistics: Vec<Stat>) -> Self {
        self.file_statistics = file_statistics;
        self
    }
}

impl VortexWriteOptions {
    /// Drop into the blocking writer API using the given runtime.
    ///
    /// The returned adapter drives async writer internals on `runtime` while accepting ordinary
    /// [`std::io::Write`] sinks and [`ArrayIterator`] inputs.
    pub fn blocking<B: BlockingRuntime>(self, runtime: &B) -> BlockingWrite<'_, B> {
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
        // NOTE(os): Setup an array context that already has all known encodings pre-populated.
        // This is preferred for now over having an empty context here, because only the
        // serialised array order is deterministic. The serialisation of arrays are done
        // parallel and with an empty context they can register their encodings to the context
        // in different order, changing the written bytes from run to run.
        let ctx = ArrayContext::new(ALLOWED_ENCODINGS.iter().cloned().sorted().collect())
            // Configure a registry just to ensure only known encodings are interned.
            .with_registry(self.session.arrays().registry().clone());
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
            &self.session,
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
        let session = self.session.clone();
        let layout_fut = self.session.handle().spawn_nested(move |h| async move {
            let session = session.with_handle(h);
            let layout = self
                .strategy
                .write_stream(
                    ctx2,
                    Arc::<BufferedSegmentSink>::clone(&segments),
                    stream,
                    eof,
                    &session,
                )
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
        let mut footer = Footer::new(
            Arc::clone(&layout),
            segment_specs,
            if self.file_statistics.is_empty() {
                None
            } else {
                Some(FileStatistics::new_with_dtype(
                    file_stats.stats_sets().into(),
                    &dtype,
                ))
            },
            ReadContext::new(ctx.to_ids()),
        );

        // Emit the footer buffers and EOF.
        let footer_buffers = footer
            .clone()
            .into_serializer()
            .with_offset(position)
            .with_exclude_dtype(self.exclude_dtype)
            .serialize()?;

        // Update the approx footer size in the footer object, so it can be used for caching and
        // memory management in the future.
        footer = footer.with_approx_byte_size(footer_buffers.iter().map(|b| b.len()).sum());

        for buffer in footer_buffers {
            position += buffer.len() as u64;
            write.write_all(buffer).await?;
        }

        write.flush().await?;

        Ok(WriteSummary {
            footer,
            size: position,
        })
    }

    /// Create a push-based [`Writer`] that can be used to incrementally write arrays to the file.
    ///
    /// Each pushed chunk must have dtype `dtype`. Call [`Writer::finish`] to close the input stream,
    /// flush remaining buffers, and receive the [`WriteSummary`].
    pub fn writer<'w, W: VortexWrite + Unpin + 'w>(self, write: W, dtype: DType) -> Writer<'w> {
        // Create a channel for sending arrays to the layout task.
        let (arrays_send, arrays_recv) = kanal::bounded_async(1);

        let arrays =
            ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays_recv.into_stream()));

        let write = CountingVortexWrite::new(write);
        let bytes_written = write.counter();
        let strategy = Arc::clone(&self.strategy);
        let future = self.write(write, arrays).boxed_local().fuse();

        Writer {
            arrays: Some(arrays_send),
            future,
            bytes_written,
            strategy,
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
    // The layout strategy that is being used for the write.
    strategy: Arc<dyn LayoutStrategy>,
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
        self.bytes_written.load(Ordering::Relaxed)
    }

    /// Returns the number of bytes currently buffered by the layout writers.
    pub fn buffered_bytes(&self) -> u64 {
        self.strategy.buffered_bytes()
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

/// Blocking adapter for [`VortexWriteOptions`].
pub struct BlockingWrite<'rt, B: BlockingRuntime> {
    options: VortexWriteOptions,
    runtime: &'rt B,
}

impl<'rt, B: BlockingRuntime> BlockingWrite<'rt, B> {
    /// Write a Vortex file into the given `Write` sink.
    ///
    /// The iterator is converted to an [`ArrayStream`] and driven to completion on
    /// the configured blocking runtime.
    pub fn write<W: Write + Unpin>(
        self,
        write: W,
        iter: impl ArrayIterator + Send + 'static,
    ) -> VortexResult<WriteSummary> {
        self.runtime.block_on(async move {
            self.options
                .write(BlockingWriteAdapter(write), iter.into_array_stream())
                .await
        })
    }

    /// Create a blocking push-based writer for chunks with dtype `dtype`.
    pub fn writer<'w, W: Write + Unpin + 'w>(
        self,
        write: W,
        dtype: DType,
    ) -> BlockingWriter<'rt, 'w, B> {
        BlockingWriter {
            writer: self.options.writer(BlockingWriteAdapter(write), dtype),
            runtime: self.runtime,
        }
    }
}

/// A blocking adapter around a [`Writer`], allowing incremental writing of arrays to a Vortex file.
pub struct BlockingWriter<'rt, 'w, B: BlockingRuntime> {
    runtime: &'rt B,
    writer: Writer<'w>,
}

impl<B: BlockingRuntime> BlockingWriter<'_, '_, B> {
    /// Push one array chunk into the file.
    pub fn push(&mut self, chunk: ArrayRef) -> VortexResult<()> {
        self.runtime.block_on(self.writer.push(chunk))
    }

    /// Returns the number of bytes written to the sink so far.
    pub fn bytes_written(&self) -> u64 {
        self.writer.bytes_written()
    }

    /// Returns the number of bytes currently buffered by layout strategies.
    pub fn buffered_bytes(&self) -> u64 {
        self.writer.buffered_bytes()
    }

    /// Finish writing and return the written file summary.
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

/// Summary returned after a Vortex file is written.
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

    /// The total number of rows in the written Vortex file.
    pub fn row_count(&self) -> u64 {
        self.footer.row_count()
    }
}
