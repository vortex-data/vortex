// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;
use std::sync::Arc;

use async_stream::try_stream;
use futures::channel::oneshot;
use futures::future::ready;
use futures::io::AllowStdIo;
use futures::stream::BoxStream;
use futures::{StreamExt, TryFutureExt, TryStreamExt, pin_mut};
use vortex_array::ArrayContext;
use vortex_array::iter::{ArrayIterator, ArrayIteratorExt};
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamExt, SendableArrayStream};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexResult, vortex_err};
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::{BlockingRuntime, Handle};
use vortex_io::{AsyncWriteAdapter, VortexWrite};
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::{SequenceId, SequentialStreamAdapter, SequentialStreamExt};

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

    /// Perform an async write of the provided stream of `Array`.
    pub async fn write<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: &mut W,
        stream: S,
        handle: Handle,
    ) -> VortexResult<Footer> {
        self.write_stream(
            ArrayStreamExt::boxed(stream),
            handle,
            |mut bytes| async move {
                while let Some(buffer) = bytes.next().await {
                    write.write_all(buffer?).await?;
                }
                Ok(write.flush().await?)
            },
        )
        .await
    }

    /// Write an [`ArrayStream`] as a Vortex file.
    ///
    /// The sink is passed a stream of byte buffers that should be written contiguously. Once
    /// complete, the returned future will resolve to a [`Footer`] object that describes the layout
    /// of the written data.
    pub async fn write_stream<F, Fut>(
        self,
        stream: SendableArrayStream,
        handle: Handle,
        sink: F,
    ) -> VortexResult<Footer>
    where
        F: FnOnce(BoxStream<'static, VortexResult<ByteBuffer>>) -> Fut,
        Fut: Future<Output = VortexResult<()>>,
    {
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

            // Assemble the Footer object now that we have all the segments.
            let footer = Footer::new(
                layout.clone(),
                segment_specs.into(),
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
                yield buffer;
            }

            // Emit the footer to the caller.
            let _ = footer_send.send(footer);
        };

        sink(byte_stream.boxed()).await?;

        footer_recv
            .map_err(|_canceled| vortex_err!("Cannot return Footer from failed write"))
            .await
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
            self.options
                .write(
                    &mut AsyncWriteAdapter(AllowStdIo::new(write)),
                    ArrayStreamExt::boxed(iter.into_array_stream()),
                    handle,
                )
                .await
        })
    }
}
