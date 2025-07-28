// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arcref::ArcRef;
use futures::channel::{mpsc, oneshot};
use futures::executor::block_on;
use futures::{SinkExt, Stream, StreamExt, TryStreamExt, stream};
use std::future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc};
use vortex_array::ArrayContext;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::VortexWrite;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::segments::SequenceWriter;
use vortex_layout::{LayoutContext, LayoutStrategy, LocalExecutor};

use crate::footer::{FileStatistics, FooterFlatBufferWriter, Postscript, PostscriptSegment};
use crate::segments::writer::SerialSegmentWriter;
use crate::{EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION, VortexLayoutStrategy};

/// Configure a new writer, which can eventually be used to write an [`ArrayStream`] into a sink that implements [`VortexWrite`].
///
/// By default, the [`LayoutStrategy`] will be the [`VortexLayoutStrategy`], which includes re-chunking and will also
/// uncompress all data back to its canonical form before compressing it using the `vortex_btrblocks::BtrBlocksCompressor`.
pub struct VortexWriteOptions {
    strategy: ArcRef<dyn LayoutStrategy>,
    exclude_dtype: bool,
    max_variable_length_statistics_size: usize,
    file_statistics: Vec<Stat>,
}

impl Default for VortexWriteOptions {
    fn default() -> Self {
        Self {
            strategy: VortexLayoutStrategy::with_executor(Arc::new(LocalExecutor)),
            exclude_dtype: false,
            file_statistics: PRUNING_STATS.to_vec(),
            max_variable_length_statistics_size: 64,
        }
    }
}

impl VortexWriteOptions {
    /// Replace the default layout strategy with the provided one.
    pub fn with_strategy(mut self, strategy: ArcRef<dyn LayoutStrategy>) -> Self {
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
        use vortex_io::ObjectStoreWriter;

        self.write(
            ObjectStoreWriter::new(object_store.clone(), path).await?,
            stream,
        )
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
        block_on(self.write(write, stream))
    }

    /// Perform an async write of the provided stream of `Array`.
    ///
    /// Returns a tuple containing:
    /// - A stream of `VortexResult<ByteBuffer>` that should be written contiguously to disk.
    /// - A future that, when awaited, will drive the computation of the writer.
    ///
    /// A blocking implementation should alternate between flushing buffers and driving the
    /// computation future.
    ///
    /// An async implementation should spawn background tasks to perform I/O and drive computation
    /// future.
    pub fn write<S: ArrayStream + Unpin + Send + 'static>(
        self,
        stream: S,
    ) -> VortexResult<(
        impl Future<Output=VortexResult<()>>,
        impl Stream<Item=ByteBuffer>,
    )> {
        // Set up a Context to capture the encodings used in the file.
        let ctx = ArrayContext::empty();
        let dtype = stream.dtype().clone();

        // Create a channel for sending buffers to be written.
        // For now, we arbitrarily bound this channel. We should create a channel that's bounded
        // by the total size of the pending buffers.
        let (mut send, recv) = mpsc::channel(64);

        // TODO(ngates): send a mut ref to the send queue. Make it spsc
        let segment_writer = SerialSegmentWriter::create(MAGIC_BYTES.len() as u64, send.clone());
        let sequence_writer = SequenceWriter::new(Box::new(segment_writer));

        let stream = stream.try_filter(|chunk| future::ready(!chunk.is_empty()));
        let stream = sequence_writer.new_sequential(ArrayStreamExt::boxed(
            ArrayStreamAdapter::new(dtype.clone(), stream),
        ));
        let (file_stats, stream) = accumulate_stats(
            stream,
            self.file_statistics.clone().into(),
            self.max_variable_length_statistics_size,
        );

        // Now we construct a future that encapsulates the write operation.
        let (layout_send, layout_recv) = oneshot::channel();
        let writer = async move {
            let layout = self
                .strategy
                .write_stream(&ctx, &sequence_writer, stream)
                .await?;

            // Once we finish writing our layout, we need to extract the segment specs.

            sequence_writer.into
            send.send((layout));
        };

        // Finally, we combine the magic bytes, the data buffers, and then the footer buffers into a single stream.
        let position = Arc::new(AtomicU64::new(0));
        let position2 = position.clone();
        let buffer_stream = stream::iter([ByteBuffer::copy_from(MAGIC_BYTES)])
            .chain(recv)
            .inspect(move |buffer| {
                position.fetch_add(buffer.len() as u64, Ordering::Relaxed);
            })
            .chain(
                stream::once(async move {
                    let layout = writer.await?;
                    let mut position = position2.load(Ordering::SeqCst);
                    let mut buffers = vec![];

                    let dtype_segment = if self.exclude_dtype {
                        None
                    } else {
                        Some(write_flatbuffer(&mut position, &mut buffers, &dtype)?)
                    };

                    let layout_ctx = LayoutContext::empty();
                    let layout_segment = write_flatbuffer(
                        &mut position,
                        &mut buffers,
                        &layout.flatbuffer_writer(&layout_ctx),
                    )?;

                    let statistics_segment = if self.file_statistics.is_empty() {
                        None
                    } else {
                        let file_statistics = FileStatistics(file_stats.stats_sets().into());
                        Some(write_flatbuffer(
                            &mut position,
                            &mut buffers,
                            &file_statistics,
                        )?)
                    };

                    let footer_segment = write_flatbuffer(
                        &mut position,
                        &mut buffers,
                        &FooterFlatBufferWriter {
                            ctx,
                            layout_ctx,
                            segment_specs: segment_specs.into(),
                        },
                    )?;

                    // Assemble the postscript, and write it manually to avoid any framing.
                    let postscript = Postscript {
                        dtype: dtype_segment,
                        layout: layout_segment,
                        statistics: statistics_segment,
                        footer: footer_segment,
                    };
                    let postscript_buffer = postscript.write_flatbuffer_bytes();
                    if postscript_buffer.len() > MAX_FOOTER_SIZE as usize {
                        vortex_bail!(
                            "Postscript is too large ({} bytes); max postscript size is {}",
                            postscript_buffer.len(),
                            MAX_FOOTER_SIZE
                        );
                    }
                    let postscript_len = u16::try_from(postscript_buffer.len())
                        .vortex_expect("Postscript already verified to fit into u16");
                    buffers.push(postscript_buffer);

                    // And finally, the EOF 8-byte footer.
                    let mut eof = [0u8; EOF_SIZE];
                    eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
                    eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
                    eof[4..8].copy_from_slice(&MAGIC_BYTES);
                    buffers.push(ByteBuffer::copy_from(eof));

                    Ok(())
                })
                    .flatten(),
            );
    }
}

fn write_flatbuffer<F: FlatBufferRoot + WriteFlatBuffer>(
    offset: &mut u64,
    buffers: &mut Vec<ByteBuffer>,
    flatbuffer: &F,
) -> VortexResult<PostscriptSegment> {
    let buffer = flatbuffer.write_flatbuffer_bytes();
    let length = u32::try_from(buffer.len())
        .map_err(|_| vortex_err!("flatbuffer length exceeds maximum u32"))?;

    let segment = PostscriptSegment {
        offset: *offset,
        length,
        alignment: FlatBuffer::alignment(),
    };

    buffers.push(buffer.into_inner());
    *offset += u64::from(length);

    Ok(segment)
}
