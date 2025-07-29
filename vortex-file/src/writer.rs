// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::footer::{FileStatistics, FooterFlatBufferWriter, Postscript, PostscriptSegment};
use crate::segments::writer::FileSegmentWriter;
use crate::{EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION, VortexLayoutStrategy};
use async_stream::try_stream;
use futures::executor::block_on;
use futures::{Stream, StreamExt, TryStreamExt, pin_mut, poll};
use std::future;
use std::sync::Arc;
use std::task::Poll;
use tokio::runtime::Handle;
use vortex_array::ArrayContext;
use vortex_array::iter::{ArrayIterator, ArrayIteratorExt};
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt, SendableArrayStream};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::VortexWrite;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::SequenceId;
use vortex_layout::{
    LayoutContext, LayoutRef, LayoutStrategy, LocalExecutor, SequentialArrayStreamExt, TaskExecutor,
};

/// Configure a new writer, which can eventually be used to write an [`ArrayStream`] into a sink that implements [`VortexWrite`].
///
/// By default, the [`LayoutStrategy`] will be the [`VortexLayoutStrategy`], which includes re-chunking and will also
/// uncompress all data back to its canonical form before compressing it using the `vortex_btrblocks::BtrBlocksCompressor`.
pub struct VortexWriteOptions {
    strategy: Arc<dyn LayoutStrategy>,
    exclude_dtype: bool,
    max_variable_length_statistics_size: usize,
    file_statistics: Vec<Stat>,
}

impl Default for VortexWriteOptions {
    fn default() -> Self {
        Self {
            strategy: VortexLayoutStrategy::new(),
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
        _object_store: &Arc<dyn object_store::ObjectStore>,
        _path: &object_store::path::Path,
        _stream: S,
    ) -> VortexResult<()> {
        todo!()
        // use vortex_io::ObjectStoreWriter;
        //
        // self.write(
        //     ObjectStoreWriter::new(object_store.clone(), path).await?,
        //     stream,
        // )
        // .await?
        // .shutdown()
        // .await?;
        // Ok(())
    }

    /// Write to a file using the provided `VortexWrite` implementation, spawning CPU tasks on the
    /// given Tokio runtime handle.
    #[cfg(feature = "tokio")]
    pub async fn write_tokio<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        mut write: W,
        stream: S,
        handle: Handle,
    ) -> VortexResult<()> {
        // Configure a Tokio executor to spawn concurrent CPU-bound tasks.
        let executor: Arc<dyn TaskExecutor> = Arc::new(handle);
        let buffers = self.write_stream(ArrayStreamExt::boxed(stream), &executor);
        pin_mut!(buffers);

        while let Some(bufs) = buffers.next().await {
            for buf in bufs? {
                write.write_all(buf).await?;
            }
        }
        write.flush().await?;
        Ok(())
    }

    /// Perform a blocking single-threaded write of the provided [`ArrayIterator`].
    pub fn write<W: VortexWrite, I: ArrayIterator + Send + 'static>(
        self,
        mut write: W,
        iter: I,
    ) -> VortexResult<W> {
        let executor = LocalExecutor::new();
        let buffers = self.write_stream(ArrayStreamExt::boxed(iter.into_array_stream()), &executor);
        pin_mut!(buffers);

        while let Some(bufs) = block_on(buffers.next()) {
            for buf in bufs? {
                block_on(write.write_all(buf))?;
            }
        }
        block_on(write.flush())?;

        Ok(write)
    }

    /// Writes the given [`ArrayStream`] using the configured layout strategy.
    ///
    /// The returned stream of buffers should be written contiguously to a file or other byte
    /// sink.
    ///
    /// While this function is async, it drives CPU-bound operations either in a single-threaded
    /// mode, or with spawned tasks using the provided executor. For async I/O runtimes, this means
    /// the returned stream should be polled using `spawn_blocking` or similar. For async CPU
    /// runtimes, the stream can be polled directly. For blocking callers, the stream can be
    /// iterated using `futures::executor::block_on` or similar.
    pub fn write_stream(
        self,
        stream: SendableArrayStream,
        executor: &Arc<dyn TaskExecutor>,
    ) -> impl Stream<Item = VortexResult<Vec<ByteBuffer>>> {
        // Create an initial sequence pointer along with an end-of-file pointer.
        let (ptr, eof) = SequenceId::root().split();

        // Wrap the input stream to remove empty chunks and collect file-level stats.
        let array_stream = ArrayStreamAdapter::new(
            stream.dtype().clone(),
            stream.try_filter(|chunk| future::ready(!chunk.is_empty())),
        );
        let (file_stats, array_stream) = accumulate_stats(
            array_stream,
            self.file_statistics.clone().into(),
            self.max_variable_length_statistics_size,
        );
        let array_stream = array_stream.sequenced(ptr);
        let dtype = array_stream.dtype().clone();

        // Now we emit the buffers in a stream, which will be driven by the caller
        try_stream! {
            // Create a segment writer for collecting segment specs and buffers.
            // We offset the position by the len of the magic bytes, since they are emitted first.
            let segment_writer = Arc::new(FileSegmentWriter::new(MAGIC_BYTES.len() as u64));

            // Set up a Context to capture the encodings used in the file.
            let ctx = ArrayContext::empty();
            let segment_writer2 = segment_writer.clone();
            let layout_fut = self
                .strategy
                .write_stream(&ctx, segment_writer2.as_ref(), executor, array_stream, eof);
            pin_mut!(layout_fut);

            // First, we emit the magic bytes.
            yield vec![ByteBuffer::copy_from(MAGIC_BYTES)];

            // Now, we sit in a loop polling the layout future and draining the segment writer.
            let layout: LayoutRef;
            loop {
                // On each iteration, attempt to drain the segment writer to send buffers.
                let buffers = segment_writer.drain_to_vec();
                if !buffers.is_empty() {
                    yield buffers;
                }

                // Then we poll the layout future once.
                if let Poll::Ready(result) = poll!(&mut layout_fut) {
                    layout = result?;

                    let buffers = segment_writer.drain_to_vec();
                    if !buffers.is_empty() {
                        yield buffers;
                    }

                    break;
                }
            }

             // Once we finish writing our layout, we need to extract the segment specs.
            let mut position = segment_writer.byte_offset();
            let segment_specs = segment_writer.segment_specs();

            // Collect together buffers to write.
            let mut buffers = Vec::with_capacity(4);

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
                    ctx: ctx.clone(),
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
                Err(vortex_err!(
                    "Postscript is too large ({} bytes); max postscript size is {}",
                    postscript_buffer.len(),
                    MAX_FOOTER_SIZE
                ))?;
            }

            let postscript_len = u16::try_from(postscript_buffer.len())
                .vortex_expect("Postscript already verified to fit into u16");
            buffers.push(postscript_buffer.into_inner());

            // And finally, the EOF 8-byte footer.
            let mut eof = [0u8; EOF_SIZE];
            eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
            eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
            eof[4..8].copy_from_slice(&MAGIC_BYTES);
            buffers.push(ByteBuffer::copy_from(eof));

            yield buffers
        }
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
