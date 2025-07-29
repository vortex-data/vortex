// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::footer::{FileStatistics, FooterFlatBufferWriter, Postscript, PostscriptSegment};
use crate::segments::writer::FileSegmentWriter;
use crate::{EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION, VortexLayoutStrategy};
use async_stream::try_stream;
use futures::{Stream, TryStreamExt, pin_mut, poll};
use std::future;
use std::sync::Arc;
use std::task::Poll;
use vortex_array::ArrayContext;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, SendableArrayStream};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::VortexWrite;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::sequence::SequencePointer;
use vortex_layout::{
    ArrayStreamSequentialExt, LayoutContext, LayoutRef, LayoutStrategy, LocalExecutor, TaskExecutor,
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
            strategy: VortexLayoutStrategy::with_executor(Arc::new(LocalExecutor)),
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

    /// Perform a blocking single-threaded write of the provided stream of `Array`.
    pub fn write_blocking<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<W> {
        todo!()
        // block_on(self.write(write, stream))
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
    pub fn write(
        self,
        // TODO(ngates): pass into write_stream as eof pointer.
        sequence_pointer: SequencePointer,
        stream: SendableArrayStream,
        executor: Arc<dyn TaskExecutor>,
    ) -> impl Stream<Item = VortexResult<Vec<ByteBuffer>>> {
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
        let array_stream = array_stream.sequenced(sequence_pointer);
        let dtype = array_stream.dtype().clone();

        // Create a segment writer for collecting segment specs and buffers.
        // We offset the position by the len of the magic bytes, since they are emitted first.
        let segment_writer = FileSegmentWriter::new(MAGIC_BYTES.len() as u64);

        // Set up a Context to capture the encodings used in the file.
        let ctx = ArrayContext::empty();
        let layout_fut = self
            .strategy
            .write_stream(&ctx, &segment_writer, array_stream);
        pin_mut!(layout_fut);

        // Now we emit the buffers in a stream, which will be driven by the caller
        let layout: LayoutRef;
        try_stream! {
            // First, we emit the magic bytes.
            yield vec![ByteBuffer::copy_from(MAGIC_BYTES)];

            // Now, we sit in a loop polling the layout future and draining the segment writer.
            loop {
                // On each iteration, attempt to drain the segment writer to send buffers.
                let buffers = segment_writer.drain_to_vec();
                if !buffers.is_empty() {
                    yield buffers;
                }

                // Then we poll the layout future once.
                if let Poll::Ready(result) = poll!(&mut layout_fut) {
                    layout = result?;
                    break;
                }
            }

             // Once we finish writing our layout, we need to extract the segment specs.
            let (mut position, segment_specs) = segment_writer.into_parts();

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
