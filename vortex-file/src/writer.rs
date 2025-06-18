use std::future;
use std::sync::Arc;

use arcref::ArcRef;
use futures::TryStreamExt;
use futures::future::try_join;
use vortex_array::ArrayContext;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter, ArrayStreamExt};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::VortexWrite;
use vortex_layout::layouts::file_stats::accumulate_stats;
use vortex_layout::scan::LocalExecutor;
use vortex_layout::segments::SequenceWriter;
use vortex_layout::{LayoutContext, LayoutStrategy};

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

    /// Perform an async write of the provided stream of `Array`.
    pub async fn write<W: VortexWrite, S: ArrayStream + Unpin + Send + 'static>(
        self,
        write: W,
        stream: S,
    ) -> VortexResult<W> {
        // Set up a Context to capture the encodings used in the file.
        let ctx = ArrayContext::empty();

        let dtype = stream.dtype().clone();
        let (segment_writer, flusher) = SerialSegmentWriter::create();
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

        // First we write the magic number
        let mut write = futures::io::Cursor::new(write);
        write.write_all(MAGIC_BYTES).await?;

        let io_fut = flusher.flush(write);
        let compute_fut = self.strategy.write_stream(&ctx, sequence_writer, stream);
        let (layout, (mut write, segment_specs)) = try_join(compute_fut, io_fut).await?;

        // We write our footer components in order of least likely to be needed to most likely.
        // DType is the least likely to be needed, as many readers may provide this from an
        // external source.
        let dtype_segment = if self.exclude_dtype {
            None
        } else {
            Some(self.write_flatbuffer(&mut write, &dtype).await?)
        };

        let layout_ctx = LayoutContext::empty();
        let layout_segment = self
            .write_flatbuffer(&mut write, &layout.flatbuffer_writer(&layout_ctx))
            .await?;

        let statistics_segment = if self.file_statistics.is_empty() {
            None
        } else {
            let file_statistics = FileStatistics(file_stats.stats_sets().into());
            Some(self.write_flatbuffer(&mut write, &file_statistics).await?)
        };

        let footer_segment = self
            .write_flatbuffer(
                &mut write,
                &FooterFlatBufferWriter {
                    ctx,
                    layout_ctx,
                    segment_specs: segment_specs.into(),
                },
            )
            .await?;

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
        write.write_all(postscript_buffer).await?;

        // And finally, the EOF 8-byte footer.
        let mut eof = [0u8; EOF_SIZE];
        eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
        eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
        eof[4..8].copy_from_slice(&MAGIC_BYTES);
        write.write_all(eof).await?;

        write.flush().await?;

        Ok(write.into_inner())
    }

    async fn write_flatbuffer<W: VortexWrite, F: FlatBufferRoot + WriteFlatBuffer>(
        &self,
        write: &mut futures::io::Cursor<W>,
        flatbuffer: &F,
    ) -> VortexResult<PostscriptSegment> {
        let layout_offset = write.position();
        write.write_all(flatbuffer.write_flatbuffer_bytes()).await?;
        Ok(PostscriptSegment {
            offset: layout_offset,
            length: u32::try_from(write.position() - layout_offset)
                .map_err(|_| vortex_err!("segment length exceeds maximum u32"))?,
            alignment: FlatBuffer::alignment(),
        })
    }
}
