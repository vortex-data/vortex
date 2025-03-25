use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_array::stats::{PRUNING_STATS, Stat};
use vortex_array::stream::ArrayStream;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::VortexWrite;
use vortex_layout::stats::FileStatsLayoutWriter;
use vortex_layout::{LayoutStrategy, LayoutWriter};

use crate::footer::{Footer, Postscript, SegmentSpec};
use crate::segments::writer::BufferedSegmentWriter;
use crate::strategy::VortexLayoutStrategy;
use crate::{EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION};

pub struct VortexWriteOptions {
    strategy: Box<dyn LayoutStrategy>,
    exclude_dtype: bool,
    file_statistics: Option<Vec<Stat>>,
}

impl Default for VortexWriteOptions {
    fn default() -> Self {
        Self {
            strategy: Box::new(VortexLayoutStrategy),
            exclude_dtype: false,
            file_statistics: Some(PRUNING_STATS.to_vec()),
        }
    }
}

impl VortexWriteOptions {
    /// Replace the default layout strategy with the provided one.
    pub fn with_strategy<S: LayoutStrategy>(mut self, strategy: S) -> Self {
        self.strategy = Box::new(strategy);
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
        self.file_statistics = Some(file_statistics);
        self
    }
}

impl VortexWriteOptions {
    /// Perform an async write of the provided stream of `Array`.
    pub async fn write<W: VortexWrite, S: ArrayStream + Unpin>(
        self,
        write: W,
        mut stream: S,
    ) -> VortexResult<W> {
        // Set up a Context to capture the encodings used in the file.
        let ctx = ArrayContext::empty();

        // Set up the root layout
        let mut layout_writer = FileStatsLayoutWriter::new(
            self.strategy.new_writer(&ctx, stream.dtype())?,
            stream.dtype(),
            self.file_statistics.clone().unwrap_or_default().into(),
        )?;

        // First we write the magic number
        let mut write = futures::io::Cursor::new(write);
        write.write_all(MAGIC_BYTES).await?;

        // Our buffered message writer accumulates messages for each batch so we can flush them
        // into the file.
        let mut segment_writer = BufferedSegmentWriter::default();
        let mut segment_map = vec![];

        // Then write the stream via the root layout
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            layout_writer.push_chunk(&mut segment_writer, chunk)?;
            // NOTE(ngates): we could spawn this task and continue to compress the next chunk.
            segment_writer
                .flush_async(&mut write, &mut segment_map)
                .await?;
        }

        // Flush the final layout messages into the file
        layout_writer.flush(&mut segment_writer)?;
        segment_writer
            .flush_async(&mut write, &mut segment_map)
            .await?;

        // Finish the layouts and flush the finishing messages into the file
        let layout = layout_writer.finish(&mut segment_writer)?;
        segment_writer
            .flush_async(&mut write, &mut segment_map)
            .await?;

        // Write the DType, followed by the Footer. We choose this order because in many cases
        // the reader can provide an external DType, therefore we may not even need to read it.
        // We will always need the footer, therefore we write it closest to the end of the file
        // and therefore most likely to be picked up in the initial read.
        let dtype_segment = if self.exclude_dtype {
            None
        } else {
            Some(self.write_flatbuffer(&mut write, stream.dtype()).await?)
        };

        let footer_segment = self
            .write_flatbuffer(
                &mut write,
                &Footer::flatbuffer_writer(
                    ctx,
                    layout,
                    segment_map.into(),
                    self.file_statistics
                        .is_some()
                        .then(|| layout_writer.into_stats_sets().into()),
                ),
            )
            .await?;

        // Assemble the postscript, and write it manually to avoid any framing.
        let postscript = Postscript {
            dtype: dtype_segment,
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

        Ok(write.into_inner())
    }

    async fn write_flatbuffer<W: VortexWrite, F: FlatBufferRoot + WriteFlatBuffer>(
        &self,
        write: &mut futures::io::Cursor<W>,
        flatbuffer: &F,
    ) -> VortexResult<SegmentSpec> {
        let layout_offset = write.position();
        write.write_all(flatbuffer.write_flatbuffer_bytes()).await?;
        Ok(SegmentSpec {
            offset: layout_offset,
            length: u32::try_from(write.position() - layout_offset)
                .map_err(|_| vortex_err!("segment length exceeds maximum u32"))?,
            alignment: FlatBuffer::alignment(),
        })
    }
}
