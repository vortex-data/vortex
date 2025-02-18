use futures::StreamExt;
use vortex_array::stats::PRUNING_STATS;
use vortex_array::stream::ArrayStream;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_flatbuffers::{FlatBuffer, FlatBufferRoot, WriteFlatBuffer, WriteFlatBufferExt};
use vortex_io::VortexWrite;
use vortex_layout::stats::StatsLayoutWriter;
use vortex_layout::{LayoutStrategy, LayoutWriter};

use crate::footer::{FileLayout, Postscript, Segment};
use crate::segments::writer::BufferedSegmentWriter;
use crate::strategy::VortexLayoutStrategy;
use crate::{EOF_SIZE, MAGIC_BYTES, MAX_FOOTER_SIZE, VERSION};

pub struct VortexWriteOptions {
    strategy: Box<dyn LayoutStrategy>,
}

impl Default for VortexWriteOptions {
    fn default() -> Self {
        Self {
            strategy: Box::new(VortexLayoutStrategy::default()),
        }
    }
}

impl VortexWriteOptions {
    /// Replace the default layout strategy with the provided one.
    pub fn with_strategy<S: LayoutStrategy>(mut self, strategy: S) -> Self {
        self.strategy = Box::new(strategy);
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
        // Set up the root layout
        let mut layout_writer = StatsLayoutWriter::new(
            self.strategy.new_writer(stream.dtype())?,
            stream.dtype(),
            PRUNING_STATS.into(),
        )?;

        // First we write the magic number
        let mut write = futures::io::Cursor::new(write);
        write.write_all(MAGIC_BYTES).await?;

        // Our buffered message writer accumulates messages for each batch so we can flush them
        // into the file.
        let mut segment_writer = BufferedSegmentWriter::default();
        let mut segments = vec![];

        // Then write the stream via the root layout
        while let Some(chunk) = stream.next().await {
            layout_writer.push_chunk(&mut segment_writer, chunk?)?;
            // NOTE(ngates): we could spawn this task and continue to compress the next chunk.
            segment_writer
                .flush_async(&mut write, &mut segments)
                .await?;
        }

        // Flush the final layout messages into the file
        let root_layout = layout_writer.finish(&mut segment_writer)?;
        segment_writer
            .flush_async(&mut write, &mut segments)
            .await?;

        // Write the DType + FileLayout segments
        let dtype_segment = self.write_flatbuffer(&mut write, stream.dtype()).await?;
        let file_layout_segment = self
            .write_flatbuffer(
                &mut write,
                &FileLayout::new(
                    root_layout,
                    segments.into(),
                    layout_writer.into_stats_sets().into(),
                ),
            )
            .await?;

        // Assemble the postscript, and write it manually to avoid any framing.
        let postscript = Postscript {
            dtype: dtype_segment,
            file_layout: file_layout_segment,
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
    ) -> VortexResult<Segment> {
        let layout_offset = write.position();
        write.write_all(flatbuffer.write_flatbuffer_bytes()).await?;
        Ok(Segment {
            offset: layout_offset,
            length: u32::try_from(write.position() - layout_offset)
                .map_err(|_| vortex_err!("segment length exceeds maximum u32"))?,
            alignment: FlatBuffer::alignment(),
        })
    }
}
