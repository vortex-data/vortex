use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{vortex_err, VortexResult};
use vortex_io::VortexWrite;
use vortex_layout::segments::{SegmentId, SegmentWriter};

use crate::footer::Segment;

/// A segment writer that holds buffers in memory until they are flushed by a writer.
#[derive(Default)]
pub(crate) struct BufferedSegmentWriter {
    segments: Vec<Vec<ByteBuffer>>,
    next_id: SegmentId,
}

impl SegmentWriter for BufferedSegmentWriter {
    fn put(&mut self, data: &[ByteBuffer]) -> SegmentId {
        self.segments.push(data.to_vec());
        let id = self.next_id;
        self.next_id = SegmentId::from(*self.next_id + 1);
        id
    }
}

impl BufferedSegmentWriter {
    /// Flush the segments to the provided async writer.
    pub async fn flush_async<W: VortexWrite>(
        &mut self,
        write: &mut futures::io::Cursor<W>,
        segments: &mut Vec<Segment>,
    ) -> VortexResult<()> {
        for buffers in self.segments.drain(..) {
            // The API requires us to write these buffers contiguously. Therefore, we can only
            // respect the alignment of the first one.
            // Don't worry, in most cases the caller knows what they're doing and will align the
            // buffers themselves, inserting padding buffers where necessary.
            let alignment = buffers
                .first()
                .map(|buffer| buffer.alignment())
                .unwrap_or_else(Alignment::none);

            // Add any padding required to align the segment.
            let offset = write.position();
            let padding = offset.next_multiple_of(*alignment as u64) - offset;
            if padding > 0 {
                write
                    .write_all(ByteBuffer::zeroed(padding as usize))
                    .await?;
            }
            let offset = write.position();

            for buffer in buffers {
                write.write_all(buffer).await?;
            }

            segments.push(Segment {
                offset,
                length: u32::try_from(write.position() - offset)
                    .map_err(|_| vortex_err!("segment length exceeds maximum u32"))?,
                alignment,
            });
        }
        Ok(())
    }
}
