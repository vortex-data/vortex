use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexResult};
use vortex_io::VortexWrite;
use vortex_layout::segments::{SegmentId, SegmentWriter};

use crate::v2::footer::Segment;

/// A segment writer that holds buffers in memory until they are flushed by a writer.
#[derive(Default)]
pub(crate) struct BufferedSegmentWriter {
    segments: Vec<ByteBuffer>,
    next_id: SegmentId,
}

impl SegmentWriter for BufferedSegmentWriter {
    fn put(&mut self, data: ByteBuffer) -> SegmentId {
        self.segments.push(data);
        let id = self.next_id;
        self.next_id = SegmentId::from(*self.next_id + 1);
        id
    }
}

impl BufferedSegmentWriter {
    /// Flush the segments to the provided async writer.
    pub async fn flush_async<W: VortexWrite>(
        &mut self,
        write: &mut futures_util::io::Cursor<W>,
        segments: &mut Vec<Segment>,
    ) -> VortexResult<()> {
        for buffer in self.segments.drain(..) {
            let offset = write.position();
            let alignment = buffer.alignment();
            write.write_all(buffer.into_inner()).await?;
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
