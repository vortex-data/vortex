use bytes::Bytes;
use vortex_error::{vortex_err, VortexResult};
use vortex_io::VortexWrite;
use vortex_layout::segments::{SegmentId, SegmentWriter};

use crate::v2::footer::Segment;

/// A segment writer that holds buffers in memory until they are flushed by a writer.
#[derive(Default)]
pub(crate) struct BufferedSegmentWriter {
    segments: Vec<Vec<Bytes>>,
    next_id: SegmentId,
}

impl SegmentWriter for BufferedSegmentWriter {
    fn put(&mut self, data: Vec<Bytes>) -> SegmentId {
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
        for segment in self.segments.drain(..) {
            let offset = write.position();
            for buffer in segment {
                write.write_all(buffer).await?;
            }
            let length = usize::try_from(write.position() - offset)
                .map_err(|_| vortex_err!("segment length exceeds maximum usize"))?;
            segments.push(Segment { offset, length });
        }
        Ok(())
    }
}
