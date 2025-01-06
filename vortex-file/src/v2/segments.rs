use std::sync::{Arc, RwLock};

use bytes::Bytes;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_io::VortexWrite;
use vortex_layout::segments::{SegmentId, SegmentReader, SegmentWriter};

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

/// A segment cache that holds segments in memory.
///
/// TODO(ngates): switch to a Moka LRU cache.
#[derive(Default, Clone)]
pub(crate) struct SegmentCache {
    segments: Arc<RwLock<HashMap<SegmentId, Bytes>>>,
}

impl SegmentCache {
    pub(crate) fn set(&self, id: SegmentId, data: Bytes) {
        self.segments
            .write()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned")
            .insert(id, data);
    }
}

impl SegmentReader for SegmentCache {
    fn get(&self, id: SegmentId) -> Option<Bytes> {
        self.segments
            .read()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned")
            .get(&id)
            .cloned()
    }
}
