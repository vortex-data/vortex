use std::sync::Arc;

use futures::Stream;
use futures_util::stream;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexResult};
use vortex_io::VortexReadAt;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::segments::source::SegmentSource;
use crate::Segment;

pub struct FileSegmentSource<R> {
    read: R,
    segment_map: Arc<[Segment]>,
}

struct FileSegmentReader<R> {
    read: R,
    segment_map: Arc<[Segment]>,
}

impl<R> AsyncSegmentReader for FileSegmentSource<R> {
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
        let segment: &Segment = self
            .segment_map
            .get(id)
            .ok_or_else(|| vortex_err!("segment not found"))?;

        Ok(self
            .read
            .read_at(segment.offset, segment.length as usize)
            .await?)
    }
}

impl<R: VortexReadAt> SegmentSource for FileSegmentSource<R> {
    fn reader(&self) -> Arc<dyn AsyncSegmentReader> {
        todo!()
    }

    fn into_driver(self) -> impl Stream<Item = ()> {
        // A no-op driver, since we have no work to do
        stream::repeat(())
    }
}
