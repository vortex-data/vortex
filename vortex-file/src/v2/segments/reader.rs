//! The segment reader provides an async interface to layouts for resolving individual segments.

use bytes::Bytes;
use futures::channel::oneshot;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::VortexResult;
use vortex_layout::segments::SegmentId;

type OneShot<T> = (oneshot::Sender<T>, oneshot::Receiver<T>);

pub struct SegmentCache {
    inflight: HashMap<SegmentId, Vec<OneShot<Bytes>>>,
}

impl SegmentCache {
    /// Returns a future that waits for a segment to be available.
    pub async fn get_segment(&mut self, id: SegmentId) -> VortexResult<Bytes> {
        let channel = oneshot::channel();
        self.inflight.entry(id).or_default().push(channel);
        channel.1.await.map_err(|e| e.into())
    }
}
