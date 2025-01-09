//! The segment reader provides an async interface to layouts for resolving individual segments.

use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use bytes::Bytes;
use futures::channel::oneshot;
use futures_util::future::try_join_all;
use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::{vortex_err, VortexResult};
use vortex_io::VortexReadAt;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::v2::footer::Segment;

pub(crate) struct SegmentCache<R> {
    read: R,
    segments: Arc<[Segment]>,
    inflight: RwLock<HashMap<SegmentId, Vec<oneshot::Sender<Bytes>>>>,
}

impl<R> SegmentCache<R> {
    pub fn new(read: R, segments: Arc<[Segment]>) -> Self {
        Self {
            read,
            segments,
            inflight: RwLock::new(HashMap::new()),
        }
    }

    pub fn set(&mut self, _segment_id: SegmentId, _bytes: Bytes) -> VortexResult<()> {
        // Do nothing for now
        Ok(())
    }
}

impl<R: VortexReadAt> SegmentCache<R> {
    /// Drives the segment cache.
    pub(crate) async fn drive(&self) -> VortexResult<()>
    where
        Self: Unpin,
    {
        // Grab a read lock and collect a set of segments to read.
        let segment_ids = self
            .inflight
            .read()
            .map_err(|_| vortex_err!("poisoned"))?
            .iter()
            .filter_map(|(id, channels)| (!channels.is_empty()).then_some(*id))
            .collect::<Vec<_>>();

        // Read all the segments.
        let buffers = try_join_all(segment_ids.iter().map(|id| {
            let segment = &self.segments[**id as usize];
            self.read
                .read_byte_range(segment.offset, segment.length as u64)
        }))
        .await?;

        // Send the buffers to the waiting channels.
        let mut inflight = self.inflight.write().map_err(|_| vortex_err!("poisoned"))?;
        for (id, buffer) in segment_ids.into_iter().zip_eq(buffers.into_iter()) {
            let channels = inflight
                .remove(&id)
                .ok_or_else(|| vortex_err!("missing inflight segment"))?;
            for sender in channels {
                sender
                    .send(buffer.clone())
                    .map_err(|_| vortex_err!("receiver dropped"))?;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl<R: VortexReadAt> AsyncSegmentReader for SegmentCache<R> {
    async fn get(&self, id: SegmentId) -> VortexResult<Bytes> {
        let (send, recv) = oneshot::channel();
        self.inflight
            .write()
            .map_err(|_| vortex_err!("poisoned"))?
            .entry(id)
            .or_default()
            .push(send);
        recv.await
            .map_err(|cancelled| vortex_err!("segment read cancelled: {:?}", cancelled))
    }
}
