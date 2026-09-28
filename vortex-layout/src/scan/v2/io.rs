// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::mpsc;

use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_io::runtime::Handle;
use vortex_utils::aliases::hash_map::HashMap;

use crate::scan::planning::SegmentLocation;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Byte ranges of a file's segments, keyed by `(offset, length)`, back to their segment ids.
pub(super) type SegmentRanges = Arc<HashMap<(u64, usize), SegmentId>>;

pub(super) fn segment_ranges(locations: &[SegmentLocation]) -> SegmentRanges {
    let ranges = locations.iter().enumerate().map(|(id, location)| {
        let id = SegmentId::from(u32::try_from(id).unwrap_or(u32::MAX));
        ((location.offset, location.length as usize), id)
    });
    Arc::new(ranges.collect())
}

/// Serves the driver's range requests from a file's segment source.
///
/// Each requested range names one segment. The read is spawned onto the runtime, and its
/// completion comes back over a channel, so [`wait`](IoSource::wait) blocks only the driver's
/// thread. The driver must therefore not run on a thread the runtime needs to make progress.
/// Optional intents are declined.
pub(super) struct SegmentIoSource {
    segments: Arc<dyn SegmentSource>,
    ranges: SegmentRanges,
    handle: Handle,
    sender: mpsc::Sender<Completion>,
    receiver: Mutex<mpsc::Receiver<Completion>>,
}

impl SegmentIoSource {
    pub(super) fn new(
        segments: Arc<dyn SegmentSource>,
        ranges: SegmentRanges,
        handle: Handle,
    ) -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            segments,
            ranges,
            handle,
            sender,
            receiver: Mutex::new(receiver),
        }
    }
}

impl IoSource for SegmentIoSource {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        for request in batch {
            if request.intent != IoIntent::Fetch {
                continue;
            }
            let IoTarget::Range { offset, len } = request.target else {
                vortex_bail!(
                    "SegmentIoSource only serves byte ranges, not {:?}",
                    request.target
                );
            };
            let id = *self
                .ranges
                .get(&(offset, len))
                .ok_or_else(|| vortex_err!("No segment at bytes {offset}+{len}"))?;
            let read = self.segments.request(id);
            let sender = self.sender.clone();
            self.handle
                .spawn(async move {
                    let result = read.await.map(IoResult::Bytes);
                    // The driver may have stopped listening after an earlier failure.
                    drop(sender.send(Completion {
                        owner,
                        request: request.request,
                        result,
                    }));
                })
                .detach();
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        match self.receiver.lock().try_recv() {
            Ok(completion) => Ok(Some(completion)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => vortex_bail!("SegmentIoSource closed"),
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        self.receiver
            .lock()
            .recv()
            .map_err(|_| vortex_err!("SegmentIoSource closed"))
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {}
}
