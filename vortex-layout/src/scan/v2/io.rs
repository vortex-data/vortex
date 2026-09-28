// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::sync::Arc;
use std::sync::mpsc;

use futures::FutureExt;
use futures::StreamExt;
use futures::channel::mpsc as async_mpsc;
use futures::select;
use futures::stream::FuturesUnordered;
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
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

/// One read the driver asked for.
pub(super) struct Read {
    owner: IoOwnerId,
    request: IoRequestId,
    segment: SegmentId,
}

/// The driver's side of a split's IO: requests go out to [`pump`], completions come back.
///
/// Each requested range names one segment. The driver thread never starts a read itself, so it
/// needs no runtime; it only blocks in [`wait`](IoSource::wait). Optional intents are declined.
pub(super) struct SegmentIoSource {
    ranges: SegmentRanges,
    reads: async_mpsc::UnboundedSender<Read>,
    completions: Mutex<mpsc::Receiver<Completion>>,
}

/// Creates a split's IO source and the channels [`pump`] serves it through.
pub(super) fn segment_io(
    ranges: SegmentRanges,
) -> (
    SegmentIoSource,
    async_mpsc::UnboundedReceiver<Read>,
    mpsc::Sender<Completion>,
) {
    let (reads, read_receiver) = async_mpsc::unbounded();
    let (completion_sender, completions) = mpsc::channel();
    let source = SegmentIoSource {
        ranges,
        reads,
        completions: Mutex::new(completions),
    };
    (source, read_receiver, completion_sender)
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
            let segment = *self
                .ranges
                .get(&(offset, len))
                .ok_or_else(|| vortex_err!("No segment at bytes {offset}+{len}"))?;
            self.reads
                .unbounded_send(Read {
                    owner,
                    request: request.request,
                    segment,
                })
                .map_err(|_| vortex_err!("The split stopped serving reads"))?;
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        match self.completions.lock().try_recv() {
            Ok(completion) => Ok(Some(completion)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                vortex_bail!("The split stopped serving reads")
            }
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        self.completions
            .lock()
            .recv()
            .map_err(|_| vortex_err!("The split stopped serving reads"))
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {}
}

/// Serves a split's reads from `segments` until `driver` finishes, and returns its result.
///
/// Runs in the split's own future, so reads are started and polled on whatever runtime drives
/// the scan, concurrently with each other and with the driver's thread.
pub(super) async fn pump<R>(
    segments: Arc<dyn SegmentSource>,
    requests: async_mpsc::UnboundedReceiver<Read>,
    completions: mpsc::Sender<Completion>,
    driver: impl Future<Output = VortexResult<R>>,
) -> VortexResult<R> {
    let mut requests = requests.fuse();
    let mut reads = FuturesUnordered::new();
    let mut driver = Box::pin(driver.fuse());
    loop {
        select! {
            result = driver => return result,
            read = requests.next() => if let Some(read) = read {
                let bytes = segments.request(read.segment);
                reads.push(async move { (read.owner, read.request, bytes.await) });
            },
            done = reads.select_next_some() => {
                let (owner, request, result) = done;
                // The driver may have stopped listening after an earlier failure.
                drop(completions.send(Completion {
                    owner,
                    request,
                    result: result.map(IoResult::Bytes),
                }));
            },
        }
    }
}
