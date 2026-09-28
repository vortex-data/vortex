// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
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
use vortex_utils::aliases::hash_map::HashMap;

use crate::scan::planning::SegmentLocation;
use crate::segments::SegmentFuture;
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

/// Serves a split's range requests from a file's segment source, without blocking.
///
/// Each requested range names one segment. [`submit`](IoSource::submit) starts its read at once,
/// [`poll`](IoSource::poll) takes a finished one if there is any, and the split's future awaits
/// the next with [`next_completion`](Self::next_completion) when the driver has nothing else to
/// run. [`wait`](IoSource::wait) is never called. Optional intents are declined.
///
/// The split's registrations are held, never polled, until the source is dropped. A source that
/// shares requests for one segment serves each fetch through its registration, so the bytes are
/// read once however many of the split's reads name them.
pub(super) struct SegmentIoSource {
    segments: Arc<dyn SegmentSource>,
    ranges: SegmentRanges,
    reads: Mutex<FuturesUnordered<BoxFuture<'static, Completion>>>,
    _registered: Mutex<Vec<SegmentFuture>>,
}

impl SegmentIoSource {
    pub(super) fn new(
        segments: Arc<dyn SegmentSource>,
        ranges: SegmentRanges,
        registered: Vec<SegmentFuture>,
    ) -> Self {
        Self {
            segments,
            ranges,
            reads: Mutex::new(FuturesUnordered::new()),
            _registered: Mutex::new(registered),
        }
    }

    /// Waits for the next read to finish.
    pub(super) async fn next_completion(&self) -> VortexResult<Completion> {
        future::poll_fn(|cx| match self.reads.lock().poll_next_unpin(cx) {
            Poll::Ready(Some(completion)) => Poll::Ready(Ok(completion)),
            Poll::Ready(None) => Poll::Ready(Err(vortex_err!(
                "The split's driver is waiting with no reads in flight"
            ))),
            Poll::Pending => Poll::Pending,
        })
        .await
    }
}

impl IoSource for SegmentIoSource {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        let reads = self.reads.lock();
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
            let bytes = self.segments.request(segment);
            let id = request.request;
            reads.push(
                async move {
                    Completion {
                        owner,
                        request: id,
                        result: bytes.await.map(IoResult::Bytes),
                    }
                }
                .boxed(),
            );
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        // The split's future polls again with its own waker before it waits, so a read that
        // finishes after this check is not missed.
        let mut cx = Context::from_waker(Waker::noop());
        match self.reads.lock().poll_next_unpin(&mut cx) {
            Poll::Ready(completion) => Ok(completion),
            Poll::Pending => Ok(None),
        }
    }

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("SegmentIoSource is awaited through next_completion, not waited on")
    }

    fn release(&self, _owner: IoOwnerId) {}

    fn clear(&self) {
        self.reads.lock().clear();
    }
}
