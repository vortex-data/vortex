// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A segment source that discovers IO by being polled.
//!
//! The existing `LayoutReader` asks for segments through `SegmentSource::request` and awaits the
//! futures it gets back; nothing else on its read path performs IO. This source answers from a
//! cache and records every segment it could not answer. A morsel polls the reader's future once
//! per compute, turns the recorded misses into a `NeedsIO` batch keyed by segment id, and feeds
//! delivered bytes back into the cache. Pending leaves honour the waker protocol, so the reader's
//! combinators re-poll them once their bytes arrive, and nothing above the awaits runs twice.

use std::sync::Arc;
use std::task::Poll;
use std::task::Waker;

use futures::FutureExt;
use futures::future;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoBatch;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoTarget;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_utils::aliases::hash_map::HashMap;

use crate::SegmentSpec;

enum Slot {
    /// Requested, not yet delivered; the wakers of every leaf waiting on it.
    Wanted(Vec<Waker>),
    Ready(BufferHandle),
}

/// Answers segment requests from a cache and records the misses for the protocol to fill.
///
/// The miss list is drained by whichever morsel polled last, which is sound under the
/// single-worker driver because only one morsel computes at a time.
pub struct PollingSegmentSource {
    specs: Arc<[SegmentSpec]>,
    slots: Arc<Mutex<HashMap<SegmentId, Slot>>>,
    missed: Mutex<Vec<SegmentId>>,
}

impl PollingSegmentSource {
    /// Creates an empty source over the file's segment specs.
    pub fn new(specs: Arc<[SegmentSpec]>) -> Self {
        Self {
            specs,
            slots: Arc::new(Mutex::new(HashMap::default())),
            missed: Mutex::new(Vec::new()),
        }
    }

    /// Drains the misses recorded since the last call into a batch, one `Fetch` per segment,
    /// keyed by the segment id. Fails for a segment the footer does not describe.
    pub fn take_batch(&self) -> VortexResult<IoBatch> {
        let mut missed: Vec<SegmentId> = self.missed.lock().drain(..).collect();
        missed.sort_unstable();
        missed.dedup();
        missed
            .into_iter()
            .map(|id| {
                let spec = self.spec(id)?;
                Ok(IoRequest {
                    intent: IoIntent::Fetch,
                    request: IoRequestId(*id),
                    target: IoTarget::Range {
                        offset: spec.offset,
                        len: spec.length as usize,
                    },
                })
            })
            .collect()
    }

    /// Stores delivered bytes at the segment's required alignment and wakes every leaf waiting
    /// on it. The protocol's range target carries no alignment yet, so a misaligned delivery is
    /// copied here.
    pub fn deliver(&self, request: IoRequestId, bytes: BufferHandle) -> VortexResult<()> {
        let id = SegmentId::from(request.0);
        let alignment = self.spec(id)?.alignment;
        let bytes = BufferHandle::new_host(bytes.try_into_host_sync()?.aligned(alignment));
        let previous = self.slots.lock().insert(id, Slot::Ready(bytes));
        if let Some(Slot::Wanted(wakers)) = previous {
            wakers.into_iter().for_each(Waker::wake);
        }
        Ok(())
    }

    fn spec(&self, id: SegmentId) -> VortexResult<&SegmentSpec> {
        self.specs
            .get(*id as usize)
            .ok_or_else(|| vortex_err!("segment {id} is not in the footer"))
    }
}

impl SegmentSource for PollingSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        {
            let mut slots = self.slots.lock();
            match slots.get(&id) {
                Some(Slot::Ready(bytes)) => return future::ready(Ok(bytes.clone())).boxed(),
                Some(Slot::Wanted(_)) => {}
                None => {
                    slots.insert(id, Slot::Wanted(Vec::new()));
                }
            }
            // Recorded even when another morsel already wants it, so this morsel's batch names
            // every segment its own completion depends on.
            self.missed.lock().push(id);
        }
        let slots = Arc::clone(&self.slots);
        future::poll_fn(move |cx| {
            let mut slots = slots.lock();
            match slots.get_mut(&id) {
                Some(Slot::Ready(bytes)) => Poll::Ready(Ok(bytes.clone())),
                Some(Slot::Wanted(wakers)) => {
                    if !wakers.iter().any(|waker| waker.will_wake(cx.waker())) {
                        wakers.push(cx.waker().clone());
                    }
                    Poll::Pending
                }
                None => Poll::Ready(Err(vortex_err!("segment {id} was dropped while pending"))),
            }
        })
        .boxed()
    }
}
