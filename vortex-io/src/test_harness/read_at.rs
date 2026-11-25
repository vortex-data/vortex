// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use pin_project_lite::pin_project;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::CoalesceConfig;
use crate::ReadAtRequest;
use crate::ReadAtStream;
use crate::VortexReadAt;
use crate::default_read_ranges;
use crate::test_harness::DeterministicRng;

/// Identifier for an in-flight read registered with [`SimulatedReadAt`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PendingReadId(pub usize);

/// Fault injected into matching reads on a [`SimulatedReadAt`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadFault {
    /// Fail the next `remaining` matching reads with `message`, then succeed.
    TransientError { remaining: usize, message: String },
    /// Fail every matching read with `message`.
    FatalError { message: String },
    /// Return a truncated buffer of at most `max_len` bytes.
    ShortRead { max_len: usize },
}

struct PendingEntry {
    request: ReadAtRequest,
    sender: oneshot::Sender<VortexResult<BufferHandle>>,
}

struct SimReadState {
    next_id: usize,
    gated: bool,
    rng: Option<DeterministicRng>,
    poll_delay_max: usize,
    pending: BTreeMap<PendingReadId, PendingEntry>,
    active_ungated: BTreeMap<PendingReadId, ReadAtRequest>,
    offset_faults: BTreeMap<u64, VecDeque<ReadFault>>,
    global_faults: VecDeque<ReadFault>,
    in_flight: usize,
    max_in_flight: usize,
    cancelled_reads: usize,
    batch_sizes: Vec<usize>,
}

impl SimReadState {
    fn take_fault(&mut self, offset: u64) -> Option<ReadFault> {
        if let Some(queue) = self.offset_faults.get_mut(&offset)
            && let Some(fault) = consume_fault(queue)
        {
            return Some(fault);
        }
        consume_fault(&mut self.global_faults)
    }

    fn next_poll_delay(&mut self) -> usize {
        if self.poll_delay_max == 0 {
            return 0;
        }
        match self.rng.as_mut() {
            Some(rng) => rng.next_usize(self.poll_delay_max + 1),
            None => 0,
        }
    }
}

fn consume_fault(queue: &mut VecDeque<ReadFault>) -> Option<ReadFault> {
    while let Some(front) = queue.front_mut() {
        match front {
            ReadFault::TransientError { remaining, message } => {
                if *remaining == 0 {
                    queue.pop_front();
                    continue;
                }
                *remaining -= 1;
                let msg = message.clone();
                if *remaining == 0 {
                    queue.pop_front();
                }
                return Some(ReadFault::FatalError { message: msg });
            }
            ReadFault::FatalError { .. } | ReadFault::ShortRead { .. } => {
                return Some(front.clone());
            }
        }
    }
    None
}

struct SimReadInner {
    data: ByteBuffer,
    uri: Option<Arc<str>>,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
    state: Mutex<SimReadState>,
}

impl SimReadInner {
    fn resolve_bytes(
        &self,
        request: ReadAtRequest,
        fault: Option<ReadFault>,
    ) -> VortexResult<BufferHandle> {
        match fault {
            Some(ReadFault::FatalError { message })
            | Some(ReadFault::TransientError { message, .. }) => {
                vortex_bail!("{message}");
            }
            Some(ReadFault::ShortRead { max_len }) => {
                let start = usize::try_from(request.offset)
                    .vortex_expect("simulated read offset fits in usize");
                let actual_len = request.length.min(max_len);
                let end = start.saturating_add(actual_len).min(self.data.len());
                if start > self.data.len() {
                    vortex_bail!(
                        "Requested range {}..{} out of bounds for buffer of length {}",
                        start,
                        start.saturating_add(request.length),
                        self.data.len()
                    );
                }
                Ok(BufferHandle::new_host(
                    self.data
                        .slice_unaligned(start..end)
                        .aligned(request.alignment),
                ))
            }
            None => {
                let start = usize::try_from(request.offset)
                    .vortex_expect("simulated read offset fits in usize");
                let end = usize::try_from(
                    request
                        .offset
                        .checked_add(request.length as u64)
                        .vortex_expect("simulated read end does not overflow u64"),
                )
                .vortex_expect("simulated read end fits in usize");
                if end > self.data.len() {
                    vortex_bail!(
                        "Requested range {}..{} out of bounds for buffer of length {}",
                        start,
                        end,
                        self.data.len()
                    );
                }
                Ok(BufferHandle::new_host(
                    self.data
                        .slice_unaligned(start..end)
                        .aligned(request.alignment),
                ))
            }
        }
    }
}

struct GatedDropGuard {
    inner: Arc<SimReadInner>,
    id: PendingReadId,
}

impl Drop for GatedDropGuard {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock();
        if state.pending.remove(&self.id).is_some() {
            state.in_flight -= 1;
            state.cancelled_reads += 1;
        }
    }
}

struct UngatedDropGuard {
    inner: Arc<SimReadInner>,
    id: PendingReadId,
}

impl Drop for UngatedDropGuard {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock();
        if state.active_ungated.remove(&self.id).is_some() {
            state.in_flight -= 1;
            state.cancelled_reads += 1;
        }
    }
}

pin_project! {
    /// Ensures cancellation drop tracking runs even if the returned future is dropped before
    /// its first poll.
    struct TrackedReadFuture<F, G> {
        #[pin]
        fut: F,
        _guard: G,
    }
}

impl<F: Future, G> Future for TrackedReadFuture<F, G> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.project().fut.poll(cx)
    }
}

struct PollDelay {
    remaining: usize,
}

impl Future for PollDelay {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.remaining == 0 {
            Poll::Ready(())
        } else {
            self.remaining -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Deterministic, controllable [`VortexReadAt`] for testing concurrency, out-of-order completion,
/// fault injection, short reads, and cancellation without sleeps.
///
/// Configure builder options (`with_*`) before cloning; cloning shares the same backing state and
/// controller handle.
#[derive(Clone)]
pub struct SimulatedReadAt {
    inner: Arc<SimReadInner>,
}

impl SimulatedReadAt {
    /// Creates an ungated [`SimulatedReadAt`] backed by `data` with concurrency 4.
    pub fn new(data: impl Into<ByteBuffer>) -> Self {
        Self::build(data.into(), false)
    }

    /// Creates a gated [`SimulatedReadAt`] where every `read_at` pauses until explicitly
    /// completed via controller methods such as [`complete`](Self::complete) or
    /// [`complete_all`](Self::complete_all).
    pub fn gated(data: impl Into<ByteBuffer>) -> Self {
        Self::build(data.into(), true)
    }

    fn build(data: ByteBuffer, gated: bool) -> Self {
        Self {
            inner: Arc::new(SimReadInner {
                data,
                uri: None,
                concurrency: 4,
                coalesce_config: None,
                state: Mutex::new(SimReadState {
                    next_id: 0,
                    gated,
                    rng: None,
                    poll_delay_max: 0,
                    pending: BTreeMap::new(),
                    active_ungated: BTreeMap::new(),
                    offset_faults: BTreeMap::new(),
                    global_faults: VecDeque::new(),
                    in_flight: 0,
                    max_in_flight: 0,
                    cancelled_reads: 0,
                    batch_sizes: Vec::new(),
                }),
            }),
        }
    }

    fn inner_mut(&mut self) -> &mut SimReadInner {
        Arc::get_mut(&mut self.inner)
            .vortex_expect("SimulatedReadAt builder methods must be called before cloning")
    }

    /// Configures the URI reported by [`VortexReadAt::uri`]. Must be called before cloning.
    #[must_use]
    pub fn with_uri(mut self, uri: impl Into<Arc<str>>) -> Self {
        self.inner_mut().uri = Some(uri.into());
        self
    }

    /// Configures the concurrency limit reported by [`VortexReadAt::concurrency`]. Must be called
    /// before cloning.
    #[must_use]
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.inner_mut().concurrency = concurrency;
        self
    }

    /// Configures the coalescing settings reported by [`VortexReadAt::coalesce_config`]. Must be
    /// called before cloning.
    #[must_use]
    pub fn with_coalesce_config(mut self, config: CoalesceConfig) -> Self {
        self.inner_mut().coalesce_config = Some(config);
        self
    }

    /// Seeds deterministic ordering used by [`complete_all`](Self::complete_all). Must be called
    /// before cloning.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.inner_mut().state.get_mut().rng = Some(DeterministicRng::new(seed));
        self
    }

    /// In ungated mode, delays each read by a seeded number of self-waking `Pending` polls
    /// in `0..=max_pending_polls` so concurrent reads complete out of order deterministically
    /// without using timers or sleeps. Must be called before cloning.
    #[must_use]
    pub fn with_seeded_poll_delays(mut self, seed: u64, max_pending_polls: usize) -> Self {
        let state = self.inner_mut().state.get_mut();
        state.rng = Some(DeterministicRng::new(seed));
        state.poll_delay_max = max_pending_polls;
        self
    }

    /// Injects a fault for reads starting at `offset`.
    pub fn inject_fault_for_offset(&self, offset: u64, fault: ReadFault) {
        self.inner
            .state
            .lock()
            .offset_faults
            .entry(offset)
            .or_default()
            .push_back(fault);
    }

    /// Injects a fault that applies to all reads (after any offset-specific faults).
    pub fn inject_fault_for_all(&self, fault: ReadFault) {
        self.inner.state.lock().global_faults.push_back(fault);
    }

    /// Completes a specific gated read using the backing buffer (and any injected fault).
    pub fn complete(&self, id: PendingReadId) -> bool {
        let entry = {
            let mut state = self.inner.state.lock();
            let Some(entry) = state.pending.remove(&id) else {
                return false;
            };
            let fault = state.take_fault(entry.request.offset);
            let result = self.inner.resolve_bytes(entry.request, fault);
            state.in_flight -= 1;
            Some((entry.sender, result))
        };

        if let Some((sender, result)) = entry {
            drop(sender.send(result));
            true
        } else {
            false
        }
    }

    /// Completes a specific gated read with an explicit result.
    pub fn complete_with(&self, id: PendingReadId, result: VortexResult<BufferHandle>) -> bool {
        let sender = {
            let mut state = self.inner.state.lock();
            let Some(entry) = state.pending.remove(&id) else {
                return false;
            };
            state.in_flight -= 1;
            entry.sender
        };

        drop(sender.send(result));
        true
    }

    /// Completes the oldest pending read at `offset`, returning its id if found.
    pub fn complete_offset(&self, offset: u64) -> Option<PendingReadId> {
        let id = {
            let state = self.inner.state.lock();
            state
                .pending
                .iter()
                .find(|(_, entry)| entry.request.offset == offset)
                .map(|(id, _)| *id)
        }?;
        self.complete(id).then_some(id)
    }

    /// Completes the oldest pending read (FIFO).
    pub fn complete_next(&self) -> Option<PendingReadId> {
        let id = self.inner.state.lock().pending.keys().next().copied()?;
        self.complete(id).then_some(id)
    }

    /// Completes the newest pending read (LIFO).
    pub fn complete_last(&self) -> Option<PendingReadId> {
        let id = self
            .inner
            .state
            .lock()
            .pending
            .keys()
            .next_back()
            .copied()?;
        self.complete(id).then_some(id)
    }

    /// Completes all currently pending reads (in seeded order if seeded, otherwise FIFO).
    pub fn complete_all(&self) -> usize {
        let mut ids: Vec<PendingReadId> = {
            let mut state = self.inner.state.lock();
            let mut ids: Vec<_> = state.pending.keys().copied().collect();
            if let Some(rng) = state.rng.as_mut() {
                rng.shuffle(&mut ids);
            }
            ids
        };
        let mut count = 0;
        for id in ids.drain(..) {
            if self.complete(id) {
                count += 1;
            }
        }
        count
    }

    /// Lists all currently pending gated read ids in registration order.
    pub fn pending_ids(&self) -> Vec<PendingReadId> {
        self.inner.state.lock().pending.keys().copied().collect()
    }

    /// Number of currently paused gated reads.
    pub fn pending_count(&self) -> usize {
        self.inner.state.lock().pending.len()
    }

    /// Number of `read_at` futures that have been issued and have not yet completed or been
    /// dropped.
    ///
    /// Note that the default [`VortexReadAt::read_ranges`] creates all per-range futures in a
    /// batch up front before polling them via `buffer_unordered`, so this reflects issued futures
    /// rather than only polled futures.
    pub fn in_flight_count(&self) -> usize {
        self.inner.state.lock().in_flight
    }

    /// High-water mark of concurrently issued, unfinished `read_at` futures (see
    /// [`in_flight_count`](Self::in_flight_count)).
    pub fn max_in_flight(&self) -> usize {
        self.inner.state.lock().max_in_flight
    }

    /// Number of reads whose futures were dropped before completing.
    pub fn cancelled_reads(&self) -> usize {
        self.inner.state.lock().cancelled_reads
    }

    /// Recorded batch lengths passed to [`VortexReadAt::read_ranges`].
    pub fn read_ranges_batch_sizes(&self) -> Vec<usize> {
        self.inner.state.lock().batch_sizes.clone()
    }
}

impl VortexReadAt for SimulatedReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        self.inner.uri.as_ref()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.inner.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.inner.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.inner.data.len() as u64;
        async move { Ok(len) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let request = ReadAtRequest::new(offset, length, alignment);
        let inner = Arc::clone(&self.inner);

        let (id, gated, delay, receiver) = {
            let mut state = inner.state.lock();
            let id = PendingReadId(state.next_id);
            state.next_id += 1;
            state.in_flight += 1;
            state.max_in_flight = state.max_in_flight.max(state.in_flight);

            if state.gated {
                let (sender, receiver) = oneshot::channel();
                state.pending.insert(id, PendingEntry { request, sender });
                (id, true, 0, Some(receiver))
            } else {
                state.active_ungated.insert(id, request);
                let delay = state.next_poll_delay();
                (id, false, delay, None)
            }
        };

        if gated {
            let receiver = receiver.vortex_expect("gated read always creates receiver");
            let guard = GatedDropGuard { inner, id };
            TrackedReadFuture {
                fut: async move {
                    receiver.await.map_err(|_| {
                        vortex_err!("SimulatedReadAt dropped while read {id:?} pending")
                    })?
                },
                _guard: guard,
            }
            .boxed()
        } else {
            let guard = UngatedDropGuard {
                inner: Arc::clone(&inner),
                id,
            };
            TrackedReadFuture {
                fut: async move {
                    if delay > 0 {
                        PollDelay { remaining: delay }.await;
                    }
                    let mut state = inner.state.lock();
                    let fault = state.take_fault(request.offset);
                    let result = inner.resolve_bytes(request, fault);
                    if state.active_ungated.remove(&id).is_some() {
                        state.in_flight -= 1;
                    }
                    result
                },
                _guard: guard,
            }
            .boxed()
        }
    }

    fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
        self.inner.state.lock().batch_sizes.push(requests.len());
        default_read_ranges(self, requests)
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::test_harness::SimulatedExecutor;

    #[test]
    fn test_gated_out_of_order_and_cancellation() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let sim =
            SimulatedReadAt::gated(ByteBuffer::from(b"abcdefgh".to_vec())).with_concurrency(3);

        let f0 = sim.read_at(0, 2, Alignment::none());
        let f1 = sim.read_at(2, 2, Alignment::none());
        let f2 = sim.read_at(4, 2, Alignment::none());

        assert_eq!(sim.in_flight_count(), 3);
        assert_eq!(sim.max_in_flight(), 3);
        assert_eq!(sim.pending_count(), 3);

        // Drop the middle read before first poll and verify eager cancellation tracking.
        drop(f1);
        assert_eq!(sim.cancelled_reads(), 1);
        assert_eq!(sim.in_flight_count(), 2);

        // Complete the newest remaining read first (out of order).
        let last_id = sim.complete_last().vortex_expect("pending read exists");
        assert_eq!(last_id, PendingReadId(2));
        let buf2 = exec.drive(f2)?;
        assert_eq!(exec.drive(buf2.to_host()).as_ref(), b"ef");

        let first_id = sim.complete_next().vortex_expect("pending read exists");
        assert_eq!(first_id, PendingReadId(0));
        let buf0 = exec.drive(f0)?;
        assert_eq!(exec.drive(buf0.to_host()).as_ref(), b"ab");

        assert_eq!(sim.in_flight_count(), 0);
        Ok(())
    }

    #[test]
    fn test_fault_injection_transient_fatal_and_short_read() -> VortexResult<()> {
        let exec = SimulatedExecutor::new();
        let sim = SimulatedReadAt::new(ByteBuffer::from(b"0123456789".to_vec()));

        sim.inject_fault_for_offset(
            0,
            ReadFault::TransientError {
                remaining: 1,
                message: "transient boom".to_string(),
            },
        );
        sim.inject_fault_for_offset(
            4,
            ReadFault::FatalError {
                message: "fatal boom".to_string(),
            },
        );
        sim.inject_fault_for_offset(6, ReadFault::ShortRead { max_len: 2 });

        let err = exec
            .drive(sim.read_at(0, 4, Alignment::none()))
            .expect_err("transient fault must trigger on first attempt");
        assert!(err.to_string().contains("transient boom"), "{err}");

        // Second read at offset 0 succeeds once transient budget is exhausted.
        let ok = exec.drive(sim.read_at(0, 4, Alignment::none()))?;
        assert_eq!(exec.drive(ok.to_host()).as_ref(), b"0123");

        // Fatal error persists across retries.
        for _ in 0..2 {
            let err = exec
                .drive(sim.read_at(4, 2, Alignment::none()))
                .expect_err("fatal fault must persist");
            assert!(err.to_string().contains("fatal boom"), "{err}");
        }

        // Short read returns truncated slice.
        let short = exec.drive(sim.read_at(6, 4, Alignment::none()))?;
        assert_eq!(exec.drive(short.to_host()).as_ref(), b"67");
        Ok(())
    }

    #[test]
    fn test_seeded_poll_delays_produce_deterministic_order() -> VortexResult<()> {
        let run = |seed: u64| -> VortexResult<Vec<u64>> {
            let exec = SimulatedExecutor::with_seed(seed);
            let sim = SimulatedReadAt::new(ByteBuffer::from(vec![0u8; 32]))
                .with_concurrency(8)
                .with_seeded_poll_delays(seed, 5);
            let requests: Arc<[ReadAtRequest]> = (0..8)
                .map(|i| ReadAtRequest::new(i * 4, 4, Alignment::none()))
                .collect();
            let items = exec.drive(sim.read_ranges(requests).collect::<Vec<_>>());
            let mut offsets = Vec::with_capacity(items.len());
            for (req, res) in items {
                res?;
                offsets.push(req.offset);
            }
            Ok(offsets)
        };

        assert_eq!(run(99)?, run(99)?);
        assert_ne!(run(99)?, run(100)?);
        Ok(())
    }
}
