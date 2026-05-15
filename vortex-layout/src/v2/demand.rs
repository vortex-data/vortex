// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// All `u64 ↔ usize` casts in this module are bounded by the
// partition's row count, which is itself a usize at the engine
// level. On 64-bit targets the casts are exact; on 32-bit they'd
// already have been unsafe upstream. Keeping `as` casts for
// readability since the conversions appear in tight inner loops.
#![allow(clippy::cast_possible_truncation, clippy::cast_lossless)]

//! [`RowDemand`] — partition-local SIP for tracking which rows
//! still need work.
//!
//! ## Model
//!
//! A `RowDemand` represents one partition's "rows still needed"
//! state. Bits start at all-1 (every row needed) and only go
//! `1 → 0` (rows newly known not to be needed). Multiple producers
//! independently meet their reductions in; the result is the AND
//! of all contributions.
//!
//! Producers don't need to coordinate — there's no hard-coded
//! ordering. The AND-of-monotone semantic guarantees commutativity:
//! whoever publishes first wins for those rows, and others
//! contribute when they finish. Consumers see the running
//! intersection.
//!
//! ## Consumers and threshold-based wakers
//!
//! A consumer registers a [`WaitPredicate`] over a row range:
//!
//! - [`WaitPredicate::Zero`] — wake when cardinality across `range`
//!   drops to zero. Use case: skip downstream work entirely (the
//!   filter has rejected every row).
//! - [`WaitPredicate::Below(threshold)`] — wake when cardinality
//!   drops below `threshold`. Use case: a consumer that was about to
//!   eagerly submit I/O reconsiders if too few rows survive to be
//!   worth fetching.
//!
//! Wakes fire only when the relevant threshold is crossed. Other
//! producer publishes don't disturb the consumer. This avoids the
//! "wake everyone on every update" overhead.
//!
//! Plus an EOF wake: when all producer handles drop, every
//! registered consumer is woken so it can inspect the final state.
//!
//! ## Per-window state
//!
//! The row space is split into fixed-size windows
//! ([`RowDemand::WINDOW_ROWS`] = 4 K rows). Per-window state
//! ([`Window`]) holds the demand bits, a cached cardinality (so
//! consumer reads don't have to popcount), and per-threshold
//! waiter lists. Locks are per-window so unrelated producer
//! publishes don't contend.

use std::any::Any;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::Future;
use parking_lot::Mutex;
use smallvec::SmallVec;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_mask::Mask;

use crate::v2::scan_ctx::ScanCtx;
use crate::v2::scan_ctx::ScanCtxValue;

/// A row range within a partition's row space. Half-open: `[start, end)`.
pub type RowRange = Range<u64>;

/// Per-window row count. Smaller = finer waker granularity but more
/// metadata; larger = coarser. 4096 fits common chunk granularities
/// and keeps each window's bit buffer at 512 B.
const WINDOW_ROWS: u64 = 4096;

/// Partition-local demand state. Created with [`RowDemand::new`] for
/// a fixed total row count; producers publish reductions, consumers
/// register threshold-based waits.
#[derive(Debug)]
pub struct RowDemand {
    total_rows: u64,
    /// One per [`WINDOW_ROWS`] slice of the partition's row space.
    /// The last window may be partial.
    windows: Vec<Window>,
    /// Number of live `Producer` handles. Decremented on each
    /// `Producer::drop`; reaching 0 fires the EOF wakers.
    active_producers: AtomicUsize,
    eof: AtomicBool,
    /// EOF waiters — woken once when `active_producers` reaches 0.
    eof_state: Mutex<EofState>,
}

#[derive(Default, Debug)]
struct EofState {
    waiters: Vec<Waker>,
}

#[derive(Debug)]
struct Window {
    state: Mutex<WindowState>,
    /// Cached popcount of `state.bits` — readable without locking.
    cardinality: AtomicU64,
}

#[derive(Debug)]
struct WindowState {
    bits: BitBuffer,
    /// Wake when cardinality drops to 0.
    waiters_zero: SmallVec<[Waker; 4]>,
    /// Wake when cardinality drops *below* `threshold`. Stored as
    /// `(threshold, waker)` pairs.
    waiters_below: SmallVec<[(u64, Waker); 4]>,
}

/// What a consumer is waiting on.
#[derive(Clone, Copy, Debug)]
pub enum WaitPredicate {
    /// Cardinality across `range` is 0.
    Zero,
    /// Cardinality across `range` is below `threshold`.
    Below(u64),
}

/// Reason a [`WaitFuture`] resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitResult {
    /// The predicate fired (cardinality crossed the threshold).
    PredicateFired,
    /// All producers signalled done; demand is now stable. The
    /// predicate may or may not have fired — consumer should inspect
    /// current cardinality to decide.
    AllProducersDone,
}

impl RowDemand {
    /// Window size (in rows) used by the per-window mutex sharding.
    pub const WINDOW_ROWS: u64 = WINDOW_ROWS;

    /// Create a fresh `RowDemand` covering `total_rows` rows. All
    /// rows start as "demanded" (bits=1). No producers active until
    /// [`Self::producer`] is called at least once.
    pub fn new(total_rows: u64) -> Arc<Self> {
        let window_count = total_rows.div_ceil(WINDOW_ROWS) as usize;
        let mut windows = Vec::with_capacity(window_count);
        for i in 0..window_count {
            let start = (i as u64) * WINDOW_ROWS;
            let end = (start + WINDOW_ROWS).min(total_rows);
            let len = (end - start) as usize;
            let bits = BitBufferMut::new_set(len).freeze();
            windows.push(Window {
                state: Mutex::new(WindowState {
                    bits,
                    waiters_zero: SmallVec::new(),
                    waiters_below: SmallVec::new(),
                }),
                cardinality: AtomicU64::new(len as u64),
            });
        }
        Arc::new(Self {
            total_rows,
            windows,
            active_producers: AtomicUsize::new(0),
            eof: AtomicBool::new(false),
            eof_state: Mutex::new(EofState::default()),
        })
    }

    /// Empty demand state — covers zero rows. Used by `PlanCtx`
    /// constructors that haven't decided on a real `RowDemand` yet.
    /// `consumer().cardinality()` returns 0; predicate waits resolve
    /// immediately as `AllProducersDone`.
    pub fn empty() -> Arc<Self> {
        Arc::new(Self {
            total_rows: 0,
            windows: Vec::new(),
            active_producers: AtomicUsize::new(0),
            eof: AtomicBool::new(true),
            eof_state: Mutex::new(EofState::default()),
        })
    }

    /// Total row count this `RowDemand` covers.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Allocate a producer handle. Bumps the active-producer count;
    /// the corresponding [`Producer::drop`] decrements it.
    pub fn producer(self: &Arc<Self>) -> Producer {
        self.active_producers.fetch_add(1, Ordering::AcqRel);
        Producer {
            demand: Arc::clone(self),
        }
    }

    /// Allocate a consumer handle. Consumer handles are cheap;
    /// dropping them is a no-op.
    pub fn consumer(self: &Arc<Self>) -> Consumer {
        Consumer {
            demand: Arc::clone(self),
        }
    }

    /// Range of window indices covering `range`.
    fn windows_in(&self, range: &RowRange) -> Range<usize> {
        if range.start >= range.end || self.windows.is_empty() {
            return 0..0;
        }
        let start = (range.start / WINDOW_ROWS) as usize;
        let end = ((range.end - 1) / WINDOW_ROWS) as usize + 1;
        start..end.min(self.windows.len())
    }
}

/// Producer side. Drop the handle once you're done publishing — the
/// last live producer's drop fires EOF.
pub struct Producer {
    demand: Arc<RowDemand>,
}

impl Producer {
    /// AND `mask` into the demand bits covering `range`. `mask`
    /// must have `range.len()` rows.
    ///
    /// Wakes any consumer whose threshold is crossed by this update.
    pub fn publish(&self, range: RowRange, mask: &Mask) {
        let abs_start = range.start;
        for window_idx in self.demand.windows_in(&range) {
            let window = &self.demand.windows[window_idx];
            let window_start = (window_idx as u64) * WINDOW_ROWS;
            let window_end = window_start + window.cardinality_capacity_u64();
            // Compute intersection of `range` with this window.
            let isect_start = range.start.max(window_start);
            let isect_end = range.end.min(window_end);
            if isect_start >= isect_end {
                continue;
            }
            // Local indices into the window's bit buffer.
            let local_start = (isect_start - window_start) as usize;
            let local_end = (isect_end - window_start) as usize;
            // Index into the input `mask` for this slice.
            let mask_start = (isect_start - abs_start) as usize;
            let mask_end = (isect_end - abs_start) as usize;

            let mut state = window.state.lock();
            and_into(
                &mut state.bits,
                local_start,
                local_end,
                mask,
                mask_start,
                mask_end,
            );

            // Recompute cardinality for the window.
            let new_card = state.bits.true_count() as u64;
            let prev_card = window.cardinality.swap(new_card, Ordering::AcqRel);
            // If nothing changed for this window, don't bother waking.
            if new_card == prev_card {
                continue;
            }

            // Wake `waiters_zero` if we hit zero.
            if new_card == 0 && !state.waiters_zero.is_empty() {
                for waker in state.waiters_zero.drain(..) {
                    waker.wake();
                }
            }
            // Wake `waiters_below` whose threshold was crossed.
            if !state.waiters_below.is_empty() {
                let mut i = 0;
                while i < state.waiters_below.len() {
                    if new_card < state.waiters_below[i].0 {
                        let (_, waker) = state.waiters_below.swap_remove(i);
                        waker.wake();
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }
}

impl Drop for Producer {
    fn drop(&mut self) {
        let prev = self.demand.active_producers.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            // Last producer dropped — fire EOF.
            self.demand.eof.store(true, Ordering::Release);
            // Wake all per-window threshold waiters too — they can
            // now decide based on final state. (They'll see
            // active_producers == 0 and the future will resolve as
            // AllProducersDone.)
            for window in &self.demand.windows {
                let mut state = window.state.lock();
                for waker in state.waiters_zero.drain(..) {
                    waker.wake();
                }
                for (_, waker) in state.waiters_below.drain(..) {
                    waker.wake();
                }
            }
            // Wake EOF-only waiters.
            let mut eof_state = self.demand.eof_state.lock();
            for waker in eof_state.waiters.drain(..) {
                waker.wake();
            }
        }
    }
}

/// Consumer side. Cheap to clone (it's just an Arc handle).
#[derive(Clone)]
pub struct Consumer {
    demand: Arc<RowDemand>,
}

impl Consumer {
    /// Sum of cached cardinality across windows in `range`. Cheap —
    /// each window's cardinality is an `AtomicU64` read. This is an
    /// upper bound (the over-estimate is by at most a window-edge
    /// fraction; per-window cardinality counts the whole window even
    /// if `range` only partially covers it). For exact counts use
    /// [`Self::cardinality_exact`].
    pub fn cardinality(&self, range: RowRange) -> u64 {
        let mut total = 0u64;
        for window_idx in self.demand.windows_in(&range) {
            total += self.demand.windows[window_idx]
                .cardinality
                .load(Ordering::Acquire);
        }
        total
    }

    /// Exact cardinality over `range` (handles partial-window edges).
    /// Locks each affected window briefly.
    pub fn cardinality_exact(&self, range: RowRange) -> u64 {
        let mut total = 0u64;
        for window_idx in self.demand.windows_in(&range) {
            let window = &self.demand.windows[window_idx];
            let window_start = (window_idx as u64) * WINDOW_ROWS;
            let window_end = window_start + window.cardinality_capacity_u64();
            let isect_start = range.start.max(window_start);
            let isect_end = range.end.min(window_end);
            if isect_start >= isect_end {
                continue;
            }
            if isect_start == window_start && isect_end == window_end {
                total += window.cardinality.load(Ordering::Acquire);
            } else {
                let local_start = (isect_start - window_start) as usize;
                let local_end = (isect_end - window_start) as usize;
                let state = window.state.lock();
                total += slice_true_count(&state.bits, local_start, local_end) as u64;
            }
        }
        total
    }

    /// True iff all producers have signalled done (no more updates).
    pub fn is_eof(&self) -> bool {
        self.demand.eof.load(Ordering::Acquire)
    }

    /// Wait until `predicate` is satisfied for `range`, or until all
    /// producers signal done. Returns whichever happened first.
    /// Wakes only fire on threshold crossings or EOF — unrelated
    /// producer publishes do not wake this future.
    pub fn wait_for(&self, range: RowRange, predicate: WaitPredicate) -> WaitFuture {
        WaitFuture {
            consumer: self.clone(),
            range,
            predicate,
            registered_windows: SmallVec::new(),
        }
    }
}

impl Window {
    fn cardinality_capacity_u64(&self) -> u64 {
        self.state.lock().bits.len() as u64
    }
}

/// Future returned by [`Consumer::wait_for`].
pub struct WaitFuture {
    consumer: Consumer,
    range: RowRange,
    predicate: WaitPredicate,
    /// Window indices we've registered wakers in (so we know what
    /// to clean up on Drop, though we never explicitly clean up —
    /// stale wakers are wasted wakes, not bugs).
    registered_windows: SmallVec<[usize; 4]>,
}

impl WaitFuture {
    /// Check the predicate against current cardinality.
    fn predicate_satisfied(&self) -> bool {
        let card = self.consumer.cardinality(self.range.clone());
        match self.predicate {
            WaitPredicate::Zero => card == 0,
            WaitPredicate::Below(t) => card < t,
        }
    }
}

impl Future for WaitFuture {
    type Output = WaitResult;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        // Check predicate first — cheap atomic loads.
        if this.predicate_satisfied() {
            return Poll::Ready(WaitResult::PredicateFired);
        }
        // Check EOF.
        if this.consumer.is_eof() {
            return Poll::Ready(WaitResult::AllProducersDone);
        }

        // Register waker in each window covering `range`.
        // (We accept the duplication: every relevant publish will
        // wake us, and we re-check the aggregate predicate.)
        for window_idx in this.consumer.demand.windows_in(&this.range) {
            let window = &this.consumer.demand.windows[window_idx];
            let mut state = window.state.lock();
            match this.predicate {
                WaitPredicate::Zero => {
                    state.waiters_zero.push(cx.waker().clone());
                }
                WaitPredicate::Below(t) => {
                    state.waiters_below.push((t, cx.waker().clone()));
                }
            }
            if !this.registered_windows.contains(&window_idx) {
                this.registered_windows.push(window_idx);
            }
        }

        // Also register in eof_state in case all producers drop
        // before any threshold fires.
        let mut eof_state = this.consumer.demand.eof_state.lock();
        eof_state.waiters.push(cx.waker().clone());

        // Race: producer might have published or signalled EOF
        // *between* our predicate check and our waker registration.
        // Re-check after registering to avoid missed wake.
        if this.predicate_satisfied() {
            return Poll::Ready(WaitResult::PredicateFired);
        }
        if this.consumer.is_eof() {
            return Poll::Ready(WaitResult::AllProducersDone);
        }

        Poll::Pending
    }
}

/// AND `mask`'s slice `[mask_start, mask_end)` into `bits`'s slice
/// `[bits_start, bits_end)`. Both slices must have the same length.
fn and_into(
    bits: &mut BitBuffer,
    bits_start: usize,
    bits_end: usize,
    mask: &Mask,
    mask_start: usize,
    mask_end: usize,
) {
    debug_assert_eq!(bits_end - bits_start, mask_end - mask_start);
    // Convert `bits` to mut (clones if shared), AND mask in
    // bit-by-bit. A SIMD-optimised path lives in vortex_buffer; we
    // use a simple loop here for clarity. If this becomes hot we
    // switch to bulk word-AND.
    let mask_slice = mask.slice(mask_start..mask_end);
    let mut mut_bits =
        match BitBuffer::try_into_mut(std::mem::replace(bits, BitBufferMut::new_set(0).freeze())) {
            Ok(m) => m,
            Err(buf) => {
                // Buffer was shared; copy to mutable.
                BitBufferMut::copy_from(&buf)
            }
        };
    for i in 0..(bits_end - bits_start) {
        if !bit_at(&mask_slice, i) {
            mut_bits.set_to(bits_start + i, false);
        }
    }
    *bits = mut_bits.freeze();
}

fn bit_at(mask: &Mask, i: usize) -> bool {
    // Mask::value or similar — check the API
    mask.to_bit_buffer().value(i)
}

fn slice_true_count(bits: &BitBuffer, start: usize, end: usize) -> usize {
    // Fast path: full buffer
    if start == 0 && end == bits.len() {
        return bits.true_count();
    }
    // Slow path: count bits in the slice manually
    let mut count = 0;
    for i in start..end {
        if bits.value(i) {
            count += 1;
        }
    }
    count
}

/// Typed [`ScanCtx`] slot holding the partition's [`RowDemand`].
///
/// The "right" SIP plumbing pattern: a top-level [`crate::v2::scan::ScanPlan`]
/// installs the demand at execute-start; producer/consumer plans
/// (anywhere lower in the tree) resolve it via [`Self::resolve`] when
/// they need handles.
///
/// If no `ScanPlan` has installed a demand (e.g. unfiltered scan, or
/// a layout running outside a wrapping `ScanPlan`), `resolve` returns
/// an [`RowDemand::empty`] — producers publish into a no-op,
/// consumers see immediate EOF and proceed without demand-driven
/// optimisation. Callers don't need to special-case absence.
#[derive(Default, Debug)]
pub struct RowDemandSlot {
    demand: Option<Arc<RowDemand>>,
}

impl ScanCtxValue for RowDemandSlot {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl RowDemandSlot {
    /// Install a fresh `RowDemand` for the partition. Called by
    /// `ScanPlan::execute` at the top of a partition's execution.
    /// Replaces any previous installation (subsequent re-executes
    /// of the same plan get a fresh demand state).
    pub fn install(ctx: &ScanCtx, demand: Arc<RowDemand>) {
        let mut slot = ctx.get_mut::<RowDemandSlot>();
        slot.demand = Some(demand);
    }

    /// Resolve the partition's `RowDemand`. Returns the installed
    /// instance if a `ScanPlan` has set one, otherwise an empty
    /// no-op demand.
    pub fn resolve(ctx: &ScanCtx) -> Arc<RowDemand> {
        ctx.get::<RowDemandSlot>()
            .demand
            .clone()
            .unwrap_or_else(RowDemand::empty)
    }
}

#[cfg(test)]
mod tests {
    use futures::FutureExt;
    use vortex_buffer::BitBufferMut;
    use vortex_mask::Mask;

    use super::*;

    fn empty_mask(len: usize) -> Mask {
        Mask::from_buffer(BitBufferMut::new_unset(len).freeze())
    }

    #[test]
    fn fresh_demand_starts_full() {
        let demand = RowDemand::new(10_000);
        let consumer = demand.consumer();
        assert_eq!(consumer.cardinality(0..10_000), 10_000);
        assert!(!consumer.is_eof());
    }

    #[test]
    fn publish_zero_mask_drops_cardinality() {
        let demand = RowDemand::new(10_000);
        let producer = demand.producer();
        producer.publish(0..1000, &empty_mask(1000));
        let consumer = demand.consumer();
        // 1000 rows zeroed out, 9000 still demanded.
        assert_eq!(consumer.cardinality_exact(0..10_000), 9000);
        assert_eq!(consumer.cardinality_exact(0..1000), 0);
    }

    #[test]
    fn dropping_last_producer_signals_eof() {
        let demand = RowDemand::new(100);
        let consumer = demand.consumer();
        assert!(!consumer.is_eof());
        let p1 = demand.producer();
        let p2 = demand.producer();
        drop(p1);
        assert!(!consumer.is_eof());
        drop(p2);
        assert!(consumer.is_eof());
    }

    #[test]
    fn wait_for_zero_fires_when_cardinality_drops() {
        let demand = RowDemand::new(100);
        let producer = demand.producer();
        let consumer = demand.consumer();

        let mut fut = consumer.wait_for(0..100, WaitPredicate::Zero).boxed();
        // Not yet — full mask still demanded.
        assert!(fut.as_mut().now_or_never().is_none());

        // Publish a partial zero — still 50 rows demanded; not
        // satisfied yet.
        producer.publish(0..50, &empty_mask(50));
        assert_eq!(consumer.cardinality_exact(0..100), 50);

        // Zero out the rest; future should be ready.
        producer.publish(50..100, &empty_mask(50));
        let result = futures::executor::block_on(fut);
        assert_eq!(result, WaitResult::PredicateFired);
    }

    #[test]
    fn wait_for_below_fires_on_threshold() {
        let demand = RowDemand::new(100);
        let producer = demand.producer();
        let consumer = demand.consumer();

        let fut = consumer.wait_for(0..100, WaitPredicate::Below(40)).boxed();
        // Drop to 60 — still above threshold.
        producer.publish(0..40, &empty_mask(40));
        assert_eq!(consumer.cardinality_exact(0..100), 60);
        // Drop to 30 — fires.
        producer.publish(40..70, &empty_mask(30));
        assert_eq!(consumer.cardinality_exact(0..100), 30);
        let result = futures::executor::block_on(fut);
        assert_eq!(result, WaitResult::PredicateFired);
    }

    #[test]
    fn wait_resolves_on_eof_even_if_predicate_unmet() {
        let demand = RowDemand::new(100);
        let producer = demand.producer();
        let consumer = demand.consumer();
        let fut = consumer.wait_for(0..100, WaitPredicate::Zero).boxed();
        // Don't publish anything — drop the producer.
        drop(producer);
        let result = futures::executor::block_on(fut);
        assert_eq!(result, WaitResult::AllProducersDone);
    }

    #[test]
    fn empty_demand_eofs_immediately() {
        let demand = RowDemand::empty();
        let consumer = demand.consumer();
        assert!(consumer.is_eof());
        assert_eq!(consumer.cardinality(0..0), 0);
    }

    #[test]
    fn idempotent_publish_doesnt_wake_unnecessarily() {
        let demand = RowDemand::new(100);
        let producer = demand.producer();
        producer.publish(0..50, &empty_mask(50));
        let card1 = demand.consumer().cardinality_exact(0..100);
        producer.publish(0..50, &empty_mask(50));
        let card2 = demand.consumer().cardinality_exact(0..100);
        assert_eq!(card1, card2);
    }
}
