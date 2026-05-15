// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// All `u64 ↔ usize` casts in this module are bounded by the
// partition's row count, which is itself a usize at the engine
// level. On 64-bit targets the casts are exact; on 32-bit they'd
// already have been unsafe upstream. Keeping `as` casts for
// readability since the conversions appear in tight inner loops.
#![allow(clippy::cast_possible_truncation, clippy::cast_lossless)]

//! [`RowDemand`] — partition-local SIP for tracking which rows still
//! need work.
//!
//! ## Model
//!
//! A `RowDemand` is a clone-cheap, coordinate-aware view onto shared
//! per-scan demand state. Bits start at all-1 (every row needed) and
//! only go `1 → 0` (rows newly known not to be needed). Anyone can
//! publish reductions; the result is the AND of all contributions.
//!
//! Publishers don't need to coordinate — there's no hard-coded
//! ordering. The AND-of-monotone semantic guarantees commutativity:
//! whoever publishes first wins for those rows, and others contribute
//! when they finish. Reads see the running intersection.
//!
//! ## Coordinate translation
//!
//! `RowDemand` is passed to [`crate::v2::plan::LayoutPlan::execute`]
//! alongside `row_range`, in the same coordinate system. A layout
//! that delegates to children at different row offsets (e.g.
//! `ChunkedPlan`) calls [`RowDemand::scope`] to produce a child-local
//! view. Layouts that don't change row domain just pass it through.
//!
//! ## EOF and producer tracking
//!
//! Anything that intends to publish should hold a [`ProducerGuard`]
//! for the duration. EOF fires when the last guard drops. Use
//! [`RowDemand::spawn_producer`] for the common case (a side-task
//! spawned at scan start that publishes for its lifetime), or
//! [`RowDemand::producer_guard`] inside an async stream that wants
//! to publish without being a separately-spawned task.
//!
//! ## Threshold-based wakers
//!
//! Readers register a [`WaitPredicate`] over a row range:
//!
//! - [`WaitPredicate::Zero`] — wake when cardinality across `range`
//!   drops to zero. Skip downstream work when the filter rejected
//!   every row.
//! - [`WaitPredicate::Below(threshold)`] — wake when cardinality
//!   drops below `threshold`. A reader about to eagerly submit I/O
//!   reconsiders if too few rows survive to be worth fetching.
//!
//! Wakes fire only when the relevant threshold is crossed, plus once
//! at EOF. Unrelated publishes don't disturb readers.
//!
//! ## Per-window state
//!
//! The row space is split into fixed-size windows
//! ([`RowDemand::WINDOW_ROWS`] = 4 K rows). Per-window state holds
//! the demand bits, a cached cardinality (so reads don't have to
//! popcount), and per-threshold waiter lists. Locks are per-window
//! so unrelated publishes don't contend.

use std::future::Future;
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

use parking_lot::Mutex;
use smallvec::SmallVec;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_mask::Mask;

/// A row range. Half-open: `[start, end)`.
pub type RowRange = Range<u64>;

/// Per-window row count. Smaller = finer waker granularity but more
/// metadata; larger = coarser. 4096 fits common chunk granularities
/// and keeps each window's bit buffer at 512 B.
const WINDOW_ROWS: u64 = 4096;

/// Shared per-scan demand state. Hidden behind [`RowDemand`].
#[derive(Debug)]
struct RowDemandState {
    /// One per [`WINDOW_ROWS`] slice of the partition's row space.
    /// The last window may be partial.
    windows: Vec<Window>,
    /// Number of live producer guards. Decremented on each
    /// `ProducerGuard::drop`; reaching 0 fires the EOF wakers.
    active_producers: AtomicUsize,
    eof: AtomicBool,
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

/// What a reader is waiting on.
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
    /// predicate may or may not have fired — readers should inspect
    /// current cardinality to decide.
    AllProducersDone,
}

/// Coordinate-aware view onto a partition's [`RowDemandState`].
///
/// Clone-cheap (an `Arc` plus a row range). Pass by reference through
/// `LayoutPlan::execute`; clone when handing into spawned tasks.
///
/// The view is in the same coordinate system as the surrounding
/// `row_range` parameter. Child layouts at different row offsets call
/// [`Self::scope`] to translate. Subtrees in unrelated row spaces
/// (e.g. a Zoned plan's stats child, a Dict plan's values child)
/// should pass [`Self::detached`] instead.
#[derive(Clone, Debug)]
pub struct RowDemand {
    state: Arc<RowDemandState>,
    /// Range of `state` row coordinates this view exposes. Local
    /// coord 0 maps to `state[scope.start]`. `total_rows()` is
    /// `scope.len()`.
    scope: RowRange,
}

impl RowDemand {
    /// Window size (in rows) used by the per-window mutex sharding.
    pub const WINDOW_ROWS: u64 = WINDOW_ROWS;

    /// Create a fresh `RowDemand` covering `total_rows` rows. All
    /// rows start as "demanded" (bits=1). No producers active until
    /// [`Self::producer_guard`] is called at least once.
    pub fn new(total_rows: u64) -> Self {
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
        Self {
            state: Arc::new(RowDemandState {
                windows,
                active_producers: AtomicUsize::new(0),
                eof: AtomicBool::new(false),
                eof_state: Mutex::new(EofState::default()),
            }),
            scope: 0..total_rows,
        }
    }

    /// A detached `RowDemand` covering `total_rows` rows that is
    /// already at EOF and has no waiters.
    ///
    /// Use this for subtrees in unrelated row spaces (a stats read, a
    /// dict-values read) — publishes are no-ops, reads return full
    /// cardinality, and `wait_for` resolves immediately as
    /// `AllProducersDone`. Callers don't need to special-case absence.
    pub fn detached(total_rows: u64) -> Self {
        let demand = Self::new(total_rows);
        // Force EOF immediately — no producers will ever publish here.
        // The Arc is unique at this point (we just allocated it), but
        // we use the atomic store rather than Arc::get_mut to keep the
        // method panic-free under arbitrary call patterns.
        demand.state.eof.store(true, Ordering::Release);
        demand
    }

    /// Total row count this view exposes (in local coords).
    pub fn total_rows(&self) -> u64 {
        self.scope.end - self.scope.start
    }

    /// View this demand restricted to a sub-range, in local coords.
    /// The returned view's local coord 0 corresponds to this view's
    /// `sub_range.start`. Cheap (clones an `Arc`, computes a range).
    pub fn scope(&self, sub_range: RowRange) -> Self {
        let global_start = self.scope.start + sub_range.start;
        let global_end = self.scope.start + sub_range.end;
        debug_assert!(
            global_end <= self.scope.end,
            "RowDemand::scope: sub_range {sub_range:?} exceeds parent total {}",
            self.total_rows()
        );
        Self {
            state: Arc::clone(&self.state),
            scope: global_start..global_end,
        }
    }

    /// Translate a local range to a global one, clamped to this scope.
    fn to_global(&self, local: &RowRange) -> RowRange {
        let start = (self.scope.start + local.start).min(self.scope.end);
        let end = (self.scope.start + local.end).min(self.scope.end);
        start..end
    }

    /// AND `mask` into the demand bits covering `range` (local coords).
    /// `mask` must have `range.len()` rows. Wakes any reader whose
    /// threshold is crossed by this update.
    pub fn publish(&self, range: RowRange, mask: &Mask) {
        let global_range = self.to_global(&range);
        if global_range.start >= global_range.end {
            return;
        }
        let mask_offset = (global_range.start - (self.scope.start + range.start)) as usize;
        publish_global(&self.state, global_range, mask, mask_offset);
    }

    /// Sum of cached cardinality across windows touching `range`.
    /// Cheap (atomic loads). Upper bound — see [`Self::cardinality_exact`]
    /// for an exact count.
    pub fn cardinality(&self, range: RowRange) -> u64 {
        let global = self.to_global(&range);
        let mut total = 0u64;
        for window_idx in windows_in(&self.state, &global) {
            total += self.state.windows[window_idx]
                .cardinality
                .load(Ordering::Acquire);
        }
        total
    }

    /// Exact cardinality over `range` (handles partial-window edges).
    /// Locks each touched window briefly.
    pub fn cardinality_exact(&self, range: RowRange) -> u64 {
        let global = self.to_global(&range);
        let mut total = 0u64;
        for window_idx in windows_in(&self.state, &global) {
            let window = &self.state.windows[window_idx];
            let window_start = (window_idx as u64) * WINDOW_ROWS;
            let window_end = window_start + window.cardinality_capacity_u64();
            let isect_start = global.start.max(window_start);
            let isect_end = global.end.min(window_end);
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
        self.state.eof.load(Ordering::Acquire)
    }

    /// Wait until `predicate` is satisfied for `range`, or until all
    /// producers signal done. Returns whichever happened first.
    pub fn wait_for(&self, range: RowRange, predicate: WaitPredicate) -> WaitFuture {
        WaitFuture {
            state: Arc::clone(&self.state),
            global_range: self.to_global(&range),
            predicate,
        }
    }

    /// Acquire a producer guard. While alive, demand reports
    /// `!is_eof()`. Drop the guard when done publishing.
    pub fn producer_guard(&self) -> ProducerGuard {
        self.state.active_producers.fetch_add(1, Ordering::AcqRel);
        ProducerGuard {
            state: Arc::clone(&self.state),
        }
    }

    /// Spawn a future that holds a producer guard for its lifetime.
    /// The future receives a clone of this `RowDemand` to publish into.
    pub fn spawn_producer<F, Fut>(&self, handle: &Handle, f: F)
    where
        F: FnOnce(RowDemand) -> Fut + Send + 'static,
        Fut: Future<Output = VortexResult<()>> + Send + 'static,
    {
        let demand = self.clone();
        let guard = self.producer_guard();
        handle
            .spawn(async move {
                let _guard = guard;
                // Errors from producer tasks are intentionally
                // swallowed here: a failing falsifier must not bring
                // down the scan, it just stops contributing demand
                // reductions.
                drop(f(demand).await);
            })
            .detach();
    }
}

/// RAII guard contributing to producer-active count. EOF fires when
/// the last guard drops.
pub struct ProducerGuard {
    state: Arc<RowDemandState>,
}

impl Drop for ProducerGuard {
    fn drop(&mut self) {
        let prev = self.state.active_producers.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.state.eof.store(true, Ordering::Release);
            // Wake all per-window waiters — they can decide based on
            // final state.
            for window in &self.state.windows {
                let mut state = window.state.lock();
                for waker in state.waiters_zero.drain(..) {
                    waker.wake();
                }
                for (_, waker) in state.waiters_below.drain(..) {
                    waker.wake();
                }
            }
            let mut eof_state = self.state.eof_state.lock();
            for waker in eof_state.waiters.drain(..) {
                waker.wake();
            }
        }
    }
}

impl Window {
    fn cardinality_capacity_u64(&self) -> u64 {
        self.state.lock().bits.len() as u64
    }
}

/// Future returned by [`RowDemand::wait_for`].
pub struct WaitFuture {
    state: Arc<RowDemandState>,
    global_range: RowRange,
    predicate: WaitPredicate,
}

impl WaitFuture {
    fn current_cardinality(&self) -> u64 {
        let mut total = 0u64;
        for window_idx in windows_in(&self.state, &self.global_range) {
            total += self.state.windows[window_idx]
                .cardinality
                .load(Ordering::Acquire);
        }
        total
    }

    fn predicate_satisfied(&self) -> bool {
        let card = self.current_cardinality();
        match self.predicate {
            WaitPredicate::Zero => card == 0,
            WaitPredicate::Below(t) => card < t,
        }
    }
}

impl Future for WaitFuture {
    type Output = WaitResult;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.predicate_satisfied() {
            return Poll::Ready(WaitResult::PredicateFired);
        }
        if self.state.eof.load(Ordering::Acquire) {
            return Poll::Ready(WaitResult::AllProducersDone);
        }

        // Register waker in each window touching `global_range`.
        for window_idx in windows_in(&self.state, &self.global_range) {
            let window = &self.state.windows[window_idx];
            let mut state = window.state.lock();
            match self.predicate {
                WaitPredicate::Zero => state.waiters_zero.push(cx.waker().clone()),
                WaitPredicate::Below(t) => state.waiters_below.push((t, cx.waker().clone())),
            }
        }

        // Also register for EOF in case all producers drop before any
        // threshold fires.
        self.state.eof_state.lock().waiters.push(cx.waker().clone());

        // Race: producer might have published or signalled EOF
        // *between* our predicate check and waker registration.
        // Re-check after registering to avoid missed wake.
        if self.predicate_satisfied() {
            return Poll::Ready(WaitResult::PredicateFired);
        }
        if self.state.eof.load(Ordering::Acquire) {
            return Poll::Ready(WaitResult::AllProducersDone);
        }

        Poll::Pending
    }
}

/// Range of window indices touching `range` (in global coords).
fn windows_in(state: &RowDemandState, range: &RowRange) -> Range<usize> {
    if range.start >= range.end || state.windows.is_empty() {
        return 0..0;
    }
    let start = (range.start / WINDOW_ROWS) as usize;
    let end = ((range.end - 1) / WINDOW_ROWS) as usize + 1;
    start..end.min(state.windows.len())
}

/// Apply a publish in global coords. `mask_offset` is the index in
/// `mask` corresponding to `global_range.start`.
fn publish_global(state: &RowDemandState, global_range: RowRange, mask: &Mask, mask_offset: usize) {
    for window_idx in windows_in(state, &global_range) {
        let window = &state.windows[window_idx];
        let window_start = (window_idx as u64) * WINDOW_ROWS;
        let window_end = window_start + window.cardinality_capacity_u64();
        let isect_start = global_range.start.max(window_start);
        let isect_end = global_range.end.min(window_end);
        if isect_start >= isect_end {
            continue;
        }
        let local_start = (isect_start - window_start) as usize;
        let local_end = (isect_end - window_start) as usize;
        let mask_start = mask_offset + (isect_start - global_range.start) as usize;
        let mask_end = mask_offset + (isect_end - global_range.start) as usize;

        let mut wstate = window.state.lock();
        and_into(
            &mut wstate.bits,
            local_start,
            local_end,
            mask,
            mask_start,
            mask_end,
        );

        let new_card = wstate.bits.true_count() as u64;
        let prev_card = window.cardinality.swap(new_card, Ordering::AcqRel);
        if new_card == prev_card {
            continue;
        }

        if new_card == 0 && !wstate.waiters_zero.is_empty() {
            for waker in wstate.waiters_zero.drain(..) {
                waker.wake();
            }
        }
        if !wstate.waiters_below.is_empty() {
            let mut i = 0;
            while i < wstate.waiters_below.len() {
                if new_card < wstate.waiters_below[i].0 {
                    let (_, waker) = wstate.waiters_below.swap_remove(i);
                    waker.wake();
                } else {
                    i += 1;
                }
            }
        }
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
    let mask_slice = mask.slice(mask_start..mask_end);
    let mut mut_bits =
        match BitBuffer::try_into_mut(std::mem::replace(bits, BitBufferMut::new_set(0).freeze())) {
            Ok(m) => m,
            Err(buf) => BitBufferMut::copy_from(&buf),
        };
    for i in 0..(bits_end - bits_start) {
        if !bit_at(&mask_slice, i) {
            mut_bits.set_to(bits_start + i, false);
        }
    }
    *bits = mut_bits.freeze();
}

fn bit_at(mask: &Mask, i: usize) -> bool {
    mask.to_bit_buffer().value(i)
}

fn slice_true_count(bits: &BitBuffer, start: usize, end: usize) -> usize {
    if start == 0 && end == bits.len() {
        return bits.true_count();
    }
    let mut count = 0;
    for i in start..end {
        if bits.value(i) {
            count += 1;
        }
    }
    count
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
        assert_eq!(demand.cardinality(0..10_000), 10_000);
        assert!(!demand.is_eof());
    }

    #[test]
    fn publish_zero_mask_drops_cardinality() {
        let demand = RowDemand::new(10_000);
        let _guard = demand.producer_guard();
        demand.publish(0..1000, &empty_mask(1000));
        assert_eq!(demand.cardinality_exact(0..10_000), 9000);
        assert_eq!(demand.cardinality_exact(0..1000), 0);
    }

    #[test]
    fn dropping_last_guard_signals_eof() {
        let demand = RowDemand::new(100);
        assert!(!demand.is_eof());
        let g1 = demand.producer_guard();
        let g2 = demand.producer_guard();
        drop(g1);
        assert!(!demand.is_eof());
        drop(g2);
        assert!(demand.is_eof());
    }

    #[test]
    fn wait_for_zero_fires_when_cardinality_drops() {
        let demand = RowDemand::new(100);
        let _guard = demand.producer_guard();

        let mut fut = demand.wait_for(0..100, WaitPredicate::Zero).boxed();
        assert!(fut.as_mut().now_or_never().is_none());

        demand.publish(0..50, &empty_mask(50));
        assert_eq!(demand.cardinality_exact(0..100), 50);

        demand.publish(50..100, &empty_mask(50));
        let result = futures::executor::block_on(fut);
        assert_eq!(result, WaitResult::PredicateFired);
    }

    #[test]
    fn wait_for_below_fires_on_threshold() {
        let demand = RowDemand::new(100);
        let _guard = demand.producer_guard();

        let fut = demand.wait_for(0..100, WaitPredicate::Below(40)).boxed();
        demand.publish(0..40, &empty_mask(40));
        assert_eq!(demand.cardinality_exact(0..100), 60);
        demand.publish(40..70, &empty_mask(30));
        assert_eq!(demand.cardinality_exact(0..100), 30);
        let result = futures::executor::block_on(fut);
        assert_eq!(result, WaitResult::PredicateFired);
    }

    #[test]
    fn wait_resolves_on_eof_even_if_predicate_unmet() {
        let demand = RowDemand::new(100);
        let guard = demand.producer_guard();
        let fut = demand.wait_for(0..100, WaitPredicate::Zero).boxed();
        drop(guard);
        let result = futures::executor::block_on(fut);
        assert_eq!(result, WaitResult::AllProducersDone);
    }

    #[test]
    fn detached_demand_is_eof_immediately() {
        let demand = RowDemand::detached(100);
        assert!(demand.is_eof());
        assert_eq!(demand.cardinality(0..100), 100);
    }

    #[test]
    fn idempotent_publish_doesnt_change_cardinality() {
        let demand = RowDemand::new(100);
        let _guard = demand.producer_guard();
        demand.publish(0..50, &empty_mask(50));
        let card1 = demand.cardinality_exact(0..100);
        demand.publish(0..50, &empty_mask(50));
        let card2 = demand.cardinality_exact(0..100);
        assert_eq!(card1, card2);
    }

    #[test]
    fn scope_translates_publish_and_read() {
        let demand = RowDemand::new(1000);
        let _guard = demand.producer_guard();

        // Scope to rows [200, 700) — local 0..500 maps to global
        // 200..700.
        let scoped = demand.scope(200..700);
        assert_eq!(scoped.total_rows(), 500);

        // Publish zeros over local [0, 200) — should clear global
        // [200, 400).
        scoped.publish(0..200, &empty_mask(200));
        assert_eq!(demand.cardinality_exact(0..200), 200);
        assert_eq!(demand.cardinality_exact(200..400), 0);
        assert_eq!(demand.cardinality_exact(400..1000), 600);

        // Read in scoped coords matches.
        assert_eq!(scoped.cardinality_exact(0..200), 0);
        assert_eq!(scoped.cardinality_exact(200..500), 300);
    }

    #[test]
    fn scope_of_scope_composes() {
        let demand = RowDemand::new(1000);
        let _guard = demand.producer_guard();

        let outer = demand.scope(100..900); // global 100..900
        let inner = outer.scope(50..150); // local 50..150 of outer = global 150..250
        assert_eq!(inner.total_rows(), 100);

        inner.publish(0..50, &empty_mask(50)); // global 150..200
        assert_eq!(demand.cardinality_exact(150..200), 0);
        assert_eq!(demand.cardinality_exact(200..250), 50);
    }
}
