// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The scheduler-visible IO plane.
//!
//! Nodes *name* reads during planning: [`PlanCx::register`](crate::PlanCx::register) takes an
//! [`IoBatch`] of [`IoUse`]s, each keyed to a whole stored unit, and hands back an [`IoTicket`].
//! Execution may resolve an unissued required ticket through a source-provided non-blocking probe;
//! otherwise it can only clone an already-ready cell or suspend on that exact ticket.
//!
//! A scan owns one [`IoService`], while each affinity-owned morsel has a small [`IoPlane`] that
//! records only the tickets named by that morsel. The service deduplicates raw reads scan-wide and
//! the shared worker pool polls required and speculative futures as independent work items.

use std::cell::RefCell;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;
use std::time::Duration;
use std::time::Instant;

use futures::FutureExt;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::segments::ReadAtNowait;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_utils::aliases::hash_map::HashMap;

use crate::stats::ScanStats;

/// The scan-wide key of one whole stored unit.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IoKey {
    /// A layout segment.
    Segment(SegmentId),
}

/// A ticket handed back by registration, naming the cell the read will land in.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IoTicket(IoKey);

impl IoTicket {
    /// The cell this ticket names.
    pub fn key(&self) -> IoKey {
        self.0
    }
}

/// Scheduler priority attached by the parent operator while planning a read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IoPriority {
    /// Needed to start the next execution phase.
    Required,
    /// Useful lookahead that may finish while required CPU work runs.
    Speculative,
}

/// Identifies the node that emitted a use, so the scheduler can attribute and cancel it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProducerId(pub u32);

/// One named read.
#[derive(Clone, Debug)]
pub struct IoUse {
    /// The whole stored unit this use covers.
    pub key: IoKey,
    /// The rows of the stored unit, frozen at emission.
    pub extent: Range<u64>,
    /// The inverse image of `extent` in root coordinates, stamped at emission. The scheduler
    /// reads demand verdicts over this range without ever seeing an offset map.
    pub source_range: Range<u64>,
    /// The node that emitted this use.
    pub producer: ProducerId,
    /// The estimated size of the read, for admission accounting.
    pub estimated_bytes: usize,
}

/// A batch of uses emitted by one planning step.
#[derive(Clone, Debug, Default)]
pub struct IoBatch {
    uses: Vec<IoUse>,
}

impl IoBatch {
    /// An empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a use to the batch.
    pub fn push(&mut self, r#use: IoUse) {
        self.uses.push(r#use);
    }

    /// The uses in this batch.
    pub fn uses(&self) -> &[IoUse] {
        &self.uses
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.uses.is_empty()
    }
}

impl FromIterator<IoUse> for IoBatch {
    fn from_iter<T: IntoIterator<Item = IoUse>>(iter: T) -> Self {
        Self {
            uses: iter.into_iter().collect(),
        }
    }
}

enum CellState {
    Unissued,
    Pending {
        future: SegmentFuture,
        wait_started: Option<Instant>,
    },
    Ready(BufferHandle),
}

struct IoCell {
    key: IoKey,
    state: Mutex<CellState>,
    waiters: Mutex<Vec<Waker>>,
    required: AtomicBool,
    submitted: AtomicBool,
}

/// Scan-wide registry of raw segment requests.
///
/// A segment future is created once per scan. Morsel-local planes hold references to these cells,
/// so two overlapping morsels share both an in-flight request and its completed bytes.
pub(crate) struct IoService {
    source: Arc<dyn SegmentSource>,
    cells: Mutex<HashMap<IoKey, Arc<IoCell>>>,
    nowait_support: AtomicU8,
}

impl IoService {
    pub(crate) fn new(source: Arc<dyn SegmentSource>) -> Arc<Self> {
        Arc::new(Self {
            source,
            cells: Mutex::new(HashMap::default()),
            nowait_support: AtomicU8::new(0),
        })
    }

    fn register(&self, key: IoKey, priority: IoPriority) -> (Arc<IoCell>, bool) {
        let mut cells = self.cells.lock();
        if let Some(cell) = cells.get(&key) {
            if priority == IoPriority::Required {
                cell.required.store(true, Ordering::Release);
            }
            return (Arc::clone(cell), false);
        }

        let cell = Arc::new(IoCell {
            key,
            state: Mutex::new(CellState::Unissued),
            waiters: Mutex::new(Vec::new()),
            required: AtomicBool::new(priority == IoPriority::Required),
            submitted: AtomicBool::new(false),
        });
        cells.insert(key, Arc::clone(&cell));
        (cell, true)
    }

    pub(crate) fn issue(&self, read: &IoRead) {
        let mut state = read.cell.state.lock();
        if !matches!(*state, CellState::Unissued) {
            return;
        }
        let future = match read.key() {
            IoKey::Segment(id) => self.source.request(id),
        };
        *state = CellState::Pending {
            future,
            wait_started: None,
        };
    }

    pub(crate) fn nowait_unsupported(&self) -> bool {
        self.nowait_support.load(Ordering::Acquire) == 2
    }

    pub(crate) fn read(&self, ticket: IoTicket) -> Option<IoRead> {
        self.cells
            .lock()
            .get(&ticket.key())
            .cloned()
            .map(|cell| IoRead { cell })
    }
}

/// One registered segment future that the shared scheduler can poll as an IO work item.
#[derive(Clone)]
pub(crate) struct IoRead {
    cell: Arc<IoCell>,
}

impl IoRead {
    pub(crate) fn key(&self) -> IoKey {
        self.cell.key
    }

    pub(crate) fn priority(&self) -> IoPriority {
        if self.cell.required.load(Ordering::Acquire) {
            IoPriority::Required
        } else {
            IoPriority::Speculative
        }
    }

    pub(crate) fn promote(&self) {
        self.cell.required.store(true, Ordering::Release);
    }

    /// Subscribe an affinity-owned continuation to this exact cell.
    ///
    /// Returns `true` when the continuation was parked. The state lock closes the completion race:
    /// a completion either drains this waker or is observed here before insertion.
    pub(crate) fn park(&self, waker: Waker) -> bool {
        let state = self.cell.state.lock();
        if matches!(*state, CellState::Ready(_)) {
            return false;
        }
        self.cell.waiters.lock().push(waker);
        true
    }
}

/// Outcome of polling one scheduler-owned segment future.
pub(crate) enum IoReadPoll {
    /// The future retained the waker and will requeue this IO work item.
    Pending,
    /// This poll completed the read.
    Ready {
        /// Bytes in the returned segment.
        bytes: usize,
        /// Time since this future first returned `Pending`.
        wait_time: Duration,
    },
    /// A stale wake observed a read another worker had already completed.
    AlreadyReady,
}

impl IoRead {
    /// Poll this one future without blocking the worker.
    pub(crate) fn poll(&self, waker: &Waker) -> VortexResult<IoReadPoll> {
        let mut state = self.cell.state.lock();
        let CellState::Pending {
            future,
            wait_started,
        } = &mut *state
        else {
            return match &*state {
                CellState::Ready(_) => Ok(IoReadPoll::AlreadyReady),
                CellState::Unissued => Err(vortex_err!("IO cell was polled before submission")),
                CellState::Pending { .. } => unreachable!(),
            };
        };

        let mut cx = Context::from_waker(waker);
        loop {
            match future.poll_unpin(&mut cx) {
                Poll::Ready(result) => {
                    let handle = result?;
                    if handle.is_on_device() {
                        let copy = handle.try_into_host()?;
                        *future = async move { copy.await.map(BufferHandle::new_host) }.boxed();
                        continue;
                    }
                    let bytes = handle.len();
                    let wait_time = wait_started
                        .take()
                        .map_or(Duration::ZERO, |started| started.elapsed());
                    *state = CellState::Ready(handle);
                    drop(state);
                    for waiter in std::mem::take(&mut *self.cell.waiters.lock()) {
                        waiter.wake();
                    }
                    return Ok(IoReadPoll::Ready { bytes, wait_time });
                }
                Poll::Pending => {
                    wait_started.get_or_insert_with(Instant::now);
                    return Ok(IoReadPoll::Pending);
                }
            }
        }
    }
}

/// The ticket view owned by one affinity-local morsel continuation.
///
/// The keyed map uses interior mutability because only planning and execution touch its shape.
/// Individual cell futures live in the scan-wide service and are synchronized because any worker
/// in the pool may poll them.
pub struct IoPlane {
    service: Arc<IoService>,
    cells: RefCell<HashMap<IoKey, Arc<IoCell>>>,
    unsubmitted: RefCell<Vec<Arc<IoCell>>>,
}

impl IoPlane {
    /// Create a morsel-local view over the scan's shared IO service.
    pub(crate) fn new(service: Arc<IoService>) -> Self {
        Self {
            service,
            cells: RefCell::new(HashMap::default()),
            unsubmitted: RefCell::new(Vec::new()),
        }
    }

    /// Register a batch of uses, issuing any cell that does not already exist.
    pub(crate) fn register(
        &self,
        batch: IoBatch,
        priority: IoPriority,
        stats: &mut ScanStats,
    ) -> VortexResult<Vec<IoTicket>> {
        let mut cells = self.cells.borrow_mut();
        let mut tickets = Vec::with_capacity(batch.uses().len());
        for r#use in batch.uses() {
            if !cells.contains_key(&r#use.key) {
                stats.io_registered += 1;
                let (cell, created) = self.service.register(r#use.key, priority);
                if created {
                    stats.io_requests += 1;
                } else {
                    stats.io_cell_hits += 1;
                }
                self.unsubmitted.borrow_mut().push(Arc::clone(&cell));
                cells.insert(r#use.key, cell);
            } else {
                stats.io_cell_hits += 1;
                if priority == IoPriority::Required {
                    cells[&r#use.key].required.store(true, Ordering::Release);
                }
            }
            tickets.push(IoTicket(r#use.key));
        }
        Ok(tickets)
    }

    /// Take newly registered reads for submission to the shared work queue.
    ///
    /// Each future is returned at most once even when planning spans several quanta. Duplicate
    /// logical uses retain one keyed cell and cannot submit duplicate reads.
    pub(crate) fn take_reads(&self) -> Vec<IoRead> {
        std::mem::take(&mut *self.unsubmitted.borrow_mut())
            .into_iter()
            .filter(|cell| {
                !matches!(*cell.state.lock(), CellState::Ready(_))
                    && !cell.submitted.swap(true, Ordering::AcqRel)
            })
            .map(|cell| IoRead { cell })
            .collect()
    }

    /// Resolve a ticket inline when the source can prove the bytes are immediately available.
    ///
    /// The cell is retained so duplicate uses inside this morsel share the same handle.
    pub(crate) fn ready(
        &self,
        ticket: IoTicket,
        stats: &mut ScanStats,
    ) -> VortexResult<Option<BufferHandle>> {
        let cell = self
            .cells
            .borrow()
            .get(&ticket.key())
            .cloned()
            .ok_or_else(|| vortex_err!("IO ticket was accessed without registration"))?;
        let mut state = cell.state.lock();
        match &*state {
            CellState::Ready(handle) => return Ok(Some(handle.clone())),
            CellState::Pending { .. } => return Ok(None),
            CellState::Unissued => {}
        }

        let IoKey::Segment(segment) = cell.key;
        if self.service.nowait_unsupported() {
            let future = self.service.source.request(segment);
            *state = CellState::Pending {
                future,
                wait_started: None,
            };
            return Ok(None);
        }
        stats.nowait_attempts += 1;
        match self.service.source.request_nowait(segment)? {
            ReadAtNowait::Ready(handle) => {
                self.service.nowait_support.store(1, Ordering::Release);
                stats.nowait_hits += 1;
                stats.io_bytes += handle.len() as u64;
                *state = CellState::Ready(handle.clone());
                drop(state);
                for waiter in std::mem::take(&mut *cell.waiters.lock()) {
                    waiter.wake();
                }
                Ok(Some(handle))
            }
            ReadAtNowait::WouldBlock => {
                self.service.nowait_support.store(1, Ordering::Release);
                stats.nowait_misses += 1;
                let future = self.service.source.request(segment);
                *state = CellState::Pending {
                    future,
                    wait_started: None,
                };
                Ok(None)
            }
            ReadAtNowait::Unsupported => {
                self.service.nowait_support.store(2, Ordering::Release);
                stats.nowait_unsupported += 1;
                let future = self.service.source.request(segment);
                *state = CellState::Pending {
                    future,
                    wait_started: None,
                };
                Ok(None)
            }
        }
    }

    /// Drop every cell. Called between morsel batches to bound retained bytes.
    pub fn clear(&self) {
        self.cells.borrow_mut().clear();
        self.unsubmitted.borrow_mut().clear();
    }

    /// Drop the cell behind a key, if it is resolved.
    pub fn release(&self, key: IoKey) {
        self.cells.borrow_mut().remove(&key);
    }
}

/// Error helper for a ticket consumed without ever having been planned.
pub fn unplanned_ticket(producer: ProducerId) -> vortex_error::VortexError {
    vortex_err!(
        "node {} waited on a ticket its planning stream never emitted",
        producer.0
    )
}
