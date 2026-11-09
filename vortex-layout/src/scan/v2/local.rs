// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a prepared V2 scan on the task that polls it.
//!
//! The `vortex-file` scan driver spreads a scan's chunks over the runtime's tasks. A caller that scans a small row range, or that has no runtime to spawn on, can instead drive
//! every planner of the scan on one [`Run`] stepped by the stream's own polls: CPU work runs on
//! the polling task and only reads are awaited. Batches come out in row order.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::future;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoService;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::RootId;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::scan::v2::PreparedScan;
use crate::scan::v2::PrunePlanner;

/// The chunks a local scan runs beside each other when the caller does not say.
pub const DEFAULT_LOCAL_CHUNKS: usize = 4;

impl PreparedScan {
    /// Streams the scan's batches in row order, driving every planner of the scan on the task
    /// that polls the stream, with at most `max_chunks` chunks running beside each other.
    ///
    /// Use it for scans of a few chunks, or where no runtime can run tasks. Larger scans run
    /// faster on a driver that spreads chunks over threads.
    pub fn local_stream(
        self: Arc<Self>,
        max_chunks: usize,
    ) -> BoxStream<'static, VortexResult<ArrayRef>> {
        let io = self.io();
        let mut run = Run::new();
        let scope = WorkScope {
            file_ordinal: 0,
            rows: self.rows.clone(),
        };
        let root = run.admit(self.planner(), scope, io.session());
        let mut held = BTreeMap::new();
        held.insert(root, Held::default());

        let state = LocalScan {
            _scan: self,
            run,
            io,
            root,
            queued: VecDeque::new(),
            running: 0,
            max_chunks: max_chunks.max(1),
            held,
            ready: VecDeque::new(),
        };

        stream::try_unfold(state, |mut state| async move {
            let next = state.next().await?;
            Ok(next.map(|array| (array, state)))
        })
        .boxed()
    }
}

impl PreparedScan {
    /// The rows of [`rows`](Self::rows) that the filter's pruning plans keep, over `rows`,
    /// evaluated on the calling task. Only zone maps are read. Every row is kept when the scan
    /// has no filter or the filter has no pruning plan.
    ///
    /// The scan's selection does not limit pruning: combine the result with it as needed.
    pub async fn kept_rows(self: &Arc<Self>) -> VortexResult<Mask> {
        let len = usize::try_from(self.rows.end - self.rows.start)?;
        if self.filter.is_none() {
            return Ok(Mask::new_true(len));
        }

        // Pruning that keeps no row ends the scan without continuing, so the mask stays unset.
        let kept = Arc::new(Mutex::new(None));
        let next = {
            let kept = Arc::clone(&kept);
            Box::new(move |mask: Option<Mask>| -> Box<dyn Planner> {
                *kept.lock() = Some(mask.unwrap_or_else(|| Mask::new_true(len)));
                Box::new(Finished)
            })
        };
        let pruning = PrunePlanner::new(Arc::clone(self), self.rows.clone(), next);

        let io = self.io();
        let mut run = Run::new();
        let scope = WorkScope {
            file_ordinal: 0,
            rows: self.rows.clone(),
        };
        run.admit(Box::new(pruning), scope, io.session());
        loop {
            match run.advance()? {
                Progress::Waiting => {
                    future::poll_fn(|cx| run.poll_completion(cx)).await?;
                }
                Progress::Idle => break,
                Progress::Batch(_) | Progress::Spawn(..) | Progress::RootDone(_) => {}
            }
        }

        let kept = kept.lock().take();
        Ok(kept.unwrap_or_else(|| Mask::new_false(len)))
    }
}

/// A planner with nothing to do, continued into once pruning has run.
struct Finished;

impl IoConsumer for Finished {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for Finished {
    fn state(&self) -> State {
        State::Done
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        Ok(PlannerOutput::Done)
    }
}

/// The batches of one root of the run, held until every root admitted before it is yielded.
#[derive(Default)]
struct Held {
    done: bool,
    batches: Vec<(u64, ArrayRef)>,
}

struct LocalScan {
    /// Keeps the file's shared scan cache alive while the scan runs.
    _scan: Arc<PreparedScan>,
    run: Run,
    io: Arc<dyn IoService>,
    /// The root that plans the scan. Every other root is one chunk it spawned.
    root: RootId,
    /// Chunks spawned and not yet admitted, in the order they were spawned.
    queued: VecDeque<(WorkScope, Box<dyn Planner>)>,
    /// Chunks admitted and not yet done.
    running: usize,
    max_chunks: usize,
    /// Batches by root. Roots are admitted in row order, so their ids order their batches.
    held: BTreeMap<RootId, Held>,
    /// Batches in row order, ready to yield.
    ready: VecDeque<ArrayRef>,
}

impl LocalScan {
    async fn next(&mut self) -> VortexResult<Option<ArrayRef>> {
        loop {
            if let Some(array) = self.ready.pop_front() {
                return Ok(Some(array));
            }

            self.admit();

            match self.run.advance()? {
                Progress::Batch(batch) => {
                    let start = batch.scope.rows.start;
                    self.held
                        .entry(batch.root)
                        .or_default()
                        .batches
                        .push((start, batch.array));
                }
                Progress::Spawn(scope, chunk) => self.queued.push_back((scope, chunk)),
                Progress::RootDone(root) => {
                    if root != self.root {
                        self.running -= 1;
                    }
                    self.held.entry(root).or_default().done = true;
                    self.flush();
                }
                Progress::Waiting => {
                    future::poll_fn(|cx| self.run.poll_completion(cx)).await?;
                }
                Progress::Idle => {
                    if self.queued.is_empty() {
                        self.flush();
                        return Ok(self.ready.pop_front());
                    }
                }
            }
        }
    }

    /// Admits queued chunks while fewer than `max_chunks` run. Each root reads through a session
    /// of its own, because the run clears a root's session once the root is done.
    fn admit(&mut self) {
        while self.running < self.max_chunks {
            let Some((scope, chunk)) = self.queued.pop_front() else {
                return;
            };
            let root = self.run.admit(chunk, scope, self.io.session());
            self.held.insert(root, Held::default());
            self.running += 1;
        }
    }

    /// Moves the batches of every done root that no earlier root holds back to `ready`.
    fn flush(&mut self) {
        while let Some(entry) = self.held.first_entry() {
            if !entry.get().done {
                return;
            }
            let mut batches = entry.remove().batches;
            // Morsels of one chunk finish in whatever order their reads arrive.
            batches.sort_by_key(|(start, _)| *start);
            self.ready
                .extend(batches.into_iter().map(|(_, array)| array));
        }
    }
}
