// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a prepared V2 scan on the task that polls it.
//!
//! The `vortex-file` scan driver spreads a scan's chunks over the runtime's tasks. A caller that
//! scans a small row range, or that has no runtime to spawn on, can instead drive every planner
//! of the scan on one [`Run`] stepped by the stream's own polls: CPU work runs on the polling
//! task and only reads are awaited. Batches come out in row order. A local scan is a
//! [`ScanScheduler`] with one scan.

use std::future;
use std::sync::Arc;

use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;

use crate::scan::v2::PreparedScan;
use crate::scan::v2::PrunePlanner;
use crate::scan::v2::ScanScheduler;

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
        ScanScheduler::new()
            .with_max_chunks(max_chunks)
            .scan(self, 0)
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
