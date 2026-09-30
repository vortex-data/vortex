// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_io::request::IoService;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::next::PendingPlanner;
use vortex_scan::planning::planner::WorkScope;

/// One filter split of a scan, ready to be admitted to a planning run.
///
/// The split's planners announce, prune, filter, and project the rows of `scope`, and its morsels
/// emit one batch per projection split with selected rows, in whatever order their reads arrive.
pub struct SplitPlan {
    /// The rows of the scan's root row domain the split covers.
    pub scope: WorkScope,
    /// The split's first planner.
    pub root: Box<dyn PendingPlanner>,
}

impl SplitPlan {
    /// Runs the split on a planning run of its own, reading through a session of `io`, and
    /// returns its batches in row order.
    ///
    /// The run advances inside this future, on whichever thread polls it, and the future awaits
    /// its reads between steps. Nothing is handed to another thread.
    pub(super) async fn run(self, io: Arc<dyn IoService>) -> VortexResult<Vec<ArrayRef>> {
        let mut run = Run::new();
        run.admit(self.root, self.scope, io.session());
        let mut batches = Vec::new();
        loop {
            match run.advance()? {
                Progress::Batch(batch) => batches.push(batch),
                Progress::RootDone(_) | Progress::Idle => break,
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
        // Morsels finish in whatever order their reads arrive.
        batches.sort_by_key(|batch| batch.scope.rows.start);
        Ok(batches.into_iter().map(|batch| batch.array).collect())
    }
}
