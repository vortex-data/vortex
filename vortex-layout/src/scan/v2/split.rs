// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::future;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_io::request::IoService;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoSource;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::WorkScope;

/// One filter split of a scan, ready to be admitted to a planning run.
///
/// The split's planners announce, prune, filter, and project the rows of `scope`, and its morsels
/// emit one batch per projection split with selected rows, in whatever order their reads arrive.
pub struct SplitPlan {
    /// The rows of the scan's root row domain the split covers.
    pub scope: WorkScope,
    /// The split's first planner.
    pub root: Box<dyn Planner>,
}

impl SplitPlan {
    /// Runs the split on a planning run of its own, reading through a session of `io`, and
    /// returns its batches in row order.
    ///
    /// The run advances inside this future, on whichever thread polls it, and the future awaits
    /// its reads between steps. Nothing is handed to another thread.
    pub(super) fn run(
        mut self,
        io: Arc<dyn IoService>,
    ) -> impl Future<Output = VortexResult<Vec<ArrayRef>>> + Send + 'static {
        static EARLY_ANNOUNCE: LazyLock<bool> = LazyLock::new(|| {
            !env::var("VORTEX_SCAN_EARLY_ANNOUNCE").is_ok_and(|value| value == "0")
        });
        // Register neighboring ranges before any split starts fetching, so the driver can
        // coalesce them. Keep the session alive for fetches and cancellation. Setting
        // VORTEX_SCAN_EARLY_ANNOUNCE=0 restores late announcements for comparisons.
        let prepared: VortexResult<Option<Arc<dyn IoSource>>> = (|| {
            if !*EARLY_ANNOUNCE {
                return Ok(None);
            }
            let source = io.session();
            match self.root.compute()? {
                PlannerOutput::NeedsIO(batch) => {
                    vortex_ensure!(
                        batch.iter().all(|request| request.intent == IoIntent::Announce),
                        "early announcement encountered an active read"
                    );
                    source.submit(IoOwnerId(0), batch)?;
                }
                PlannerOutput::Planner(scope, root) => {
                    self.scope = scope;
                    self.root = root;
                }
                output => vortex_bail!("unexpected initial announcement output: {output:?}"),
            }
            Ok(Some(source))
        })();
        async move {
            let source = prepared?.unwrap_or_else(|| io.session());
            let mut run = Run::new();
            run.admit(self.root, self.scope, source);
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
}
