// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan for DataFusion.
//!
//! A planner says what it needs: compute, or reads. The opener composes the planners over a
//! [`PreparedScan`] and hands the first to [`run_scan`], which runs it and everything it emits
//! on a [`ParallelDriver`]. Each worker of the driver runs on a task of its own and maps the
//! batches it produces to Arrow, so the DataFusion thread polling the stream does no scan work.

use std::sync::Arc;
use std::task::Poll;

use arrow_array::RecordBatch;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex::array::ArrayRef;
use vortex::error::VortexResult;
use vortex::io::runtime::Handle;
use vortex::layout::scan::v2::PreparedScan;
use vortex::scan::planning::drive::ParallelDriver;
use vortex::scan::planning::drive::PlanRoot;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::WorkScope;

/// Runs one partition's V2 scan on DataFusion's runtime: `root`, the first planner composed over
/// `scan`, and everything it emits, on `workers` workers, streaming their batches mapped by
/// `map`. With `ordered`, batches are yielded in row order, which holds them until the scan is
/// done.
pub(crate) fn run_scan(
    handle: Handle,
    workers: usize,
    ordered: bool,
    scan: Arc<PreparedScan>,
    root: Box<dyn Planner>,
    map: impl Fn(ArrayRef) -> VortexResult<RecordBatch> + Send + Sync + 'static,
) -> BoxStream<'static, VortexResult<RecordBatch>> {
    let scope = WorkScope {
        file_ordinal: 0,
        rows: scan.rows().clone(),
    };
    let root = PlanRoot::new(scope, root, scan.io());
    let batches = ParallelDriver::new(handle)
        .with_workers(workers)
        .with_ordered(ordered)
        .execute_with(root, Arc::new(map))
        .map_ok(|scoped| scoped.batch);
    // The prepared scan holds the file's shared cache: keep it until the stream ends, so scans
    // of the file that start meanwhile find it.
    batches
        .chain(stream::poll_fn(move |_| {
            let _ = &scan;
            Poll::Ready(None)
        }))
        .boxed()
}
