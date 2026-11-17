// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_error::VortexResult;
use vortex_io::request::IoService;

use super::MapFn;
use super::PlanDriver;
use super::PlanRoot;
use super::ScopedBatch;
use super::admit;
use super::sort_by_scope;
use crate::execution::planning::driver::Progress;
use crate::execution::planning::driver::Run;

/// Runs a plan on one [`Run`], on the task that polls its stream.
///
/// Independent children run as further roots of the same run, so batches follow the run's
/// depth-first priority: an earlier file's batches come before a later file's, unless the earlier
/// one waits for a read. Nothing is spawned, so this needs no runtime.
#[derive(Clone)]
pub struct SequentialDriver {
    io: Arc<dyn IoService>,
    ordered: bool,
}

impl SequentialDriver {
    /// A driver that runs every root on the polling task, reading through `io`.
    pub fn new(io: Arc<dyn IoService>) -> Self {
        Self { io, ordered: false }
    }

    /// Whether to yield batches by file and first row, holding them until the plan is done.
    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }

    /// Runs `root`, applying `map_fn` to each batch.
    pub fn execute_with<A: Send + 'static>(
        &self,
        root: PlanRoot,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<ScopedBatch<A>>> {
        let mut run = Run::new();
        admit(&mut run, self.io.as_ref(), root);
        let io = Arc::clone(&self.io);
        let batches = stream::try_unfold((run, io, map_fn), |(mut run, io, map_fn)| async move {
            loop {
                match run.advance()? {
                    Progress::Batch(batch) => {
                        let batch = ScopedBatch {
                            scope: batch.scope,
                            batch: map_fn(batch.array)?,
                        };
                        return Ok(Some((batch, (run, io, map_fn))));
                    }
                    Progress::Spawn(scope, planner) => {
                        admit(&mut run, io.as_ref(), PlanRoot::new(scope, planner));
                    }
                    Progress::RootDone(_) => {}
                    Progress::Idle => return Ok(None),
                    Progress::Waiting => {
                        future::poll_fn(|cx| run.poll_completion(cx)).await?;
                    }
                }
            }
        })
        .boxed();
        if !self.ordered {
            return batches;
        }
        stream::once(async move {
            let mut held: Vec<ScopedBatch<A>> = batches.try_collect().await?;
            sort_by_scope(&mut held);
            VortexResult::Ok(stream::iter(held.into_iter().map(Ok)))
        })
        .try_flatten()
        .boxed()
    }
}

impl PlanDriver for SequentialDriver {
    fn execute(&self, root: PlanRoot) -> BoxStream<'static, VortexResult<ScopedBatch>> {
        self.execute_with(root, Arc::new(Ok))
    }
}
