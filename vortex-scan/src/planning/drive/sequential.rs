// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use super::MapFn;
use super::PlanRoot;
use super::ScopedBatch;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;

/// Runs a plan on one [`Run`], on the task that polls its stream.
///
/// Independent children run as further roots of the same run, so batches follow the run's
/// depth-first priority: an earlier chunk's batches come before a later one's, unless the earlier
/// one waits for a read. Nothing is spawned, so this needs no runtime.
#[derive(Clone, Copy, Debug, Default)]
pub struct SequentialDriver;

impl SequentialDriver {
    /// A driver that runs every root on the polling task.
    pub fn new() -> Self {
        Self
    }

    /// Runs `root` and everything it emits, streaming the batches of its morsels.
    pub fn execute(&self, root: PlanRoot) -> BoxStream<'static, VortexResult<ScopedBatch>> {
        self.execute_with(root, Arc::new(Ok))
    }

    /// Runs `root`, applying `map_fn` to each batch.
    pub fn execute_with<A: Send + 'static>(
        &self,
        root: PlanRoot,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<ScopedBatch<A>>> {
        let io = Arc::clone(&root.io);
        let mut run = Run::new();
        root.admit_to(&mut run);
        stream::try_unfold((run, io, map_fn), |(mut run, io, map_fn)| async move {
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
                        PlanRoot::new(scope, planner, Arc::clone(&io)).admit_to(&mut run);
                    }
                    Progress::RootDone(_) => {}
                    Progress::Idle => return Ok(None),
                    Progress::Waiting => {
                        future::poll_fn(|cx| run.poll_completion(cx)).await?;
                    }
                }
            }
        })
        .boxed()
    }
}
