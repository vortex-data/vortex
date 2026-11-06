// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_error::VortexResult;

use super::MapFn;
use super::PlanDriver;
use super::PlanRoot;
use super::Roots;
use super::ScopedBatch;
use crate::planning::driver::Progress;

/// Runs a plan on one [`Run`](crate::planning::driver::Run), on the task that polls its stream.
///
/// Independent children run as further roots of the same run, so batches follow the run's
/// depth-first priority: an earlier file's batches come before a later file's, unless the earlier
/// one waits for a read. Nothing is spawned, so this needs no runtime.
#[derive(Clone, Copy, Debug, Default)]
pub struct SequentialDriver;

impl SequentialDriver {
    /// A driver that runs every root on the polling task.
    pub fn new() -> Self {
        Self
    }

    /// Runs `root`, applying `map_fn` to each batch.
    pub fn execute_with<A: Send + 'static>(
        &self,
        root: PlanRoot,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<ScopedBatch<A>>> {
        let mut roots = Roots::default();
        roots.admit(root);
        stream::try_unfold((roots, map_fn), |(mut roots, map_fn)| async move {
            loop {
                match roots.run.advance()? {
                    Progress::Batch(batch) => {
                        let batch = ScopedBatch {
                            scope: batch.scope,
                            batch: map_fn(batch.array)?,
                        };
                        return Ok(Some((batch, (roots, map_fn))));
                    }
                    Progress::Spawn(spawned) => {
                        let root = roots.root_of(spawned)?;
                        roots.admit(root);
                    }
                    Progress::RootDone(root) => roots.finish(root),
                    Progress::Idle => return Ok(None),
                    Progress::Waiting => {
                        future::poll_fn(|cx| roots.run.poll_completion(cx)).await?;
                    }
                }
            }
        })
        .boxed()
    }
}

impl PlanDriver for SequentialDriver {
    fn execute(&self, root: PlanRoot) -> BoxStream<'static, VortexResult<ScopedBatch>> {
        self.execute_with(root, Arc::new(Ok))
    }
}
