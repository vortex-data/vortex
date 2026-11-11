// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drivers that run a plan to its end and choose its parallelism.
//!
//! A planner never says how parallel a scan is. It only declares which children are
//! independent, by emitting [`PlannerOutput::Spawn`](crate::planning::planner::PlannerOutput::Spawn)
//! for a child over the same byte source or
//! [`PlannerOutput::Open`](crate::planning::planner::PlannerOutput::Open) for a child over another
//! one, such as the next file of a table. The driver decides where each independent child runs:
//!
//! - [`SequentialDriver`] runs everything on one [`Run`], on the task that polls its stream.
//! - [`ParallelDriver`] runs a pool of workers, each with a [`Run`] of its own. Every independent
//!   child goes to a queue the workers share, and a worker with spare capacity takes the next one.
//!
//! Both read each root through a fresh session of the root's [`IoService`], so a root that
//! finishes clears only its own reads.

mod parallel;
mod sequential;

use std::sync::Arc;

use futures::stream::BoxStream;
pub use parallel::ParallelDriver;
pub use sequential::SequentialDriver;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoService;
use vortex_utils::aliases::hash_map::HashMap;

use crate::planning::driver::RootId;
use crate::planning::driver::Run;
use crate::planning::driver::Spawned;
use crate::planning::planner::Planner;
use crate::planning::planner::WorkScope;

/// A planner to run as a root, with the scope it covers and the service its reads go through.
pub struct PlanRoot {
    /// The scope the planner is responsible for.
    pub scope: WorkScope,
    /// The root planner.
    pub planner: Box<dyn Planner>,
    /// The byte source the root and its dependent children read.
    pub io: Arc<dyn IoService>,
}

impl PlanRoot {
    /// A root over `scope`, reading through `io`.
    pub fn new(scope: WorkScope, planner: Box<dyn Planner>, io: Arc<dyn IoService>) -> Self {
        Self { scope, planner, io }
    }
}

/// One output of a driver: a batch, mapped on the worker that produced it, and the scope of the
/// morsel that produced it.
pub struct ScopedBatch<A = ArrayRef> {
    /// The scope of the morsel that produced the batch.
    pub scope: WorkScope,
    /// The batch.
    pub batch: A,
}

/// A mapping a driver applies to each batch on the worker that produced it, such as decoding to
/// Arrow, so that the work runs in parallel rather than on the task polling the stream.
pub type MapFn<A> = Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>;

/// Runs a plan to its end, choosing where its independent children run.
pub trait PlanDriver: Send + Sync {
    /// Runs `root` and everything it emits, streaming the batches of its morsels.
    fn execute(&self, root: PlanRoot) -> BoxStream<'static, VortexResult<ScopedBatch>>;
}

/// A [`Run`] with the service of each root it has admitted, so that a child spawned over the
/// root's byte source opens a session of the same service.
#[derive(Default)]
struct Roots {
    run: Run,
    services: HashMap<RootId, Arc<dyn IoService>>,
}

impl Roots {
    fn admit(&mut self, root: PlanRoot) {
        let id = self.run.admit(root.planner, root.scope, root.io.session());
        self.services.insert(id, root.io);
    }

    fn finish(&mut self, root: RootId) {
        self.services.remove(&root);
    }

    /// Turns a spawned child into a root of its own, over its own service or its root's.
    fn root_of(&self, spawned: Spawned) -> VortexResult<PlanRoot> {
        let io = match spawned.io {
            Some(io) => io,
            None => Arc::clone(
                self.services
                    .get(&spawned.root)
                    .ok_or_else(|| vortex_err!("{:?} spawned after it finished", spawned.root))?,
            ),
        };
        Ok(PlanRoot::new(spawned.scope, spawned.planner, io))
    }
}

/// Sorts batches into the order of their scopes: by file, then by first row. A morsel's own
/// batches keep the order it emitted them in.
fn sort_by_scope<A>(batches: &mut [ScopedBatch<A>]) {
    batches.sort_by_key(|batch| (batch.scope.file_ordinal, batch.scope.rows.start));
}

#[cfg(test)]
mod tests;
