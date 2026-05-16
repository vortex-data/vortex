//! `PipelineSubmitter`: runtime-time pipeline submission.
//!
//! Lets operators emit child pipelines mid-execution instead of
//! statically at plan build time. The submitter takes a fully-built
//! `Operator` subtree, lowers it against a fresh ctx, and schedules
//! the resulting pipelines onto the engine's sync work pool. Each
//! pipeline becomes a closure pushed onto the pool — the closure
//! does `smol::block_on(<pipeline future>)` on the worker that picks
//! it up, so the pipeline runs end-to-end on one worker thread.
//!
//! Used by dynamic-expansion operators like `Gather` to keep the
//! in-flight pipeline count bounded — submit just `target_concurrency`
//! children up front, await each to complete, then submit the next.
//! Lazier admission than a static plan with N pre-spawned pipelines;
//! `init_local` on a child only happens once it's submitted.

use std::pin::Pin;
use std::sync::Arc;

use futures::Future;
use futures::channel::oneshot;

use crate::EngineError;
use crate::EngineResult;
use crate::physical_plan::ids::PipelineBarrier;
use crate::physical_plan::lowering::PipelineBuilder;
use crate::physical_plan::lowering::PipelineTail;
use crate::physical_plan::plan::Operator;
use crate::physical_plan::pool::Runtime;
use crate::physical_plan::spawn::SpawnRuntime;

/// Type-erased future returned by `PipelineSubmitter::submit`.
/// `Send` so it can live inside a Send future / be moved to a
/// different worker. The future's body only awaits oneshot
/// receivers, which are Send.
pub type SubmissionFuture = Pin<Box<dyn Future<Output = EngineResult<()>> + Send>>;

/// Submit an `Operator` subtree onto a running runtime. Cheap to
/// clone (one Arc bump); typically operators stash one per source.
#[derive(Clone)]
pub struct PipelineSubmitter {
    inner: Arc<SubmitterInner>,
}

struct SubmitterInner {
    runtime: Arc<Runtime>,
    registry: Arc<crate::physical_plan::runtime::BarrierRegistry>,
    spawn: SpawnRuntime,
}

impl PipelineSubmitter {
    pub(crate) fn new(
        runtime: Arc<Runtime>,
        registry: Arc<crate::physical_plan::runtime::BarrierRegistry>,
        spawn: SpawnRuntime,
    ) -> Self {
        Self {
            inner: Arc::new(SubmitterInner {
                runtime,
                registry,
                spawn,
            }),
        }
    }

    /// Lower `op` against a fresh `PipelineBuilder`, scheduling the
    /// resulting pipelines on the runtime. Returns a future that
    /// resolves once *all* of them complete (or the first one errors).
    pub fn submit(
        &self,
        op: Box<dyn Operator>,
        tail: PipelineTail,
    ) -> EngineResult<SubmissionFuture> {
        // 1. Lower the operator subtree to a fresh plan.
        let mut builder = PipelineBuilder::new();
        op.lower(&mut builder, tail)
            .map_err(|e| EngineError::message(format!("submitter: lower: {e}")))?;
        let plan = builder.into_plan();

        // 2. Push each pipeline as a sync closure onto the worker
        //    pool. The closure does `block_on(<pipeline future>)`
        //    on whatever worker picks it up.
        let mut completion_rxs = Vec::with_capacity(plan.pipelines.len());
        for pipeline in plan.pipelines {
            let (tx, rx) = oneshot::channel::<EngineResult<()>>();
            let reg = Arc::clone(&self.inner.registry);
            let spawn = self.inner.spawn.clone();
            let submitter = self.clone();
            self.inner.runtime.spawn(Box::new(move || {
                let result = smol::block_on(run_pipeline(pipeline, reg, spawn, submitter));
                drop(tx.send(result));
            }));
            completion_rxs.push(rx);
        }

        // 3. Return a future that joins all of them.
        Ok(Box::pin(async move {
            let mut first_err: Option<EngineError> = None;
            for rx in completion_rxs {
                match rx.await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        if first_err.is_none() {
                            first_err = Some(e);
                        }
                    }
                    Err(_) => {
                        if first_err.is_none() {
                            first_err = Some(EngineError::message(
                                "submitted pipeline was dropped without completing",
                            ));
                        }
                    }
                }
            }
            match first_err {
                Some(e) => Err(e),
                None => Ok(()),
            }
        }))
    }
}

/// The same `run_pipeline_async` logic as `runtime.rs` runs at plan
/// startup. Inlined here to avoid an awkward visibility juggling
/// (the submitter is in a sibling module to `runtime`).
async fn run_pipeline(
    pipeline: crate::physical_plan::lowering::Pipeline,
    registry: Arc<crate::physical_plan::runtime::BarrierRegistry>,
    spawn: SpawnRuntime,
    submitter: PipelineSubmitter,
) -> EngineResult<()> {
    // Wait for any upstream barriers.
    for barrier in pipeline.depends_on() {
        let latch = registry.get_or_insert(*barrier);
        latch.wait().await;
    }
    let publishes: Vec<PipelineBarrier> = pipeline.publishes().iter().copied().collect();
    let (source, transforms, sink) = pipeline.take_source();
    let result = crate::physical_plan::runtime::drive_pipeline(
        source,
        transforms,
        sink,
        &spawn,
        &submitter,
    )
    .await;
    for barrier in publishes {
        let latch = registry.get_or_insert(barrier);
        latch.fire();
    }
    result
}
