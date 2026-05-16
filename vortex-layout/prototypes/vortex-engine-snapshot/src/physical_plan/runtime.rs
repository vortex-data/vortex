//! v0 pipeline runtime.
//!
//! Each pipeline runs as a sync closure dispatched onto the engine's
//! [`Runtime`] work pool. The closure does
//! `smol::block_on(<pipeline future>)` on the worker that picks it
//! up, so the pipeline's source/transforms/sink state machine runs
//! end-to-end on one worker thread. Async waits inside the pipeline
//! (channel receives, async I/O via `DriverIo`) park that block_on;
//! the worker thread is dedicated to this pipeline for its lifetime.
//!
//! Cross-pipeline coordination uses latching barriers (sticky
//! `event_listener::Event`s) registered against `PipelineBarrier`
//! ids. The runtime owns the `SpawnRuntime` and threads it through
//! to every operator ctx.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::thread;

use event_listener::Event;
use futures::channel::oneshot;

use crate::EngineError;
use crate::EngineResult;
use crate::physical_plan::abi::{
    Batch, LocalInitRuntime, OperatorPoll, PendingSend, SinkCtx, SinkDriver, SourceCtx,
    SourceDriver, TransformCtx, TransformDriver, TransformOutput,
};
use crate::physical_plan::ids::PipelineBarrier;
use crate::physical_plan::lowering::{LoweredPlan, Pipeline};
use crate::physical_plan::pool::Runtime;
use crate::physical_plan::spawn::SpawnRuntime;

/// Sticky one-shot barrier. Once `fire()` is called, all future
/// `wait()`s return immediately. Listeners registered between the
/// barrier construction and `fire()` are notified.
pub(crate) struct LatchBarrier {
    fired: parking_lot::Mutex<bool>,
    event: Event,
}

impl LatchBarrier {
    fn new() -> Self {
        Self {
            fired: parking_lot::Mutex::new(false),
            event: Event::new(),
        }
    }

    pub(crate) fn fire(&self) {
        *self.fired.lock() = true;
        self.event.notify(usize::MAX);
    }

    pub(crate) async fn wait(self: &Arc<Self>) {
        loop {
            let listener = self.event.listen();
            if *self.fired.lock() {
                return;
            }
            listener.await;
        }
    }
}

#[derive(Default)]
pub(crate) struct BarrierRegistry {
    barriers: parking_lot::Mutex<HashMap<PipelineBarrier, Arc<LatchBarrier>>>,
}

impl BarrierRegistry {
    pub(crate) fn get_or_insert(&self, barrier: PipelineBarrier) -> Arc<LatchBarrier> {
        let mut barriers = self.barriers.lock();
        Arc::clone(
            barriers
                .entry(barrier)
                .or_insert_with(|| Arc::new(LatchBarrier::new())),
        )
    }
}

/// Drive a `LoweredPlan` to completion on a single-thread executor.
///
/// A fresh [`DriverIo`] is created for this run. Callers that want
/// to share the I/O pool across many calls (e.g. one DriverIo per
/// process, many `run_plan_blocking` calls in parallel) should use
/// [`run_plan_blocking_with_io`] instead.
pub fn run_plan_blocking(plan: LoweredPlan) -> EngineResult<()> {
    let io = crate::physical_plan::DriverIo::new(driver_io_workers());
    run_plan_blocking_with_io(plan, io)
}

/// Variant that takes a shared DriverIo. Useful when running many
/// independent plans concurrently on a thread pool — each plan picks
/// up the same I/O substrate without paying the per-call cost of
/// spawning and tearing down worker threads.
pub fn run_plan_blocking_with_io(
    plan: LoweredPlan,
    io: Arc<crate::physical_plan::DriverIo>,
) -> EngineResult<()> {
    let runtime = Runtime::new(compute_workers());
    let spawn = SpawnRuntime::new(Arc::clone(&io));
    let registry = Arc::new(BarrierRegistry::default());
    let submitter = crate::physical_plan::submitter::PipelineSubmitter::new(
        Arc::clone(&runtime),
        Arc::clone(&registry),
        spawn.clone(),
    );

    // Push each top-level pipeline as a sync closure onto the work
    // pool. Each closure runs `block_on(<pipeline future>)` on its
    // worker thread for the lifetime of the pipeline.
    let pipelines = plan.pipelines;
    let mut completion_rxs = Vec::with_capacity(pipelines.len());
    for pipeline in pipelines {
        let (tx, rx) = oneshot::channel::<EngineResult<()>>();
        let reg = Arc::clone(&registry);
        let spawn_for_pipeline = spawn.clone();
        let submitter_for_pipeline = submitter.clone();
        runtime.spawn(Box::new(move || {
            let result = smol::block_on(run_pipeline_async(
                pipeline,
                reg,
                spawn_for_pipeline,
                submitter_for_pipeline,
            ));
            drop(tx.send(result));
        }));
        completion_rxs.push(rx);
    }

    // Wait for all top-level pipelines on the calling thread. The
    // pipelines themselves are running on Runtime workers; we just
    // block here until every oneshot fires.
    let result = smol::block_on(async move {
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
                            "top-level pipeline dropped without completing",
                        ));
                    }
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok::<(), EngineError>(()),
        }
    });

    // Drop the submitter first so the pool's Arc count can return to
    // one before shutdown.
    drop(submitter);
    runtime.shutdown();
    drop(spawn);
    drop(io);
    result
}

/// How many threads to dedicate to the DriverIo I/O pool. Tunable via
/// `VORTEX_ENGINE_IO_THREADS` (default: 4).
fn driver_io_workers() -> usize {
    std::env::var("VORTEX_ENGINE_IO_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n: &usize| *n > 0)
        .unwrap_or(4)
}

/// Number of compute worker threads driving the shared executor.
/// Defaults to `num_cpus`; override via `VORTEX_ENGINE_WORKERS`.
fn compute_workers() -> usize {
    std::env::var("VORTEX_ENGINE_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n: &usize| *n > 0)
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(8))
}

async fn run_pipeline_async(
    pipeline: Pipeline,
    registry: Arc<BarrierRegistry>,
    spawn: SpawnRuntime,
    submitter: crate::physical_plan::submitter::PipelineSubmitter,
) -> EngineResult<()> {
    // Wait for upstream barriers before starting the pipeline.
    for barrier in pipeline.depends_on() {
        let latch = registry.get_or_insert(*barrier);
        latch.wait().await;
    }

    let publishes: Vec<PipelineBarrier> = pipeline.publishes().iter().copied().collect();
    let (source, transforms, sink) = pipeline.take_source();

    let result = drive_pipeline(source, transforms, sink, &spawn, &submitter).await;

    for barrier in publishes {
        let latch = registry.get_or_insert(barrier);
        latch.fire();
    }

    result
}

/// State machine that drives one source → transforms → sink chain.
/// Implements `Future` by polling each role per tick.
struct PipelineDriver {
    source_domain: crate::Domain,
    source_contract: crate::OutputContract,
    sink_domain: crate::Domain,
    sink_contract: crate::OutputContract,

    source: Box<dyn SourceDriver>,
    transforms: Vec<Box<dyn TransformDriver>>,
    sink: Box<dyn SinkDriver>,
    spawn: SpawnRuntime,

    state: DriverState,
}

enum DriverState {
    Producing,
    Finalising,
    Done,
}

impl PipelineDriver {
    fn poll_tick(&mut self, cx: &mut Context<'_>) -> OperatorPoll<()> {
        loop {
            match self.state {
                DriverState::Done => return Poll::Ready(Ok(())),
                DriverState::Finalising => {
                    // Mark all transforms as input-finished.
                    {
                        let mut t_ctx = TransformCtx::new(cx, &self.spawn);
                        for t in self.transforms.iter_mut() {
                            t.finish_input(&mut t_ctx)?;
                        }
                    }
                    // Drain any residual output the transforms still hold.
                    match self.drain_transforms(cx)? {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(()) => {}
                    }
                    // Finish the sink.
                    let mut sink_ctx = SinkCtx::new(
                        cx,
                        &self.sink_domain,
                        &self.sink_contract,
                        &self.spawn,
                    );
                    match self.sink.poll_finish(&mut sink_ctx)? {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(()) => {
                            self.state = DriverState::Done;
                            return Poll::Ready(Ok(()));
                        }
                    }
                }
                DriverState::Producing => {
                    // 1. Drain transforms top-down.
                    match self.drain_transforms(cx)? {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(()) => {}
                    }

                    // 2. Pull a batch from the source.
                    let next = {
                        let mut src_ctx = SourceCtx::new(
                            cx,
                            &self.source_domain,
                            &self.source_contract,
                            &self.spawn,
                        );
                        self.source.poll_next(&mut src_ctx)?
                    };
                    match next {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(None) => {
                            self.state = DriverState::Finalising;
                            // continue loop
                        }
                        Poll::Ready(Some(batch)) => {
                            match push_through_from(
                                &mut self.transforms,
                                0,
                                &mut *self.sink,
                                &self.sink_domain,
                                &self.sink_contract,
                                &self.spawn,
                                cx,
                                batch,
                            )? {
                                Poll::Pending => return Poll::Pending,
                                Poll::Ready(()) => {}
                            }
                            // continue loop to pull more
                        }
                    }
                }
            }
        }
    }

    /// Pull pending output from each transform (bottom-up) and route
    /// it through the rest of the chain to the sink. Used by both
    /// Producing and Finalising states.
    ///
    /// Note: per-iteration `TransformCtx::new(cx, …)` looks redundant
    /// but is essentially free (two pointer-sized field assigns) and
    /// keeps the borrow checker happy — we need `cx` again inside
    /// `push_through_from`, which we can't do while a `TransformCtx`
    /// holds `&mut cx`.
    fn drain_transforms(&mut self, cx: &mut Context<'_>) -> OperatorPoll<()> {
        for i in (0..self.transforms.len()).rev() {
            loop {
                let poll_result = {
                    let mut t_ctx = TransformCtx::new(cx, &self.spawn);
                    self.transforms[i].poll_next_output(&mut t_ctx)?
                };
                match poll_result {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(TransformOutput::Batch(batch)) => {
                        match push_through_from(
                            &mut self.transforms,
                            i + 1,
                            &mut *self.sink,
                            &self.sink_domain,
                            &self.sink_contract,
                            &self.spawn,
                            cx,
                            batch,
                        )? {
                            Poll::Pending => return Poll::Pending,
                            Poll::Ready(()) => {}
                        }
                    }
                    Poll::Ready(TransformOutput::NeedInput)
                    | Poll::Ready(TransformOutput::Finished) => break,
                }
            }
        }
        Poll::Ready(Ok(()))
    }
}

impl Future for PipelineDriver {
    type Output = EngineResult<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: PipelineDriver is Unpin (no self-references).
        let this = unsafe { self.get_unchecked_mut() };
        this.poll_tick(cx)
    }
}

pub(crate) async fn drive_pipeline(
    source: crate::physical_plan::lowering::PipelineSource,
    transforms: Vec<crate::physical_plan::lowering::PipelineTransform>,
    sink: crate::physical_plan::lowering::PipelineSink,
    spawn: &SpawnRuntime,
    submitter: &crate::physical_plan::submitter::PipelineSubmitter,
) -> EngineResult<()> {
    let source_domain = source.output_domain().clone();
    let source_contract = source.output_contract().clone();
    let sink_domain = sink.input_domain().clone();
    let sink_contract = sink.input_contract().clone();

    let mut init_rt =
        LocalInitRuntime::detached_with_spawn(spawn).with_submitter(submitter);
    let source_driver = source.node().init_local(&mut init_rt)?;
    let mut transform_drivers = Vec::with_capacity(transforms.len());
    for transform in &transforms {
        let mut init =
            LocalInitRuntime::detached_with_spawn(spawn).with_submitter(submitter);
        transform_drivers.push(transform.node().init_local(&mut init)?);
    }
    let sink_driver = sink.node().init_local(&mut init_rt)?;

    let driver = PipelineDriver {
        source_domain,
        source_contract,
        sink_domain,
        sink_contract,
        source: source_driver,
        transforms: transform_drivers,
        sink: sink_driver,
        spawn: spawn.clone(),
        state: DriverState::Producing,
    };
    driver.await
}

#[allow(clippy::too_many_arguments)]
fn push_through_from(
    transforms: &mut [Box<dyn TransformDriver>],
    start: usize,
    sink_driver: &mut dyn SinkDriver,
    sink_domain: &crate::Domain,
    sink_contract: &crate::OutputContract,
    spawn: &SpawnRuntime,
    cx: &mut Context<'_>,
    batch: Batch,
) -> OperatorPoll<()> {
    let mut current = batch;
    for t in transforms.iter_mut().skip(start) {
        // One `TransformCtx` per transform iteration, reused across
        // its `push_input` + `poll_next_output` calls. (The ctx is a
        // pair of pointer-sized references — construction is
        // essentially free either way — but threading one through is
        // cleaner.)
        let mut t_ctx = TransformCtx::new(cx, spawn);
        t.push_input(current, &mut t_ctx)?;
        match t.poll_next_output(&mut t_ctx)? {
            Poll::Ready(TransformOutput::Batch(b)) => current = b,
            Poll::Ready(TransformOutput::NeedInput) => return Poll::Ready(Ok(())),
            Poll::Ready(TransformOutput::Finished) => return Poll::Ready(Ok(())),
            Poll::Pending => return Poll::Pending,
        }
    }
    let mut sink_ctx = SinkCtx::new(cx, sink_domain, sink_contract, spawn);
    let mut send = PendingSend::new(current);
    loop {
        match sink_driver.poll_send(&mut sink_ctx, &mut send)? {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(()) => {
                if send.is_consumed() {
                    return Poll::Ready(Ok(()));
                }
                // Sink reported Ready but didn't take — should not
                // happen with well-behaved sinks. Treat as Pending
                // to be safe.
                return Poll::Pending;
            }
        }
    }
}
