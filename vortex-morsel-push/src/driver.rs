// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Affinity-owned morsel execution over one shared asynchronous IO service.
//!
//! Each worker owns one arena and at most one active morsel. The arena never crosses a thread
//! boundary. Planning submits all named segment futures to scan-wide required/speculative queues;
//! while its morsel is suspended, a worker polls IO from those queues. Exact ticket completion
//! wakes only the worker whose continuation parked on that ticket. Output order is restored by
//! morsel index after all workers finish.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::task::Wake;
use std::task::Waker;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::bounded;
use crossbeam_channel::unbounded;
use parking_lot::Condvar;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_layout::segments::SegmentSource;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::build::ExecPlan;
use crate::build::PhysicalTopology;
use crate::build::PipelineId;
use crate::build::SourceRole;
use crate::build::cut_morsels;
use crate::cells::SharedCells;
use crate::io::IoKey;
use crate::io::IoPlane;
use crate::io::IoPriority;
use crate::io::IoRead;
use crate::io::IoReadPoll;
use crate::io::IoService;
use crate::node::ActivationRows;
use crate::node::ActivationTarget;
use crate::node::Arena;
use crate::node::DemandTarget;
use crate::node::ExecNode;
use crate::node::ExecPoll;
use crate::node::ExecutionMode;
use crate::node::NodeId;
use crate::node::NodeState;
use crate::node::PlanPoll;
use crate::node::PushBatch;
use crate::node::PushCx;
use crate::node::PushProfileKind;
use crate::node::StageOutput;
use crate::node::StageSideband;
use crate::node::Wait;
use crate::node::WaitSet;
use crate::node::begin_morsel;
use crate::node::poll_execute_morsel;
use crate::node::poll_plan_morsel;
use crate::node::retire_morsel;
use crate::stats::ScanStats;
use crate::stats::push_profile_enabled;

/// The morsel row ranges for a plan.
///
/// With `target_rows` of zero every natural split is a morsel boundary, which is exactly the V1
/// split set — the fair-comparison default. A larger target coalesces consecutive splits, which
/// is where the executor's ability to straddle chunk boundaries starts to pay.
pub fn morsels(plan: &ExecPlan, target_rows: u64) -> Vec<Range<u64>> {
    cut_morsels(plan.natural_splits(), target_rows)
}

fn overlapping_morsels(morsels: &[Range<u64>], range: &Range<u64>) -> usize {
    let first = morsels.partition_point(|morsel| morsel.end <= range.start);
    let end = morsels.partition_point(|morsel| morsel.start < range.end);
    end.saturating_sub(first)
}

/// One configured run of the morsel executor.
pub struct MorselScan {
    plan: Arc<ExecPlan>,
    segments: Arc<dyn SegmentSource>,
    io: Option<Arc<IoService>>,
    session: VortexSession,
    morsels: Arc<[Range<u64>]>,
    threads: usize,
    share_decodes: bool,
    execution_mode: ExecutionMode,
    lookahead_morsels: usize,
    output_rows: usize,
    output_bytes: u64,
    demand_hints: DemandHintDelivery,
    completion: Option<CompletionSink>,
    sparse_morsels: bool,
    external_driver: Option<ExternalDriver>,
}

type CompletionSink = Arc<dyn Fn(usize, VortexResult<Option<ArrayRef>>) + Send + Sync>;
type ExternalDriver = Arc<dyn Fn() + Send + Sync>;

thread_local! {
    static EXTERNAL_ARENA: RefCell<Option<(Arc<ExecPlan>, Arena)>> = const { RefCell::new(None) };
}

/// Delivery policy for optional scheduler-only demand hints.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DemandHintDelivery {
    /// Observe hints as soon as their producing node publishes them.
    #[default]
    Immediate,
    /// Drop every hint, leaving scheduling independent of the demand plane.
    Disabled,
    /// Observe a hint after this many subsequent push-node transitions.
    Delayed(usize),
}

/// Blocking ordered receiver for a running morsel scan.
pub struct MorselStream {
    output: Option<Receiver<CreditedBatch>>,
    completion: mpsc::Receiver<VortexResult<(ScanStats, Duration)>>,
    result: Option<VortexResult<(ScanStats, Duration)>>,
    error_yielded: bool,
    handle: Option<JoinHandle<()>>,
    cancellation: Arc<StreamCancellation>,
}

struct StreamCancellation {
    cancelled: AtomicBool,
    scheduler: Mutex<Option<Weak<Scheduler>>>,
}

impl StreamCancellation {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            cancelled: AtomicBool::new(false),
            scheduler: Mutex::new(None),
        })
    }

    fn install(&self, scheduler: &Arc<Scheduler>) {
        let mut slot = self.scheduler.lock();
        if self.cancelled.load(Ordering::Acquire) {
            scheduler.stop();
        } else {
            *slot = Some(Arc::downgrade(scheduler));
        }
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(scheduler) = self.scheduler.lock().as_ref().and_then(Weak::upgrade) {
            scheduler.stop();
        }
    }
}

impl Iterator for MorselStream {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.output.as_ref().and_then(|output| output.recv().ok()) {
            return Some(Ok(item.receive()));
        }
        self.output = None;
        if self.result.is_none() {
            self.result = self.completion.recv().ok();
        }
        match self.result.as_ref() {
            Some(Err(err)) if !self.error_yielded => {
                self.error_yielded = true;
                Some(Err(vortex_err!("streaming morsel scan failed: {err}")))
            }
            _ => None,
        }
    }
}

impl MorselStream {
    /// Drain remaining output and return final counters and wall time.
    pub fn finish(mut self) -> VortexResult<(ScanStats, Duration)> {
        for item in self.by_ref() {
            drop(item?);
        }
        self.result
            .take()
            .unwrap_or_else(|| Err(vortex_err!("streaming morsel scan stopped early")))
    }
}

impl Drop for MorselStream {
    fn drop(&mut self) {
        self.output = None;
        self.cancellation.cancel();
        if let Some(handle) = self.handle.take() {
            drop(handle.join());
        }
    }
}

struct WorkerRun {
    plan: Arc<ExecPlan>,
    session: VortexSession,
    morsels: Arc<[Range<u64>]>,
    io: Arc<IoService>,
    cells: SharedCells,
    start: Instant,
    execution_mode: ExecutionMode,
    lookahead_morsels: usize,
    output_rows: usize,
    output_bytes: u64,
    demand_hints: DemandHintDelivery,
    external_driver: Option<ExternalDriver>,
}

#[derive(Clone, Copy)]
enum TaskPhase {
    Plan,
    Execute,
}

const PUSH_INLINE_QUANTUM: usize = 64;

enum PipelineCall {
    Start {
        span: Range<u64>,
        rows: ActivationRows,
    },
    Resume,
    Credit,
}

struct PipelineActivation {
    pipeline: PipelineId,
    stage: usize,
    call: PipelineCall,
    ancestors: Option<Vec<PipelineFrame>>,
}

struct PipelineFrame {
    pipeline: PipelineId,
    stage: usize,
    output: StageOutput,
    resume: bool,
    yield_after_outputs: bool,
}

struct StageInvocation {
    pipeline: PipelineId,
    stage: usize,
    node: NodeId,
    output: StageOutput,
    state: NodeState,
}

struct DeferredFrames {
    target: PipelineId,
    frames: Vec<PipelineFrame>,
}

enum PipelineEffect {
    Batch {
        batch: ArrayRef,
        terminal: bool,
    },
    End,
    #[cfg(test)]
    Demand {
        target: DemandTarget,
        coverage: Range<u64>,
        selection: vortex_mask::Mask,
    },
}

enum PipelinePoll {
    Effect(PipelineEffect),
    Waiting {
        pipeline: PipelineId,
        stage: usize,
        waits: WaitSet,
    },
    Yield,
    Idle,
}

enum SidebandAction {
    Continue,
    Start(PipelineActivation),
    #[cfg(test)]
    Poll(PipelinePoll),
}

struct PhysicalRuntime {
    topology: Arc<PhysicalTopology>,
    pending: VecDeque<PipelineActivation>,
    frames: Vec<PipelineFrame>,
    suspended: HashMap<(PipelineId, usize), Vec<PipelineFrame>>,
    deferred: VecDeque<DeferredFrames>,
    sideband_scratch: Vec<VecDeque<StageSideband>>,
    blocked: Vec<bool>,
    blocked_touched: Vec<PipelineId>,
    root: NodeId,
    root_parts: Vec<ArrayRef>,
    root_range: Range<u64>,
    root_coverage_end: u64,
    root_done: bool,
    work_since_yield: usize,
}

struct PipelineServices<'a> {
    io: &'a IoPlane,
    cells: &'a SharedCells,
    session: &'a VortexSession,
    stats: &'a mut ScanStats,
}

fn record_push_stage(stats: &mut ScanStats, kind: PushProfileKind, elapsed: Duration) {
    let profile = match kind {
        PushProfileKind::Flat => &mut stats.push_profile_flat,
        PushProfileKind::Chunked => &mut stats.push_profile_chunked,
        PushProfileKind::Struct => &mut stats.push_profile_struct,
        PushProfileKind::Conjunct => &mut stats.push_profile_conjunct,
        PushProfileKind::Filter => &mut stats.push_profile_filter,
        PushProfileKind::Other => &mut stats.push_profile_other,
    };
    profile.0 += 1;
    profile.1 += elapsed;
}

impl PhysicalRuntime {
    fn schedule_ready_sources(
        &mut self,
        ready_sources: &mut Vec<(NodeId, Range<u64>, ActivationRows)>,
    ) -> Option<PipelineActivation> {
        let mut first = None;
        for (source, span, rows) in ready_sources.drain(..) {
            let (source_pipeline, source_stage) = self.location(source);
            debug_assert_eq!(source_stage, 0, "push source must head its pipeline");
            let activation = PipelineActivation {
                pipeline: source_pipeline,
                stage: source_stage,
                call: PipelineCall::Start { span, rows },
                ancestors: None,
            };
            if first.is_none() && self.work_since_yield < PUSH_INLINE_QUANTUM {
                first = Some(activation);
            } else {
                self.pending.push_back(activation);
            }
        }
        first
    }

    fn apply_sideband(
        &mut self,
        producer: NodeId,
        sideband: StageSideband,
        host: &mut PushHost<'_>,
        services: &mut PipelineServices<'_>,
    ) -> VortexResult<SidebandAction> {
        match sideband {
            StageSideband::Consumed(port) => {
                let child = self
                    .topology
                    .credit_target(producer, port)
                    .ok_or_else(|| vortex_err!("consumed unknown pipeline input"))?;
                self.enqueue_credit(child.0, child.1);
                self.work_since_yield = self.work_since_yield.saturating_add(1);
                Ok(SidebandAction::Continue)
            }
            StageSideband::Gate {
                target,
                coverage,
                rows,
            } => {
                let completed = services.stats.push_pipeline_stage_calls;
                host.gate(services.stats, target, coverage, rows, completed)?;
                services.stats.push_inline_gates += 1;
                self.work_since_yield = self.work_since_yield.saturating_add(2);
                let drain_limit = host.control.ready_sources.len().max(1);
                host.drain_gate_io(drain_limit)?;
                let first = self.schedule_ready_sources(&mut host.control.ready_sources);
                Ok(first.map_or(SidebandAction::Continue, SidebandAction::Start))
            }
            #[cfg(test)]
            StageSideband::Demand {
                target,
                coverage,
                selection,
            } => {
                self.work_since_yield = self.work_since_yield.saturating_add(1);
                Ok(SidebandAction::Poll(PipelinePoll::Effect(
                    PipelineEffect::Demand {
                        target,
                        coverage,
                        selection,
                    },
                )))
            }
        }
    }

    fn new(plan: &ExecPlan, sideband_scratch: Vec<VecDeque<StageSideband>>) -> Self {
        Self::from_topology(
            Arc::clone(plan.topology()),
            plan.len(),
            plan.root(),
            sideband_scratch,
        )
    }

    fn from_topology(
        topology: Arc<PhysicalTopology>,
        node_count: usize,
        root: NodeId,
        sideband_scratch: Vec<VecDeque<StageSideband>>,
    ) -> Self {
        debug_assert_eq!(sideband_scratch.len(), node_count);
        Self {
            blocked: vec![false; topology.pipelines().len()],
            blocked_touched: Vec::new(),
            topology,
            pending: VecDeque::new(),
            frames: Vec::new(),
            suspended: HashMap::default(),
            deferred: VecDeque::new(),
            sideband_scratch,
            root,
            root_parts: Vec::new(),
            root_range: 0..0,
            root_coverage_end: 0,
            root_done: false,
            work_since_yield: 0,
        }
    }

    fn enqueue_start(&mut self, pipeline: PipelineId, span: Range<u64>, rows: ActivationRows) {
        self.pending.push_back(PipelineActivation {
            pipeline,
            stage: 0,
            call: PipelineCall::Start { span, rows },
            ancestors: None,
        });
    }

    fn enqueue_resume(&mut self, pipeline: PipelineId, stage: usize) {
        self.pending.push_back(PipelineActivation {
            pipeline,
            stage,
            call: PipelineCall::Resume,
            ancestors: self.suspended.remove(&(pipeline, stage)),
        });
    }

    fn enqueue_credit(&mut self, pipeline: PipelineId, stage: usize) {
        self.pending.push_back(PipelineActivation {
            pipeline,
            stage,
            call: PipelineCall::Credit,
            ancestors: None,
        });
    }

    fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.frames.is_empty()
    }

    fn reset(&mut self, root_range: Range<u64>) {
        let mut reclaimed = Vec::new();
        reclaimed.extend(
            self.pending
                .drain(..)
                .filter_map(|activation| activation.ancestors)
                .flatten(),
        );
        reclaimed.append(&mut self.frames);
        for (_, frames) in self.suspended.drain() {
            reclaimed.extend(frames);
        }
        for deferred in self.deferred.drain(..) {
            reclaimed.extend(deferred.frames);
        }
        for frame in reclaimed {
            let node =
                self.topology.pipelines()[frame.pipeline as usize].stages()[frame.stage].node();
            let mut sidebands = frame.output.into_sidebands();
            sidebands.clear();
            if sidebands.capacity() > self.sideband_scratch[node as usize].capacity() {
                self.sideband_scratch[node as usize] = sidebands;
            }
        }
        for sidebands in &mut self.sideband_scratch {
            sidebands.clear();
        }
        for pipeline in self.blocked_touched.drain(..) {
            self.blocked[pipeline as usize] = false;
        }
        self.root_parts.clear();
        self.root_coverage_end = root_range.start;
        self.root_range = root_range;
        self.root_done = false;
        self.work_since_yield = 0;
    }

    fn into_sidebands(self) -> Vec<VecDeque<StageSideband>> {
        self.sideband_scratch
    }

    fn location(&self, node: NodeId) -> (PipelineId, usize) {
        self.topology.location(node)
    }

    fn accept_root_batch(
        &mut self,
        batch: PushBatch,
        terminal: bool,
        stats: &mut ScanStats,
    ) -> VortexResult<Option<ArrayRef>> {
        let profile_started = push_profile_enabled().then(Instant::now);
        let poll = accept_root_fragment(
            &mut self.root_parts,
            &mut self.root_coverage_end,
            &self.root_range,
            batch,
            terminal,
        )?;
        if let Some(started) = profile_started {
            stats.push_profile_root_time += started.elapsed();
        }
        match poll {
            RootFragmentPoll::Continue => {
                let (pipeline, stage) = self.location(self.root);
                self.enqueue_credit(pipeline, stage);
                Ok(None)
            }
            RootFragmentPoll::Terminal(array) => Ok(array),
        }
    }

    fn invoke(
        &mut self,
        arena: &mut Arena,
        host: &mut PushHost<'_>,
        pipeline: PipelineId,
        stage: usize,
        call: PipelineCall,
        services: &mut PipelineServices<'_>,
    ) -> VortexResult<Option<PipelinePoll>> {
        let instance = self
            .topology
            .pipelines()
            .get(pipeline as usize)
            .ok_or_else(|| vortex_err!("unknown physical pipeline {pipeline}"))?;
        let node_id = instance
            .stages()
            .get(stage)
            .ok_or_else(|| vortex_err!("pipeline {pipeline} has no stage {stage}"))?
            .node();
        let sidebands = std::mem::take(&mut self.sideband_scratch[node_id as usize]);
        let node = arena.push_node_mut(node_id);
        let mut output = StageOutput::with_sidebands(sidebands);
        services.stats.push_node_transitions += 1;
        services.stats.push_pipeline_stage_calls += 1;
        let profile = push_profile_enabled().then(|| (node.push_profile_kind(), Instant::now()));
        let mut cx = PushCx::new(
            services.io,
            services.cells,
            services.session,
            services.stats,
        );
        let state = match call {
            PipelineCall::Start { span, rows } => {
                node.push_start(span, rows, &mut cx, &mut output)?
            }
            PipelineCall::Resume => node.push_resume(&mut cx, &mut output)?,
            PipelineCall::Credit => node.push_credit(&mut cx, &mut output)?,
        };
        if let Some((kind, started)) = profile {
            record_push_stage(services.stats, kind, started.elapsed());
        }
        let completed = services.stats.push_pipeline_stage_calls;
        host.flush_due(services.stats, completed)?;
        self.finish_invoke(
            arena,
            host,
            StageInvocation {
                pipeline,
                stage,
                node: node_id,
                output,
                state,
            },
            services,
        )
    }

    fn invoke_activation(
        &mut self,
        arena: &mut Arena,
        host: &mut PushHost<'_>,
        activation: PipelineActivation,
        services: &mut PipelineServices<'_>,
    ) -> VortexResult<Option<PipelinePoll>> {
        debug_assert!(activation.ancestors.is_none());
        services.stats.push_pipeline_runs += 1;
        self.work_since_yield = self.work_since_yield.saturating_add(1);
        self.invoke(
            arena,
            host,
            activation.pipeline,
            activation.stage,
            activation.call,
            services,
        )
    }

    fn invoke_input(
        &mut self,
        arena: &mut Arena,
        host: &mut PushHost<'_>,
        location: PipelineStage,
        port: crate::node::InputPort,
        input: (PushBatch, bool),
        services: &mut PipelineServices<'_>,
    ) -> VortexResult<Option<PipelinePoll>> {
        let PipelineStage { pipeline, stage } = location;
        let (batch, last_for_input) = input;
        let parent = self.topology.pipelines()[pipeline as usize].stages()[stage].node();
        let sidebands = std::mem::take(&mut self.sideband_scratch[parent as usize]);
        let mut output = StageOutput::with_sidebands(sidebands);
        services.stats.push_node_transitions += 1;
        services.stats.push_pipeline_stage_calls += 1;
        services.stats.push_pipeline_boundary_resumes += 1;
        let profile = push_profile_enabled().then(|| {
            (
                arena.push_node_mut(parent).push_profile_kind(),
                Instant::now(),
            )
        });
        let mut cx = PushCx::new(
            services.io,
            services.cells,
            services.session,
            services.stats,
        );
        let state = arena.push_node_mut(parent).push_input(
            port,
            batch,
            last_for_input,
            &mut cx,
            &mut output,
        )?;
        if let Some((kind, started)) = profile {
            record_push_stage(services.stats, kind, started.elapsed());
        }
        let completed = services.stats.push_pipeline_stage_calls;
        host.flush_due(services.stats, completed)?;
        self.finish_invoke(
            arena,
            host,
            StageInvocation {
                pipeline,
                stage,
                node: parent,
                output,
                state,
            },
            services,
        )
    }

    fn invoke_end(
        &mut self,
        arena: &mut Arena,
        host: &mut PushHost<'_>,
        pipeline: PipelineId,
        stage: usize,
        port: crate::node::InputPort,
        services: &mut PipelineServices<'_>,
    ) -> VortexResult<Option<PipelinePoll>> {
        let parent = self.topology.pipelines()[pipeline as usize].stages()[stage].node();
        let sidebands = std::mem::take(&mut self.sideband_scratch[parent as usize]);
        let mut output = StageOutput::with_sidebands(sidebands);
        services.stats.push_node_transitions += 1;
        services.stats.push_pipeline_stage_calls += 1;
        services.stats.push_pipeline_boundary_resumes += 1;
        let profile = push_profile_enabled().then(|| {
            (
                arena.push_node_mut(parent).push_profile_kind(),
                Instant::now(),
            )
        });
        let mut cx = PushCx::new(
            services.io,
            services.cells,
            services.session,
            services.stats,
        );
        let state = arena
            .push_node_mut(parent)
            .push_end(port, &mut cx, &mut output)?;
        if let Some((kind, started)) = profile {
            record_push_stage(services.stats, kind, started.elapsed());
        }
        let completed = services.stats.push_pipeline_stage_calls;
        host.flush_due(services.stats, completed)?;
        self.finish_invoke(
            arena,
            host,
            StageInvocation {
                pipeline,
                stage,
                node: parent,
                output,
                state,
            },
            services,
        )
    }

    fn finish_invoke(
        &mut self,
        arena: &mut Arena,
        host: &mut PushHost<'_>,
        invocation: StageInvocation,
        services: &mut PipelineServices<'_>,
    ) -> VortexResult<Option<PipelinePoll>> {
        let StageInvocation {
            mut pipeline,
            mut stage,
            mut node,
            mut output,
            mut state,
        } = invocation;
        loop {
            if matches!(state, NodeState::Waiting(_)) && !output.is_empty() {
                return Err(vortex_err!(
                    "pipeline {pipeline} produced output while waiting"
                ));
            }
            if let NodeState::Waiting(waits) = state {
                self.sideband_scratch[node as usize] = output.into_sidebands();
                self.work_since_yield = 0;
                let previous = self
                    .suspended
                    .insert((pipeline, stage), std::mem::take(&mut self.frames));
                if previous.is_some() {
                    return Err(vortex_err!(
                        "pipeline {pipeline} stage {stage} blocked twice without resuming"
                    ));
                }
                if !self.blocked[pipeline as usize] {
                    self.blocked[pipeline as usize] = true;
                    self.blocked_touched.push(pipeline);
                }
                return Ok(Some(PipelinePoll::Waiting {
                    pipeline,
                    stage,
                    waits,
                }));
            }
            if matches!(state, NodeState::Done) && node == self.root {
                self.root_done = true;
            }

            while let Some(sideband) = output.take_inline_sideband() {
                match self.apply_sideband(node, sideband, host, services)? {
                    SidebandAction::Continue => {}
                    #[cfg(test)]
                    SidebandAction::Poll(poll) => return Ok(Some(poll)),
                    SidebandAction::Start(activation) => {
                        let resume = matches!(state, NodeState::Ready | NodeState::Yield(_));
                        if !output.is_empty() || resume {
                            self.frames.push(PipelineFrame {
                                pipeline,
                                stage,
                                output,
                                resume,
                                yield_after_outputs: matches!(state, NodeState::Yield(_)),
                            });
                        } else {
                            self.sideband_scratch[node as usize] = output.into_sidebands();
                        }
                        return self.invoke_activation(arena, host, activation, services);
                    }
                }
                if self.work_since_yield >= PUSH_INLINE_QUANTUM && output.has_sidebands() {
                    services.stats.push_cold_frame_spills += 1;
                    services.stats.push_dispatch_spills += 1;
                    self.frames.push(PipelineFrame {
                        pipeline,
                        stage,
                        output,
                        resume: matches!(state, NodeState::Ready | NodeState::Yield(_)),
                        yield_after_outputs: matches!(state, NodeState::Yield(_)),
                    });
                    self.work_since_yield = 0;
                    return Ok(Some(PipelinePoll::Yield));
                }
            }

            let passive = matches!(state, NodeState::NeedInput | NodeState::Done);
            if passive && !output.has_sidebands() {
                if let Some((batch, last_for_input)) = output.take_batch() {
                    let target = self.topology.outgoing(node);
                    if let Some(target) = target
                        && (!target.boundary || !self.blocked[target.pipeline as usize])
                    {
                        // A one-cut Chunked node only validates root-coordinate coverage before
                        // forwarding the identical terminal batch. Bypass its queue and protocol
                        // frame when its already-compiled successor can run immediately. The
                        // node method rejects every multi-cut, partial, or credit-held case.
                        if let Some(successor) = self.topology.outgoing(target.node)
                            && (!successor.boundary || !self.blocked[successor.pipeline as usize])
                            && arena
                                .push_node_mut(target.node)
                                .try_accept_single_chunked_terminal(
                                    target.input,
                                    &batch,
                                    last_for_input,
                                )?
                        {
                            services.stats.push_fast_stage_transfers += 1;
                            services.stats.push_inline_transfers += 1;
                            services.stats.push_node_transitions += 1;
                            services.stats.push_pipeline_stage_calls += 1;
                            services.stats.push_pipeline_boundary_resumes += 1;

                            pipeline = successor.pipeline;
                            stage = successor.stage;
                            let next_sidebands =
                                std::mem::take(&mut self.sideband_scratch[successor.node as usize]);
                            self.sideband_scratch[node as usize] =
                                output.replace_sidebands(next_sidebands);
                            node = successor.node;
                            let mut cx = PushCx::new(
                                services.io,
                                services.cells,
                                services.session,
                                services.stats,
                            );
                            let profile = push_profile_enabled().then(|| {
                                (
                                    arena.push_node_mut(node).push_profile_kind(),
                                    Instant::now(),
                                )
                            });
                            state = arena.push_node_mut(node).push_input(
                                successor.input,
                                batch,
                                true,
                                &mut cx,
                                &mut output,
                            )?;
                            if let Some((kind, started)) = profile {
                                record_push_stage(services.stats, kind, started.elapsed());
                            }
                            let completed = services.stats.push_pipeline_stage_calls;
                            host.flush_due(services.stats, completed)?;
                            continue;
                        }

                        services.stats.push_fast_stage_transfers += 1;
                        services.stats.push_inline_transfers += 1;
                        services.stats.push_node_transitions += 1;
                        services.stats.push_pipeline_stage_calls += 1;
                        services.stats.push_pipeline_boundary_resumes += 1;

                        pipeline = target.pipeline;
                        stage = target.stage;
                        let next_sidebands =
                            std::mem::take(&mut self.sideband_scratch[target.node as usize]);
                        self.sideband_scratch[node as usize] =
                            output.replace_sidebands(next_sidebands);
                        node = target.node;
                        let mut cx = PushCx::new(
                            services.io,
                            services.cells,
                            services.session,
                            services.stats,
                        );
                        let profile = push_profile_enabled().then(|| {
                            (
                                arena.push_node_mut(node).push_profile_kind(),
                                Instant::now(),
                            )
                        });
                        state = arena.push_node_mut(node).push_input(
                            target.input,
                            batch,
                            last_for_input,
                            &mut cx,
                            &mut output,
                        )?;
                        if let Some((kind, started)) = profile {
                            record_push_stage(services.stats, kind, started.elapsed());
                        }
                        let completed = services.stats.push_pipeline_stage_calls;
                        host.flush_due(services.stats, completed)?;
                        continue;
                    }
                    let is_root_output = node == self.root && target.is_none();
                    if is_root_output {
                        self.sideband_scratch[node as usize] = output.into_sidebands();
                        if let Some(batch) =
                            self.accept_root_batch(batch, last_for_input, services.stats)?
                        {
                            self.work_since_yield = 0;
                            return Ok(Some(PipelinePoll::Effect(PipelineEffect::Batch {
                                batch,
                                terminal: true,
                            })));
                        }
                        return Ok(None);
                    }
                    output.set_batch(batch, last_for_input);
                } else if output.take_end() {
                    let is_root_end = node == self.root && self.topology.outgoing(node).is_none();
                    if is_root_end {
                        self.sideband_scratch[node as usize] = output.into_sidebands();
                        return Ok(Some(PipelinePoll::Effect(PipelineEffect::End)));
                    }
                    output.set_end();
                }
            }

            if output.is_empty() && passive {
                self.sideband_scratch[node as usize] = output.into_sidebands();
                return Ok(None);
            }
            services.stats.push_cold_frame_spills += 1;
            self.frames.push(PipelineFrame {
                pipeline,
                stage,
                output,
                resume: matches!(state, NodeState::Ready | NodeState::Yield(_)),
                yield_after_outputs: matches!(state, NodeState::Yield(_)),
            });
            return Ok(None);
        }
    }

    fn poll(
        &mut self,
        arena: &mut Arena,
        host: &mut PushHost<'_>,
        io: &IoPlane,
        cells: &SharedCells,
        session: &VortexSession,
        stats: &mut ScanStats,
    ) -> VortexResult<PipelinePoll> {
        let mut services = PipelineServices {
            io,
            cells,
            session,
            stats,
        };
        loop {
            if self.work_since_yield >= PUSH_INLINE_QUANTUM {
                services.stats.push_dispatch_spills += 1;
                self.work_since_yield = 0;
                return Ok(PipelinePoll::Yield);
            }
            if let Some(sideband) = self
                .frames
                .last_mut()
                .and_then(|frame| frame.output.take_sideband())
            {
                let frame = self
                    .frames
                    .last()
                    .ok_or_else(|| vortex_err!("physical pipeline lost its producer frame"))?;
                let producer =
                    self.topology.pipelines()[frame.pipeline as usize].stages()[frame.stage].node();
                match self.apply_sideband(producer, sideband, host, &mut services)? {
                    SidebandAction::Continue => {}
                    #[cfg(test)]
                    SidebandAction::Poll(poll) => return Ok(poll),
                    SidebandAction::Start(activation) => {
                        if let Some(poll) =
                            self.invoke_activation(arena, host, activation, &mut services)?
                        {
                            return Ok(poll);
                        }
                    }
                }
                if self.work_since_yield >= PUSH_INLINE_QUANTUM
                    && self
                        .frames
                        .last()
                        .is_some_and(|frame| frame.output.has_sidebands())
                {
                    services.stats.push_dispatch_spills += 1;
                    self.work_since_yield = 0;
                    return Ok(PipelinePoll::Yield);
                }
                continue;
            }
            if let Some((batch, last_for_input)) = self
                .frames
                .last_mut()
                .and_then(|frame| frame.output.take_batch())
            {
                let frame = self
                    .frames
                    .last()
                    .ok_or_else(|| vortex_err!("physical pipeline lost its producer frame"))?;
                let pipeline = frame.pipeline;
                let stage = frame.stage;
                let node = self.topology.pipelines()[pipeline as usize].stages()[stage].node();
                let Some(target) = self.topology.outgoing(node) else {
                    if let Some(batch) =
                        self.accept_root_batch(batch, last_for_input, services.stats)?
                    {
                        self.work_since_yield = 0;
                        return Ok(PipelinePoll::Effect(PipelineEffect::Batch {
                            batch,
                            terminal: true,
                        }));
                    }
                    continue;
                };
                if self.blocked[target.pipeline as usize] {
                    self.frames
                        .last_mut()
                        .ok_or_else(|| vortex_err!("physical pipeline lost its producer frame"))?
                        .output
                        .set_batch(batch, last_for_input);
                    self.deferred.push_back(DeferredFrames {
                        target: target.pipeline,
                        frames: std::mem::take(&mut self.frames),
                    });
                    continue;
                }
                services.stats.push_inline_transfers += 1;
                if let Some(poll) = self.invoke_input(
                    arena,
                    host,
                    PipelineStage {
                        pipeline: target.pipeline,
                        stage: target.stage,
                    },
                    target.input,
                    (batch, last_for_input),
                    &mut services,
                )? {
                    return Ok(poll);
                }
                continue;
            }

            if self
                .frames
                .last_mut()
                .is_some_and(|frame| frame.output.take_end())
            {
                let frame = self
                    .frames
                    .last()
                    .ok_or_else(|| vortex_err!("physical pipeline lost its producer frame"))?;
                let pipeline = frame.pipeline;
                let stage = frame.stage;
                let node = self.topology.pipelines()[pipeline as usize].stages()[stage].node();
                let Some(target) = self.topology.outgoing(node) else {
                    return Ok(PipelinePoll::Effect(PipelineEffect::End));
                };
                if self.blocked[target.pipeline as usize] {
                    self.frames
                        .last_mut()
                        .ok_or_else(|| vortex_err!("physical pipeline lost its producer frame"))?
                        .output
                        .set_end();
                    self.deferred.push_back(DeferredFrames {
                        target: target.pipeline,
                        frames: std::mem::take(&mut self.frames),
                    });
                    continue;
                }
                services.stats.push_inline_transfers += 1;
                if let Some(poll) = self.invoke_end(
                    arena,
                    host,
                    target.pipeline,
                    target.stage,
                    target.input,
                    &mut services,
                )? {
                    return Ok(poll);
                }
                continue;
            }

            if let Some(frame) = self.frames.pop() {
                let pipeline = frame.pipeline;
                let stage = frame.stage;
                let resume = frame.resume;
                let yield_after_outputs = frame.yield_after_outputs;
                let node = self.topology.pipelines()[pipeline as usize].stages()[stage].node();
                self.sideband_scratch[node as usize] = frame.output.into_sidebands();
                if yield_after_outputs {
                    if resume {
                        self.pending.push_front(PipelineActivation {
                            pipeline,
                            stage,
                            call: PipelineCall::Resume,
                            ancestors: None,
                        });
                    }
                    self.work_since_yield = 0;
                    return Ok(PipelinePoll::Yield);
                }
                if resume {
                    self.work_since_yield += 1;
                    if let Some(poll) = self.invoke(
                        arena,
                        host,
                        pipeline,
                        stage,
                        PipelineCall::Resume,
                        &mut services,
                    )? {
                        return Ok(poll);
                    }
                }
                continue;
            }

            let mut deferred = None;
            for _ in 0..self.deferred.len() {
                let Some(candidate) = self.deferred.pop_front() else {
                    break;
                };
                if !self.blocked[candidate.target as usize] {
                    deferred = Some(candidate);
                    break;
                }
                self.deferred.push_back(candidate);
            }
            if let Some(deferred) = deferred {
                debug_assert!(self.frames.is_empty());
                self.frames = deferred.frames;
                continue;
            }

            let mut activation = None;
            for _ in 0..self.pending.len() {
                let Some(candidate) = self.pending.pop_front() else {
                    break;
                };
                let is_resume =
                    matches!(candidate.call, PipelineCall::Resume) && candidate.ancestors.is_some();
                if !self.blocked[candidate.pipeline as usize] || is_resume {
                    activation = Some(candidate);
                    break;
                }
                self.pending.push_back(candidate);
            }
            let Some(activation) = activation else {
                self.work_since_yield = 0;
                return Ok(PipelinePoll::Idle);
            };
            services.stats.push_pipeline_runs += 1;
            self.work_since_yield += 1;
            if let Some(ancestors) = activation.ancestors {
                debug_assert!(self.frames.is_empty());
                self.frames = ancestors;
                self.blocked[activation.pipeline as usize] = false;
            }
            if let Some(poll) = self.invoke(
                arena,
                host,
                activation.pipeline,
                activation.stage,
                activation.call,
                &mut services,
            )? {
                return Ok(poll);
            }
        }
    }
}

struct PendingPushSource {
    span: Range<u64>,
    role: SourceRole,
    demand_io: Option<BoundDemandIo>,
    parts: Vec<(Range<u64>, ActivationRows)>,
    direct_rows: Option<ActivationRows>,
    known_rows: usize,
}

#[derive(Clone)]
struct BoundDemandIo {
    key: IoKey,
    source_range: Range<u64>,
    work: Arc<IoWork>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DemandIoAction {
    Unchanged,
    Suppressed,
    Required,
}

#[derive(Default)]
struct DemandObservationScratch {
    seen: Vec<IoKey>,
    span_selected: Vec<(Range<usize>, bool)>,
}

fn apply_unissued_demand(work: &IoWork, selected: bool) -> DemandIoAction {
    if work.required.load(Ordering::Acquire) || !work.reads.iter().any(IoRead::is_unissued) {
        return DemandIoAction::Unchanged;
    }
    if !selected {
        return DemandIoAction::Suppressed;
    }
    work.required.store(true, Ordering::Release);
    DemandIoAction::Required
}

fn observe_prebound_demand(
    stats: &mut ScanStats,
    coverage: Range<u64>,
    selection: vortex_mask::Mask,
    source_nodes: &[NodeId],
    pending: &[Option<PendingPushSource>],
    sources: &[Option<BoundDemandIo>],
    scratch: &mut DemandObservationScratch,
) {
    stats.demand_hints_observed += 1;
    scratch.seen.clear();
    scratch.span_selected.clear();
    for &node in source_nodes {
        let source = pending
            .get(node as usize)
            .and_then(Option::as_ref)
            .and_then(|source| source.demand_io.as_ref())
            .or_else(|| sources.get(node as usize).and_then(Option::as_ref));
        let Some(source) = source else {
            continue;
        };
        if scratch.seen.contains(&source.key) {
            continue;
        }
        scratch.seen.push(source.key);
        if source.work.required.load(Ordering::Acquire)
            || !source.work.reads.iter().any(IoRead::is_unissued)
        {
            continue;
        }
        stats.demand_io_candidates += 1;
        let start = coverage.start.max(source.source_range.start);
        let end = coverage.end.min(source.source_range.end);
        if start >= end {
            continue;
        }
        let (Ok(start), Ok(end)) = (
            usize::try_from(start - coverage.start),
            usize::try_from(end - coverage.start),
        ) else {
            continue;
        };
        let relative = start..end;
        let selected = scratch
            .span_selected
            .iter()
            .find(|(span, _)| span == &relative)
            .map(|(_, selected)| *selected)
            .unwrap_or_else(|| {
                let selected = selection.count_range(start, end) != 0;
                scratch.span_selected.push((relative, selected));
                selected
            });
        match apply_unissued_demand(&source.work, selected) {
            DemandIoAction::Unchanged => {}
            DemandIoAction::Suppressed => stats.demand_io_suppressed += 1,
            DemandIoAction::Required => stats.demand_io_promotions += 1,
        }
    }
}

#[derive(Default)]
struct SourceActivationScratch {
    parts: Vec<Vec<(Range<u64>, ActivationRows)>>,
    rows: Vec<(Range<usize>, ActivationRows)>,
}

#[derive(Default)]
struct PushControlState {
    projection_sources: Vec<Option<PendingPushSource>>,
    source_matches: Vec<usize>,
    source_nodes: Vec<NodeId>,
    source_activation: SourceActivationScratch,
    ready_sources: Vec<(NodeId, Range<u64>, ActivationRows)>,
    demand_hints: VecDeque<PendingDemandHint>,
    has_delayed_hints: bool,
    demand_state: HashMap<DemandTarget, Vec<(Range<u64>, vortex_mask::Mask)>>,
    demand_refine_scratch: Vec<(Range<u64>, vortex_mask::Mask)>,
    demand_sources: Vec<Option<BoundDemandIo>>,
    demand_observation: DemandObservationScratch,
    demand_state_live: usize,
    #[cfg(test)]
    gate_trace: Vec<GateTrace>,
}

impl PushControlState {
    fn with_node_count(node_count: usize) -> Self {
        Self {
            projection_sources: (0..node_count).map(|_| None).collect(),
            source_activation: SourceActivationScratch {
                parts: (0..node_count).map(|_| Vec::new()).collect(),
                rows: Vec::with_capacity(node_count),
            },
            demand_sources: (0..node_count).map(|_| None).collect(),
            demand_observation: DemandObservationScratch {
                seen: Vec::with_capacity(node_count),
                span_selected: Vec::with_capacity(node_count),
            },
            ..Self::default()
        }
    }
}

#[cfg(test)]
#[derive(Debug)]
enum GateTrace {
    Hint(ActivationTarget, Range<u64>, ActivationRows),
    Activate(ActivationTarget, Range<u64>, ActivationRows),
}

struct PushHost<'a> {
    scheduler: Option<&'a Arc<Scheduler>>,
    control: &'a mut PushControlState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WaitToken {
    generation: usize,
    epoch: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PipelineStage {
    pipeline: PipelineId,
    stage: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PipelineWaitToken {
    generation: usize,
    continuation_epoch: usize,
    pipeline: PipelineId,
    stage: usize,
}

struct PipelineContinuation {
    token: PipelineWaitToken,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputWaitToken {
    generation: usize,
    epoch: usize,
}

fn current_pipeline_wait(
    generation: usize,
    waiting: &HashMap<PipelineStage, PipelineContinuation>,
    token: PipelineWaitToken,
) -> bool {
    token.generation == generation
        && waiting
            .get(&PipelineStage {
                pipeline: token.pipeline,
                stage: token.stage,
            })
            .is_some_and(|continuation| continuation.token == token)
}

fn refine_demand_spans(
    spans: &mut Vec<(Range<u64>, vortex_mask::Mask)>,
    scratch: &mut Vec<(Range<u64>, vortex_mask::Mask)>,
    coverage: Range<u64>,
    selection: vortex_mask::Mask,
) -> bool {
    // Gate fragments normally arrive once, in monotonically increasing coverage order. Keep that
    // path to one capacity-preserving push; the overlap machinery below is only for refinements.
    if spans
        .last()
        .is_none_or(|(previous, _)| previous.end <= coverage.start)
    {
        spans.push((coverage, selection));
        return true;
    }

    for (previous_range, previous) in spans.iter() {
        let start = coverage.start.max(previous_range.start);
        let end = coverage.end.min(previous_range.end);
        if start >= end {
            continue;
        }
        let (Ok(new_start), Ok(new_end), Ok(old_start), Ok(old_end)) = (
            usize::try_from(start - coverage.start),
            usize::try_from(end - coverage.start),
            usize::try_from(start - previous_range.start),
            usize::try_from(end - previous_range.start),
        ) else {
            return false;
        };
        if !selection
            .slice(new_start..new_end)
            .bitand_not(&previous.slice(old_start..old_end))
            .all_false()
        {
            return false;
        }
    }

    scratch.clear();
    scratch.reserve(spans.len() + 1);
    for (previous_range, previous) in spans.iter() {
        if previous_range.end <= coverage.start || previous_range.start >= coverage.end {
            scratch.push((previous_range.clone(), previous.clone()));
            continue;
        }
        if previous_range.start < coverage.start {
            let Ok(end) = usize::try_from(coverage.start - previous_range.start) else {
                scratch.clear();
                return false;
            };
            scratch.push((previous_range.start..coverage.start, previous.slice(..end)));
        }
        if previous_range.end > coverage.end {
            let Ok(start) = usize::try_from(coverage.end - previous_range.start) else {
                scratch.clear();
                return false;
            };
            scratch.push((coverage.end..previous_range.end, previous.slice(start..)));
        }
    }
    scratch.push((coverage, selection));
    scratch.sort_unstable_by_key(|(range, _)| range.start);
    std::mem::swap(spans, scratch);
    scratch.clear();
    true
}

struct LocalMorsel<'a> {
    arena: &'a mut Arena,
    physical: Option<PhysicalRuntime>,
    io: IoPlane,
    phase: TaskPhase,
    index: usize,
    range: Range<u64>,
    active: bool,
    generation: usize,
    wait_epoch: usize,
    waiting: Option<WaitToken>,
    pipeline_waiting: HashMap<PipelineStage, PipelineContinuation>,
    push_control: PushControlState,
    push_root_done: bool,
    credit_waiting: Option<OutputWaitToken>,
    pull_completion: Option<(usize, ArrayRef)>,
    morsel_io_uses_start: u64,
    morsel_io_requests_start: u64,
    morsel_io_batches_start: u64,
    morsel_io_blocks_start: u64,
    stats: ScanStats,
}

enum RootFragmentPoll {
    Continue,
    Terminal(Option<ArrayRef>),
}

fn accept_root_fragment(
    parts: &mut Vec<ArrayRef>,
    coverage_end: &mut u64,
    morsel: &Range<u64>,
    batch: PushBatch,
    terminal: bool,
) -> VortexResult<RootFragmentPoll> {
    if batch.coverage.start != *coverage_end {
        return Err(vortex_err!(
            "push root produced non-contiguous coverage {:?}; expected {}",
            batch.coverage,
            *coverage_end
        ));
    }
    if batch.coverage.end > morsel.end {
        return Err(vortex_err!(
            "push root coverage {:?} exceeds morsel {morsel:?}",
            batch.coverage
        ));
    }
    *coverage_end = batch.coverage.end;
    let array = batch.value.into_array()?;
    if !array.is_empty() {
        parts.push(array);
    }
    if !terminal {
        return Ok(RootFragmentPoll::Continue);
    }
    if *coverage_end != morsel.end {
        return Err(vortex_err!(
            "terminal push root coverage ended at {}, expected {}",
            *coverage_end,
            morsel.end
        ));
    }
    let array = match parts.len() {
        0 => None,
        1 => Some(
            parts
                .pop()
                .ok_or_else(|| vortex_err!("root morsel part disappeared"))?,
        ),
        _ => {
            // Moving into a separate chunk vector retains the runtime-local accumulation capacity.
            let mut chunks = Vec::with_capacity(parts.len());
            chunks.append(parts);
            let dtype = chunks
                .first()
                .ok_or_else(|| vortex_err!("root morsel chunks disappeared"))?
                .dtype()
                .clone();
            Some(ChunkedArray::try_new(chunks, dtype)?.into_array())
        }
    };
    Ok(RootFragmentPoll::Terminal(array))
}

struct PendingDemandHint {
    due_transition: u64,
    target: DemandTarget,
    coverage: Range<u64>,
    selection: vortex_mask::Mask,
}

impl PushHost<'_> {
    fn drain_gate_io(&self, limit: usize) -> VortexResult<usize> {
        self.scheduler
            .map_or(Ok(0), |scheduler| scheduler.drain_ready_io_limit(limit))
    }

    fn observe_hint(
        &mut self,
        stats: &mut ScanStats,
        target: DemandTarget,
        coverage: Range<u64>,
        selection: vortex_mask::Mask,
        reuse_source_matches: bool,
    ) -> VortexResult<()> {
        let scheduler = self
            .scheduler
            .ok_or_else(|| vortex_err!("gate control requires a scheduler host"))?;
        let Ok(coverage_len) = usize::try_from(coverage.end.saturating_sub(coverage.start)) else {
            stats.demand_hints_dropped += 1;
            return Ok(());
        };
        if coverage.start >= coverage.end || selection.len() != coverage_len {
            stats.demand_hints_dropped += 1;
            return Ok(());
        }
        let spans = self
            .control
            .demand_state
            .entry(target)
            .or_insert_with(|| Vec::with_capacity(8));
        let previous_span_count = spans.len();
        if !refine_demand_spans(
            spans,
            &mut self.control.demand_refine_scratch,
            coverage.clone(),
            selection.clone(),
        ) {
            stats.demand_hints_dropped += 1;
            return Ok(());
        }
        self.control.demand_state_live = self
            .control
            .demand_state_live
            .saturating_sub(previous_span_count)
            .saturating_add(spans.len());
        stats.demand_state_live_max = stats
            .demand_state_live_max
            .max(u64::try_from(self.control.demand_state_live).unwrap_or(u64::MAX));
        if !reuse_source_matches {
            scheduler.run.plan.overlapping_demand_source_indices(
                target,
                &coverage,
                &mut self.control.source_matches,
            );
            self.control.source_nodes.clear();
            self.control.source_nodes.extend(
                self.control
                    .source_matches
                    .iter()
                    .map(|&index| scheduler.run.plan.sources()[index].node),
            );
        }
        observe_prebound_demand(
            stats,
            coverage,
            selection,
            &self.control.source_nodes,
            &self.control.projection_sources,
            &self.control.demand_sources,
            &mut self.control.demand_observation,
        );
        Ok(())
    }

    fn publish_hint(
        &mut self,
        stats: &mut ScanStats,
        target: DemandTarget,
        coverage: Range<u64>,
        selection: vortex_mask::Mask,
        completed: u64,
        reuse_source_matches: bool,
    ) -> VortexResult<()> {
        let scheduler = self
            .scheduler
            .ok_or_else(|| vortex_err!("gate control requires a scheduler host"))?;
        stats.demand_hints_emitted += 1;
        match scheduler.run.demand_hints {
            DemandHintDelivery::Immediate | DemandHintDelivery::Delayed(0) => {
                self.observe_hint(stats, target, coverage, selection, reuse_source_matches)?;
            }
            DemandHintDelivery::Disabled => stats.demand_hints_dropped += 1,
            DemandHintDelivery::Delayed(delay) => {
                self.control.demand_hints.push_back(PendingDemandHint {
                    due_transition: completed
                        .saturating_add(u64::try_from(delay).unwrap_or(u64::MAX)),
                    target,
                    coverage,
                    selection,
                });
                self.control.has_delayed_hints = true;
            }
        }
        Ok(())
    }

    fn flush_due(&mut self, stats: &mut ScanStats, completed: u64) -> VortexResult<()> {
        if !self.control.has_delayed_hints {
            return Ok(());
        }
        while self
            .control
            .demand_hints
            .front()
            .is_some_and(|hint| hint.due_transition <= completed)
        {
            let Some(hint) = self.control.demand_hints.pop_front() else {
                break;
            };
            self.observe_hint(stats, hint.target, hint.coverage, hint.selection, false)?;
        }
        self.control.has_delayed_hints = !self.control.demand_hints.is_empty();
        Ok(())
    }

    fn gate(
        &mut self,
        stats: &mut ScanStats,
        target: ActivationTarget,
        coverage: Range<u64>,
        rows: ActivationRows,
        completed: u64,
    ) -> VortexResult<()> {
        if target == ActivationTarget::Projection && rows.logical() != rows.materialized() {
            return Err(vortex_err!(
                "projection activation must materialize exactly its selected rows"
            ));
        }
        #[cfg(test)]
        if self.scheduler.is_none() {
            self.control
                .gate_trace
                .push(GateTrace::Hint(target, coverage.clone(), rows.clone()));
            self.control
                .gate_trace
                .push(GateTrace::Activate(target, coverage, rows));
            stats.demand_hints_emitted += 1;
            return Ok(());
        }
        let demand_target = match target {
            ActivationTarget::PredicateSlot(slot) => DemandTarget::PredicateSlot(slot),
            ActivationTarget::Projection => DemandTarget::Projection,
        };
        {
            let plan = &self
                .scheduler
                .ok_or_else(|| vortex_err!("gate control requires a scheduler host"))?
                .run
                .plan;
            // Demand and activation target the same compiled source group. Resolve it once, then
            // publish the side information before making any source runnable.
            plan.overlapping_source_indices(target, &coverage, &mut self.control.source_matches);
            self.control.source_nodes.clear();
            self.control.source_nodes.extend(
                self.control
                    .source_matches
                    .iter()
                    .map(|&index| plan.sources()[index].node),
            );
        }
        self.publish_hint(
            stats,
            demand_target,
            coverage.clone(),
            rows.logical().clone(),
            completed,
            true,
        )?;
        activate_pending_sources_into(
            &mut self.control.projection_sources,
            &mut self.control.source_activation,
            &self.control.source_nodes,
            target,
            coverage,
            &rows,
            &mut self.control.ready_sources,
        )?;
        stats.push_source_activations +=
            u64::try_from(self.control.ready_sources.len()).unwrap_or(u64::MAX);
        Ok(())
    }
}

struct IoWork {
    queued: AtomicBool,
    running: AtomicBool,
    required: AtomicBool,
    ready: Mutex<Vec<usize>>,
    scheduled: Vec<AtomicBool>,
    completed: Vec<AtomicBool>,
    reads: Vec<IoRead>,
}

fn new_io_work(read: IoRead, required: bool) -> Arc<IoWork> {
    Arc::new(IoWork {
        queued: AtomicBool::new(false),
        running: AtomicBool::new(false),
        required: AtomicBool::new(required),
        ready: Mutex::new(vec![0]),
        scheduled: vec![AtomicBool::new(true)],
        completed: vec![AtomicBool::new(false)],
        reads: vec![read],
    })
}

struct OutputCreditState {
    rows: usize,
    bytes: u64,
    rows_max: usize,
    bytes_max: u64,
    blocks: u64,
    head_bypass: bool,
    waiters: Vec<(Sender<WorkerSignal>, OutputWaitToken)>,
}

struct OutputCredits {
    max_rows: usize,
    max_bytes: u64,
    state: Mutex<OutputCreditState>,
}

struct CreditedBatch {
    array: Option<ArrayRef>,
    rows: usize,
    bytes: u64,
    credits: Arc<OutputCredits>,
    head_bypass: bool,
    released: bool,
}

type BufferedOutput = (usize, u64, ArrayRef, bool);

fn take_ordered_output(results: &mut Vec<BufferedOutput>, index: usize) -> Option<BufferedOutput> {
    let position = results
        .iter()
        .enumerate()
        .filter(|(_, result)| result.0 == index)
        .min_by_key(|(_, result)| result.1)
        .map(|(position, _)| position)?;
    Some(results.swap_remove(position))
}

impl CreditedBatch {
    fn receive(mut self) -> ArrayRef {
        self.credits
            .release(self.rows, self.bytes, self.head_bypass);
        self.released = true;
        match self.array.take() {
            Some(array) => array,
            None => unreachable!("a credited batch can only be received once"),
        }
    }
}

impl Drop for CreditedBatch {
    fn drop(&mut self) {
        if !self.released {
            self.credits
                .release(self.rows, self.bytes, self.head_bypass);
            self.released = true;
        }
    }
}

struct OutputOrder {
    completed: Vec<bool>,
    next: usize,
    retired: usize,
}

fn claim_lookahead_extension(
    cursor: &AtomicUsize,
    target: usize,
    limit: usize,
) -> Option<Range<usize>> {
    let target = target.min(limit);
    let previous = cursor.fetch_max(target, Ordering::AcqRel);
    (target > previous).then_some(previous..target)
}

fn assignment_lookahead_target(index: usize, workers: usize, lookahead: usize) -> usize {
    index
        .saturating_add(1)
        .saturating_add(workers)
        .saturating_add(lookahead)
}

fn drain_queued_io(mut try_run: impl FnMut() -> VortexResult<bool>) -> VortexResult<usize> {
    let mut drained = 0;
    while try_run()? {
        drained += 1;
    }
    Ok(drained)
}

fn drain_queued_io_limit(
    limit: usize,
    mut try_run: impl FnMut() -> VortexResult<bool>,
) -> VortexResult<usize> {
    let mut drained = 0;
    while drained < limit && try_run()? {
        drained += 1;
    }
    Ok(drained)
}

#[derive(Clone)]
enum WorkerSignal {
    Wake(WaitToken),
    PushWake(PipelineWaitToken),
    Credit(OutputWaitToken),
    Shutdown,
}

struct Scheduler {
    run: Arc<WorkerRun>,
    urgent_tx: Sender<Arc<IoWork>>,
    urgent_rx: Receiver<Arc<IoWork>>,
    ready_tx: Sender<Arc<IoWork>>,
    ready_rx: Receiver<Arc<IoWork>>,
    worker_tx: Vec<Sender<WorkerSignal>>,
    io_work: Mutex<HashMap<IoKey, Arc<IoWork>>>,
    results: Mutex<Vec<BufferedOutput>>,
    output_order: Mutex<OutputOrder>,
    output_changed: Condvar,
    output_credits: Arc<OutputCredits>,
    error: Mutex<Option<VortexError>>,
    next_morsel: AtomicUsize,
    remaining: AtomicUsize,
    stopped: AtomicBool,
    io_bytes: AtomicU64,
    io_waits: AtomicU64,
    io_wait_nanos: AtomicU64,
    lookahead_requests: AtomicU64,
    lookahead_batches: AtomicU64,
    lookahead_cursor: AtomicUsize,
    lookahead_refills: AtomicU64,
}

enum WorkerMessage {
    Run {
        scheduler: Arc<Scheduler>,
        worker: usize,
        signals: Receiver<WorkerSignal>,
        done: mpsc::Sender<ScanStats>,
    },
    Shutdown,
}

struct Worker {
    messages: mpsc::Sender<WorkerMessage>,
    handle: Option<JoinHandle<()>>,
}

/// A set of ready morsel workers whose lifecycle is outside a timed scan.
struct MorselWorkerPool {
    workers: Vec<Worker>,
}

impl MorselWorkerPool {
    fn new(threads: usize, plan: Arc<ExecPlan>) -> VortexResult<Self> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let mut workers = Vec::with_capacity(threads);

        for idx in 0..threads {
            let (message_tx, message_rx) = mpsc::channel();
            let ready_tx = ready_tx.clone();
            let plan = Arc::clone(&plan);
            let handle = std::thread::Builder::new()
                .name(format!("vortex-morsel-{idx}"))
                .spawn(move || {
                    let mut arena = plan.instantiate();
                    if ready_tx.send(()).is_err() {
                        return;
                    }
                    while let Ok(message) = message_rx.recv() {
                        match message {
                            WorkerMessage::Run {
                                scheduler,
                                worker,
                                signals,
                                done,
                            } => {
                                let stats = scheduler.worker_loop(worker, &signals, &mut arena);
                                let _ = done.send(stats);
                            }
                            WorkerMessage::Shutdown => break,
                        }
                    }
                })
                .map_err(|err| vortex_err!("failed to spawn morsel worker: {err}"))?;
            workers.push(Worker {
                messages: message_tx,
                handle: Some(handle),
            });
        }
        drop(ready_tx);

        for _ in 0..threads {
            ready_rx
                .recv()
                .map_err(|err| vortex_err!("morsel worker failed to start: {err}"))?;
        }
        Ok(Self { workers })
    }

    fn run(
        &self,
        scheduler: Arc<Scheduler>,
        signals: Vec<Receiver<WorkerSignal>>,
    ) -> VortexResult<Vec<ScanStats>> {
        let (done_tx, done_rx) = mpsc::channel();
        for (worker, (thread, signals)) in self.workers.iter().zip(signals).enumerate() {
            thread
                .messages
                .send(WorkerMessage::Run {
                    scheduler: Arc::clone(&scheduler),
                    worker,
                    signals,
                    done: done_tx.clone(),
                })
                .map_err(|err| vortex_err!("failed to dispatch morsel worker: {err}"))?;
        }
        drop(done_tx);

        let mut stats = Vec::with_capacity(self.workers.len());
        for _ in 0..self.workers.len() {
            stats.push(
                done_rx
                    .recv()
                    .map_err(|err| vortex_err!("morsel worker stopped early: {err}"))?,
            );
        }
        Ok(stats)
    }
}

impl Drop for MorselWorkerPool {
    fn drop(&mut self) {
        for worker in &self.workers {
            drop(worker.messages.send(WorkerMessage::Shutdown));
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                drop(handle.join());
            }
        }
    }
}

struct IoWake {
    scheduler: Weak<Scheduler>,
    work: Weak<IoWork>,
    index: usize,
}

struct TaskWake {
    tx: Sender<WorkerSignal>,
    signal: WorkerSignal,
}

impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        drop(self.tx.send(self.signal.clone()));
    }

    fn wake_by_ref(self: &Arc<Self>) {
        drop(self.tx.send(self.signal.clone()));
    }
}

impl Wake for IoWake {
    fn wake(self: Arc<Self>) {
        self.enqueue();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.enqueue();
    }
}

impl OutputCredits {
    fn new(max_rows: usize, max_bytes: u64) -> Arc<Self> {
        Arc::new(Self {
            max_rows,
            max_bytes,
            state: Mutex::new(OutputCreditState {
                rows: 0,
                bytes: 0,
                rows_max: 0,
                bytes_max: 0,
                blocks: 0,
                head_bypass: false,
                waiters: Vec::new(),
            }),
        })
    }

    fn acquire_or_park(
        &self,
        rows: usize,
        bytes: u64,
        ordered_head: bool,
        tx: Sender<WorkerSignal>,
        token: OutputWaitToken,
    ) -> Option<bool> {
        let mut state = self.state.lock();
        let empty = state.rows == 0 && state.bytes == 0;
        let exceeds_capacity = !empty
            && (state.rows.saturating_add(rows) > self.max_rows
                || state.bytes.saturating_add(bytes) > self.max_bytes);
        let head_bypass = exceeds_capacity && ordered_head && !state.head_bypass;
        if exceeds_capacity && !head_bypass {
            state.blocks += 1;
            state.waiters.push((tx, token));
            return None;
        }
        state.rows = state.rows.saturating_add(rows);
        state.bytes = state.bytes.saturating_add(bytes);
        state.rows_max = state.rows_max.max(state.rows);
        state.bytes_max = state.bytes_max.max(state.bytes);
        state.head_bypass |= head_bypass;
        Some(head_bypass)
    }

    fn release(&self, rows: usize, bytes: u64, head_bypass: bool) {
        let waiters = {
            let mut state = self.state.lock();
            state.rows = state.rows.saturating_sub(rows);
            state.bytes = state.bytes.saturating_sub(bytes);
            if head_bypass {
                state.head_bypass = false;
            }
            std::mem::take(&mut state.waiters)
        };
        for (tx, token) in waiters {
            drop(tx.send(WorkerSignal::Credit(token)));
        }
    }

    fn wake_waiters(&self) {
        let waiters = std::mem::take(&mut self.state.lock().waiters);
        for (tx, token) in waiters {
            drop(tx.send(WorkerSignal::Credit(token)));
        }
    }
}

impl IoWake {
    fn enqueue(&self) {
        let (Some(scheduler), Some(work)) = (self.scheduler.upgrade(), self.work.upgrade()) else {
            return;
        };
        if work.completed[self.index].load(Ordering::Acquire)
            || work.scheduled[self.index].swap(true, Ordering::AcqRel)
        {
            return;
        }
        work.ready.lock().push(self.index);
        scheduler.enqueue_io(work);
    }
}

impl Scheduler {
    fn acquire_output(
        &self,
        index: usize,
        rows: usize,
        bytes: u64,
        tx: Sender<WorkerSignal>,
        token: OutputWaitToken,
    ) -> Option<bool> {
        let order = self.output_order.lock();
        self.output_credits
            .acquire_or_park(rows, bytes, order.next == index, tx, token)
    }

    fn new(run: Arc<WorkerRun>, workers: usize) -> (Arc<Self>, Vec<Receiver<WorkerSignal>>) {
        let (urgent_tx, urgent_rx) = unbounded();
        let (ready_tx, ready_rx) = unbounded();
        let mut worker_tx = Vec::with_capacity(workers);
        let mut worker_rx = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) = unbounded();
            worker_tx.push(tx);
            worker_rx.push(rx);
        }
        let morsel_count = run.morsels.len();
        let output_rows = run.output_rows;
        let output_bytes = run.output_bytes;
        let scheduler = Arc::new(Self {
            remaining: AtomicUsize::new(morsel_count),
            run,
            urgent_tx,
            urgent_rx,
            ready_tx,
            ready_rx,
            worker_tx,
            io_work: Mutex::new(HashMap::default()),
            results: Mutex::new(Vec::new()),
            output_order: Mutex::new(OutputOrder {
                completed: vec![false; morsel_count],
                next: 0,
                retired: 0,
            }),
            output_changed: Condvar::new(),
            output_credits: OutputCredits::new(output_rows, output_bytes),
            error: Mutex::new(None),
            next_morsel: AtomicUsize::new(0),
            stopped: AtomicBool::new(false),
            io_bytes: AtomicU64::new(0),
            io_waits: AtomicU64::new(0),
            io_wait_nanos: AtomicU64::new(0),
            lookahead_requests: AtomicU64::new(0),
            lookahead_batches: AtomicU64::new(0),
            lookahead_cursor: AtomicUsize::new(0),
            lookahead_refills: AtomicU64::new(0),
        });
        if scheduler.run.morsels.is_empty() {
            scheduler.stop();
        }
        (scheduler, worker_rx)
    }

    fn issue_batch(&self, reads: &[IoRead]) -> bool {
        let issued = reads.iter().filter(|read| self.run.io.issue(read)).count();
        if issued == 0 {
            return false;
        }
        self.lookahead_requests
            .fetch_add(u64::try_from(issued).unwrap_or(u64::MAX), Ordering::Relaxed);
        self.lookahead_batches.fetch_add(1, Ordering::Relaxed);
        true
    }

    fn submit_reads(self: &Arc<Self>, mut reads: Vec<IoRead>) {
        reads.sort_unstable_by_key(|read| match read.key() {
            IoKey::Segment(id) => *id,
        });
        let (required, speculative): (Vec<_>, Vec<_>) = reads
            .into_iter()
            .partition(|read| read.priority() == IoPriority::Required);
        self.issue_batch(&speculative);
        let eager_required =
            self.run.io.prefers_background_reads() || self.run.io.nowait_unsupported();
        if eager_required {
            self.issue_batch(&required);
        }
        self.submit_io_batch(required, true, eager_required);
        self.submit_io_batch(speculative, false, true);
    }

    fn submit_exact_lookahead(self: &Arc<Self>) {
        if !self.run.io.prefers_background_reads() {
            return;
        }
        let end = if self.run.plan.has_filter() {
            (self.worker_tx.len() + self.run.lookahead_morsels).min(self.run.morsels.len())
        } else {
            self.run.morsels.len()
        };
        self.advance_lookahead(end);
    }

    fn submit_lookahead_slice(self: &Arc<Self>, start: usize, end: usize) {
        const BATCH_READS: usize = 64;

        if start >= end {
            return;
        }
        let ranges = &self.run.morsels[start..end];
        let filtered = self.run.plan.has_filter();
        let mut admission: HashMap<IoKey, bool> = HashMap::default();
        let Some(coverage) = ranges
            .first()
            .zip(ranges.last())
            .map(|(first, last)| first.start..last.end)
        else {
            return;
        };
        let mut source_indices = Vec::new();
        self.run
            .plan
            .overlapping_all_source_indices(&coverage, &mut source_indices);
        for source_index in source_indices {
            let Some((_, key, _, role)) = self.run.plan.source_io_use_at(source_index) else {
                continue;
            };
            let eager = !filtered
                || matches!(
                    role,
                    SourceRole::Predicate {
                        slot: 0,
                        mode: crate::nodes::ConjunctMode::Cascade,
                    } | SourceRole::Predicate {
                        mode: crate::nodes::ConjunctMode::Parallel,
                        ..
                    }
                );
            admission
                .entry(key)
                .and_modify(|admitted| *admitted |= eager)
                .or_insert(eager);
        }
        let priority = if filtered {
            IoPriority::Speculative
        } else {
            IoPriority::Required
        };
        let reads = self
            .run
            .io
            .register_reads(admission.keys().copied(), priority);
        for batch in reads.chunks(BATCH_READS) {
            let (eager, deferred): (Vec<_>, Vec<_>) = batch
                .iter()
                .cloned()
                .partition(|read| admission.get(&read.key()).copied().unwrap_or(false));
            self.submit_reads(eager);
            self.submit_io_batch(deferred, false, false);
        }
    }

    fn advance_lookahead(self: &Arc<Self>, target: usize) {
        if !self.run.io.prefers_background_reads() {
            return;
        }
        let Some(extension) =
            claim_lookahead_extension(&self.lookahead_cursor, target, self.run.morsels.len())
        else {
            return;
        };
        if extension.start > 0 {
            self.lookahead_refills.fetch_add(1, Ordering::Relaxed);
        }
        self.submit_lookahead_slice(extension.start, extension.end);
    }

    fn refill_lookahead(self: &Arc<Self>, retired: usize) {
        if !self.run.plan.has_filter() {
            return;
        }
        let target = (retired + self.worker_tx.len() + self.run.lookahead_morsels)
            .min(self.run.morsels.len());
        self.advance_lookahead(target);
    }

    fn submit_io_batch(self: &Arc<Self>, reads: Vec<IoRead>, required: bool, enqueue: bool) {
        if reads.is_empty() {
            return;
        }
        for read in reads {
            let key = read.key();
            let candidate = new_io_work(read, required);
            let work = Arc::clone(self.io_work.lock().entry(key).or_insert(candidate));
            if required {
                work.required.store(true, Ordering::Release);
            }
            if enqueue {
                self.enqueue_io(work);
            }
        }
    }

    fn enqueue_io(&self, work: Arc<IoWork>) {
        if self.stopped.load(Ordering::Acquire) || work.queued.swap(true, Ordering::AcqRel) {
            return;
        }
        let tx = if work.required.load(Ordering::Acquire) {
            &self.urgent_tx
        } else {
            &self.ready_tx
        };
        drop(tx.send(work));
    }

    fn promote(&self, key: IoKey) {
        let Some(work) = self.io_work.lock().get(&key).cloned() else {
            return;
        };
        self.issue_batch(&work.reads);
        for (index, read) in work.reads.iter().enumerate() {
            if read.is_failed()
                && work.completed[index].swap(false, Ordering::AcqRel)
                && !work.scheduled[index].swap(true, Ordering::AcqRel)
            {
                work.ready.lock().push(index);
            }
        }
        work.required.store(true, Ordering::Release);
        if work.queued.load(Ordering::Acquire) {
            // Leave the normal-queue copy in place and add an urgent copy. The first receiver
            // clears `queued` and owns the poll; the other copy is then a cheap stale dequeue.
            drop(self.urgent_tx.send(work));
        } else {
            self.enqueue_io(work);
        }
    }

    fn enqueue_required_io(&self) {
        let work = self
            .io_work
            .lock()
            .values()
            .filter(|work| {
                work.required.load(Ordering::Acquire)
                    && work
                        .reads
                        .iter()
                        .any(|read| read.priority() == IoPriority::Required)
            })
            .cloned()
            .collect::<Vec<_>>();
        for work in work {
            self.issue_batch(&work.reads);
            self.enqueue_io(work);
        }
    }

    fn park(&self, worker: usize, signal: WorkerSignal, waits: &WaitSet) -> VortexResult<bool> {
        if waits.is_empty() {
            return Err(vortex_err!(
                "execution blocked without naming an exact dependency"
            ));
        }

        let mut parked = false;
        let tx = self
            .worker_tx
            .get(worker)
            .cloned()
            .ok_or_else(|| vortex_err!("blocked on an unknown worker"))?;
        for wait in waits.waits() {
            let Wait::Io(ticket) = wait;
            let read = self
                .run
                .io
                .read(*ticket)
                .ok_or_else(|| vortex_err!("blocked on an unknown IO ticket"))?;
            read.promote();
            self.promote(read.key());
            let waker = Waker::from(Arc::new(TaskWake {
                tx: tx.clone(),
                signal: signal.clone(),
            }));
            if read.park(waker) {
                parked = true;
            }
        }
        if parked {
            // Pull evaluation can expose required dependencies one at a time. Once it blocks,
            // drive every already-admitted required read independently so coupled futures can
            // make progress without promoting any speculative lookahead work.
            self.enqueue_required_io();
        }
        Ok(parked)
    }

    fn run_io(self: &Arc<Self>, work: Arc<IoWork>) -> VortexResult<()> {
        if work.running.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let ready = std::mem::take(&mut *work.ready.lock());
        for index in ready {
            work.scheduled[index].store(false, Ordering::Release);
            if work.completed[index].load(Ordering::Acquire) {
                continue;
            }
            let waker = Waker::from(Arc::new(IoWake {
                scheduler: Arc::downgrade(self),
                work: Arc::downgrade(&work),
                index,
            }));
            match work.reads[index].poll(&waker)? {
                IoReadPoll::Pending => {
                    self.io_waits.fetch_add(1, Ordering::Relaxed);
                }
                IoReadPoll::Ready { bytes, wait_time } => {
                    if work.completed[index].swap(true, Ordering::AcqRel) {
                        continue;
                    }
                    self.io_bytes
                        .fetch_add(u64::try_from(bytes).unwrap_or(u64::MAX), Ordering::Relaxed);
                    self.io_wait_nanos.fetch_add(
                        u64::try_from(wait_time.as_nanos()).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                }
                IoReadPoll::AlreadyReady => {
                    work.completed[index].store(true, Ordering::Release);
                }
                IoReadPoll::Failed(error) => {
                    work.completed[index].store(true, Ordering::Release);
                    if work.required.load(Ordering::Acquire) {
                        return Err(vortex_err!("segment read failed: {error}"));
                    }
                }
            }
        }
        work.queued.store(false, Ordering::Release);
        work.running.store(false, Ordering::Release);
        if !work.ready.lock().is_empty() {
            self.enqueue_io(work);
        }
        Ok(())
    }

    fn try_run_io(self: &Arc<Self>) -> VortexResult<bool> {
        let work = self
            .urgent_rx
            .try_recv()
            .or_else(|_| self.ready_rx.try_recv());
        let Ok(work) = work else {
            return Ok(false);
        };
        self.run_io(work)?;
        Ok(true)
    }

    fn drain_ready_io(self: &Arc<Self>) -> VortexResult<()> {
        drain_queued_io(|| self.try_run_io()).map(|_| ())
    }

    fn drain_ready_io_limit(self: &Arc<Self>, limit: usize) -> VortexResult<usize> {
        drain_queued_io_limit(limit, || self.try_run_io())
    }

    fn drive_external_idle(
        self: &Arc<Self>,
        driver: &ExternalDriver,
        signals: &Receiver<WorkerSignal>,
        morsel: &mut LocalMorsel<'_>,
    ) -> Option<bool> {
        driver();
        let wake = morsel.handle_external_signal(signals.try_recv())?;
        if let Err(err) = self.try_run_io() {
            self.fail(err);
            return None;
        }
        Some(wake)
    }

    fn worker_loop(
        self: &Arc<Self>,
        worker: usize,
        signals: &Receiver<WorkerSignal>,
        arena: &mut Arena,
    ) -> ScanStats {
        let mut morsel = LocalMorsel::new(&self.run, arena);
        let mut runnable = morsel.assign_next(self);

        loop {
            if self.stopped.load(Ordering::Acquire) {
                break;
            }

            if runnable {
                let mut completion_boundary = false;
                let poll = match morsel.pull_completion.take() {
                    Some((index, batch)) => Ok(LocalPoll::Complete {
                        index,
                        batch: Some(batch),
                    }),
                    None => morsel.run(self),
                };
                match poll {
                    Ok(LocalPoll::Runnable) => runnable = true,
                    Ok(LocalPoll::Idle) => runnable = false,
                    Ok(LocalPoll::Blocked(waits)) => {
                        let token = morsel.next_wait_token();
                        match self.park(worker, WorkerSignal::Wake(token), &waits) {
                            Ok(true) => {
                                morsel.waiting = Some(token);
                                runnable = false;
                            }
                            Ok(false) => runnable = true,
                            Err(err) => {
                                self.fail(err);
                                break;
                            }
                        }
                    }
                    Ok(LocalPoll::PushBlocked {
                        pipeline,
                        stage,
                        waits,
                    }) => {
                        let token = morsel.next_pipeline_wait_token(pipeline, stage);
                        match self.park(worker, WorkerSignal::PushWake(token), &waits) {
                            Ok(true) => {
                                morsel.pipeline_waiting.insert(
                                    PipelineStage { pipeline, stage },
                                    PipelineContinuation { token },
                                );
                                runnable = morsel
                                    .physical
                                    .as_ref()
                                    .is_some_and(|runtime| !runtime.is_idle());
                            }
                            Ok(false) => {
                                if let Some(runtime) = morsel.physical.as_mut() {
                                    runtime.enqueue_resume(pipeline, stage);
                                    morsel.stats.push_pipeline_boundary_resumes += 1;
                                    runnable = true;
                                } else {
                                    self.fail(vortex_err!(
                                        "pipeline wait lost its physical runtime"
                                    ));
                                    break;
                                }
                            }
                            Err(err) => {
                                self.fail(err);
                                break;
                            }
                        }
                    }
                    Ok(LocalPoll::Complete { index, batch }) => {
                        if let Some(batch) = batch {
                            let rows = batch.len();
                            let bytes = batch.nbytes();
                            let token = morsel.next_output_wait_token();
                            if let Some(head_bypass) = self.acquire_output(
                                index,
                                rows,
                                bytes,
                                self.worker_tx[worker].clone(),
                                token,
                            ) {
                                self.emit(index, morsel.range.start, batch, head_bypass);
                                self.complete(index);
                                runnable = !self.stopped.load(Ordering::Acquire)
                                    && morsel.assign_next(self);
                                completion_boundary = !self.stopped.load(Ordering::Acquire);
                            } else {
                                morsel.pull_completion = Some((index, batch));
                                morsel.credit_waiting = Some(token);
                                runnable = false;
                            }
                        } else {
                            self.complete(index);
                            runnable =
                                !self.stopped.load(Ordering::Acquire) && morsel.assign_next(self);
                            completion_boundary = !self.stopped.load(Ordering::Acquire);
                        }
                    }
                    Err(err) => {
                        self.fail(err);
                        break;
                    }
                }

                let io_result = if completion_boundary {
                    self.drain_ready_io().map(|()| true)
                } else {
                    self.try_run_io()
                };
                if let Err(err) = io_result {
                    self.fail(err);
                    break;
                }
                continue;
            }

            if let Some(driver) = &self.run.external_driver {
                let Some(wake) = self.drive_external_idle(driver, signals, &mut morsel) else {
                    break;
                };
                runnable = wake;
                continue;
            }

            crossbeam_channel::select_biased! {
                recv(signals) -> signal => match signal {
                    Ok(WorkerSignal::Wake(token)) if morsel.waiting == Some(token) => {
                        morsel.waiting = None;
                        runnable = true;
                    }
                    Ok(WorkerSignal::Wake(_)) => {}
                    Ok(WorkerSignal::PushWake(token)) => {
                        runnable = morsel.wake_pipeline(token);
                    }
                    Ok(WorkerSignal::Credit(token)) if morsel.credit_waiting == Some(token) => {
                        morsel.credit_waiting = None;
                        runnable = true;
                    }
                    Ok(WorkerSignal::Credit(_)) => {
                        morsel.stats.push_stale_wakes += 1;
                    }
                    Ok(WorkerSignal::Shutdown) | Err(_) => break,
                },
                recv(self.urgent_rx) -> work => match work {
                    Ok(work) => if let Err(err) = self.run_io(work) {
                        self.fail(err);
                        break;
                    },
                    Err(_) => break,
                },
                recv(self.ready_rx) -> work => match work {
                    Ok(work) => if let Err(err) = self.run_io(work) {
                        self.fail(err);
                        break;
                    },
                    Err(_) => break,
                },
            }
        }
        if morsel.active {
            if let Some(physical) = morsel.physical.as_mut() {
                physical.reset(morsel.range.clone());
            }
            retire_morsel(
                morsel.arena,
                self.run.plan.root(),
                &self.run.cells,
                &mut morsel.stats,
            );
            morsel.io.clear();
            morsel.active = false;
        }
        morsel.restore_physical();
        morsel.stats
    }

    fn complete(self: &Arc<Self>, index: usize) {
        let retired = {
            let mut order = self.output_order.lock();
            order.completed[index] = true;
            while order.completed.get(order.retired).copied().unwrap_or(false) {
                order.retired += 1;
            }
            order.retired
        };
        self.output_changed.notify_all();
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.stop();
        } else {
            self.refill_lookahead(retired);
        }
    }

    fn emit(&self, index: usize, coverage_start: u64, batch: ArrayRef, head_bypass: bool) {
        self.results
            .lock()
            .push((index, coverage_start, batch, head_bypass));
        self.output_changed.notify_all();
    }

    fn stream_ordered(
        self: &Arc<Self>,
        tx: &Sender<CreditedBatch>,
        completion: Option<&CompletionSink>,
    ) {
        let mut completed_batches = Vec::new();
        loop {
            let (index, complete) = {
                let mut order = self.output_order.lock();
                while order.next < order.completed.len()
                    && !order.completed[order.next]
                    && !self
                        .results
                        .lock()
                        .iter()
                        .any(|result| result.0 == order.next)
                    && !self.stopped.load(Ordering::Acquire)
                {
                    self.output_changed.wait(&mut order);
                }
                if order.next >= order.completed.len() {
                    return;
                }
                (order.next, order.completed[order.next])
            };

            let batch = {
                let mut results = self.results.lock();
                take_ordered_output(&mut results, index)
            };
            if let Some((_, _, array, head_bypass)) = batch {
                let item = CreditedBatch {
                    rows: array.len(),
                    bytes: array.nbytes(),
                    array: Some(array),
                    credits: Arc::clone(&self.output_credits),
                    head_bypass,
                    released: false,
                };
                if completion.is_some() {
                    completed_batches.push(item.receive());
                    continue;
                }
                if tx.send(item).is_err() {
                    self.stop();
                    return;
                }
                continue;
            }
            if complete {
                if let Some(completion) = completion {
                    let batch = match completed_batches.len() {
                        0 => Ok(None),
                        1 => Ok(completed_batches.pop()),
                        _ => ChunkedArray::try_new(
                            std::mem::take(&mut completed_batches),
                            self.run.plan.output_dtype().clone(),
                        )
                        .map(IntoArray::into_array)
                        .map(Some),
                    };
                    completion(index, batch);
                }
                let mut order = self.output_order.lock();
                if order.next == index {
                    order.next += 1;
                    self.output_credits.wake_waiters();
                }
                continue;
            }
            if self.stopped.load(Ordering::Acquire) {
                return;
            }
        }
    }

    fn fail(&self, err: VortexError) {
        if !self.stopped.swap(true, Ordering::AcqRel) {
            *self.error.lock() = Some(err);
            self.send_shutdown();
            self.output_changed.notify_all();
        }
    }

    fn stop(&self) {
        if !self.stopped.swap(true, Ordering::AcqRel) {
            self.send_shutdown();
            self.output_changed.notify_all();
        }
    }

    fn send_shutdown(&self) {
        for tx in &self.worker_tx {
            drop(tx.send(WorkerSignal::Shutdown));
        }
    }

    fn finish(&self, worker_stats: Vec<ScanStats>) -> VortexResult<ScanStats> {
        if let Some(err) = self.error.lock().take() {
            return Err(err);
        }

        let mut stats = ScanStats::default();
        for worker in worker_stats {
            stats.merge(&worker);
        }
        stats.io_bytes += self.io_bytes.load(Ordering::Relaxed);
        stats.io_requests += self.lookahead_requests.load(Ordering::Relaxed);
        stats.io_batches += self.lookahead_batches.load(Ordering::Relaxed);
        stats.io_waits = self.io_waits.load(Ordering::Relaxed);
        stats.io_wait_time = Duration::from_nanos(self.io_wait_nanos.load(Ordering::Relaxed));
        stats.lookahead_refills += self.lookahead_refills.load(Ordering::Relaxed);
        let output = self.output_credits.state.lock();
        stats.output_rows_max = stats
            .output_rows_max
            .max(u64::try_from(output.rows_max).unwrap_or(u64::MAX));
        stats.output_bytes_max = stats.output_bytes_max.max(output.bytes_max);
        stats.output_credit_blocks += output.blocks;

        Ok(stats)
    }

    fn take_ordered_batches(&self) -> Vec<ArrayRef> {
        let mut results = std::mem::take(&mut *self.results.lock());
        results.sort_unstable_by_key(|(index, coverage_start, ..)| (*index, *coverage_start));
        results.into_iter().map(|(_, _, array, _)| array).collect()
    }
}

enum LocalPoll {
    Runnable,
    Idle,
    Blocked(WaitSet),
    PushBlocked {
        pipeline: PipelineId,
        stage: usize,
        waits: WaitSet,
    },
    Complete {
        index: usize,
        batch: Option<ArrayRef>,
    },
}

impl<'a> LocalMorsel<'a> {
    fn handle_external_signal(
        &mut self,
        signal: Result<WorkerSignal, crossbeam_channel::TryRecvError>,
    ) -> Option<bool> {
        match signal {
            Ok(WorkerSignal::Wake(token)) if self.waiting == Some(token) => {
                self.waiting = None;
                Some(true)
            }
            Ok(WorkerSignal::Wake(_)) | Err(crossbeam_channel::TryRecvError::Empty) => Some(false),
            Ok(WorkerSignal::PushWake(token)) => Some(self.wake_pipeline(token)),
            Ok(WorkerSignal::Credit(token)) if self.credit_waiting == Some(token) => {
                self.credit_waiting = None;
                Some(true)
            }
            Ok(WorkerSignal::Credit(_)) => {
                self.stats.push_stale_wakes += 1;
                Some(false)
            }
            Ok(WorkerSignal::Shutdown) | Err(crossbeam_channel::TryRecvError::Disconnected) => None,
        }
    }

    fn restore_physical(&mut self) {
        if let Some(physical) = self.physical.take() {
            self.arena.restore_push_sidebands(physical.into_sidebands());
        }
    }

    fn new(run: &WorkerRun, arena: &'a mut Arena) -> Self {
        let physical = Some(PhysicalRuntime::new(&run.plan, arena.take_push_sidebands()));
        Self {
            arena,
            physical,
            io: IoPlane::new(Arc::clone(&run.io)),
            phase: TaskPhase::Plan,
            index: 0,
            range: 0..0,
            active: false,
            generation: 0,
            wait_epoch: 0,
            waiting: None,
            pipeline_waiting: HashMap::default(),
            push_control: PushControlState::with_node_count(run.plan.len()),
            push_root_done: false,
            credit_waiting: None,
            pull_completion: None,
            morsel_io_uses_start: 0,
            morsel_io_requests_start: 0,
            morsel_io_batches_start: 0,
            morsel_io_blocks_start: 0,
            stats: ScanStats::default(),
        }
    }

    fn assign_next(&mut self, scheduler: &Arc<Scheduler>) -> bool {
        let index = scheduler.next_morsel.fetch_add(1, Ordering::Relaxed);
        let Some(range) = scheduler.run.morsels.get(index).cloned() else {
            self.active = false;
            return false;
        };
        if scheduler.run.plan.has_filter() {
            scheduler.advance_lookahead(assignment_lookahead_target(
                index,
                scheduler.worker_tx.len(),
                scheduler.run.lookahead_morsels,
            ));
        }

        self.index = index;
        self.range = range.clone();
        self.phase = TaskPhase::Plan;
        self.active = true;
        self.generation = self.generation.wrapping_add(1);
        self.wait_epoch = 0;
        self.waiting = None;
        if let Some(physical) = self.physical.as_mut() {
            physical.reset(range.clone());
        }
        self.pipeline_waiting.clear();
        for (node, source) in self.push_control.projection_sources.iter_mut().enumerate() {
            if let Some(mut source) = source.take() {
                source.parts.clear();
                source.direct_rows = None;
                self.push_control.source_activation.parts[node] = source.parts;
            }
        }
        self.push_control.source_matches.clear();
        self.push_control.source_nodes.clear();
        self.push_control.ready_sources.clear();
        self.push_control
            .demand_sources
            .iter_mut()
            .for_each(|source| *source = None);
        self.push_root_done = false;
        self.credit_waiting = None;
        self.pull_completion = None;
        self.push_control.demand_hints.clear();
        self.push_control.has_delayed_hints = false;
        self.push_control
            .demand_state
            .values_mut()
            .for_each(Vec::clear);
        self.push_control.demand_refine_scratch.clear();
        self.push_control.demand_observation.seen.clear();
        self.push_control.demand_observation.span_selected.clear();
        self.push_control.demand_state_live = 0;
        self.morsel_io_uses_start = self.stats.io_uses;
        self.morsel_io_requests_start = self.stats.io_requests;
        self.morsel_io_batches_start = self.stats.io_batches;
        self.morsel_io_blocks_start = self.stats.execute_io_blocks;
        self.io.clear();
        begin_morsel(self.arena, scheduler.run.plan.root(), range);
        true
    }

    fn next_wait_token(&mut self) -> WaitToken {
        self.wait_epoch = self.wait_epoch.wrapping_add(1);
        WaitToken {
            generation: self.generation,
            epoch: self.wait_epoch,
        }
    }

    fn next_pipeline_wait_token(
        &mut self,
        pipeline: PipelineId,
        stage: usize,
    ) -> PipelineWaitToken {
        self.wait_epoch = self.wait_epoch.wrapping_add(1);
        PipelineWaitToken {
            generation: self.generation,
            continuation_epoch: self.wait_epoch,
            pipeline,
            stage,
        }
    }

    fn next_output_wait_token(&mut self) -> OutputWaitToken {
        self.wait_epoch = self.wait_epoch.wrapping_add(1);
        OutputWaitToken {
            generation: self.generation,
            epoch: self.wait_epoch,
        }
    }

    fn wake_pipeline(&mut self, token: PipelineWaitToken) -> bool {
        if !current_pipeline_wait(self.generation, &self.pipeline_waiting, token) {
            self.stats.push_stale_wakes += 1;
            return self
                .physical
                .as_ref()
                .is_some_and(|runtime| !runtime.is_idle());
        }
        let Some(_continuation) = self.pipeline_waiting.remove(&PipelineStage {
            pipeline: token.pipeline,
            stage: token.stage,
        }) else {
            self.stats.push_stale_wakes += 1;
            return false;
        };
        let Some(runtime) = self.physical.as_mut() else {
            self.stats.push_stale_wakes += 1;
            return false;
        };
        runtime.enqueue_resume(token.pipeline, token.stage);
        self.stats.push_pipeline_boundary_resumes += 1;
        true
    }

    fn run(&mut self, scheduler: &Arc<Scheduler>) -> VortexResult<LocalPoll> {
        debug_assert!(self.active);
        match self.phase {
            TaskPhase::Plan => {
                let poll = poll_plan_morsel(
                    self.arena,
                    scheduler.run.plan.root(),
                    &self.io,
                    &scheduler.run.cells,
                    &mut self.stats,
                )?;
                scheduler.submit_reads(self.io.take_reads());
                match poll {
                    PlanPoll::Item(_) => Ok(LocalPoll::Runnable),
                    PlanPoll::Blocked(waits) => Ok(LocalPoll::Blocked(waits)),
                    PlanPoll::Complete => {
                        self.stats.morsels += 1;
                        self.phase = TaskPhase::Execute;
                        if scheduler.run.execution_mode == ExecutionMode::Push {
                            let activation_started = push_profile_enabled().then(Instant::now);
                            self.seed_push_sources(scheduler)?;
                            if let Some(started) = activation_started {
                                self.stats.push_profile_activation_time += started.elapsed();
                            }
                            return self.run_push(scheduler);
                        }
                        Ok(LocalPoll::Runnable)
                    }
                }
            }
            TaskPhase::Execute if scheduler.run.execution_mode == ExecutionMode::Push => {
                self.run_push(scheduler)
            }
            TaskPhase::Execute => match poll_execute_morsel(
                self.arena,
                scheduler.run.plan.root(),
                &self.range,
                &self.io,
                &scheduler.run.cells,
                &scheduler.run.session,
                &mut self.stats,
            )? {
                ExecPoll::Value(batch) => {
                    let array = batch.value.into_array()?;
                    let array = (!array.is_empty()).then_some(array);
                    self.finish_morsel(scheduler, array)
                }
                ExecPoll::Yield(_) => Ok(LocalPoll::Runnable),
                ExecPoll::Blocked(waits) => {
                    self.stats.execute_io_blocks += 1;
                    Ok(LocalPoll::Blocked(waits))
                }
                ExecPoll::Done => self.finish_morsel(scheduler, None),
            },
        }
    }

    fn seed_push_sources(&mut self, scheduler: &Scheduler) -> VortexResult<()> {
        let plan = &scheduler.run.plan;
        plan.overlapping_all_source_indices(&self.range, &mut self.push_control.source_matches);
        let is_deferred = |role| match role {
            SourceRole::Projection => plan.has_filter(),
            SourceRole::Predicate {
                slot,
                mode: crate::nodes::ConjunctMode::Cascade,
            } => slot > 0,
            SourceRole::Predicate {
                mode: crate::nodes::ConjunctMode::Parallel,
                ..
            } => false,
        };
        {
            let mut io_work = scheduler.io_work.lock();
            for &source_index in &self.push_control.source_matches {
                let source = &plan.sources()[source_index];
                if !is_deferred(source.role) {
                    continue;
                }
                let (_, key, source_range, role) = plan
                    .source_io_use_at(source_index)
                    .ok_or_else(|| vortex_err!("deferred push source has no planned IO use"))?;
                debug_assert_eq!(role, source.role);
                let work = if let Some(work) = io_work.get(&key) {
                    Arc::clone(work)
                } else {
                    let read = scheduler.run.io.read_key(key).ok_or_else(|| {
                        vortex_err!("planned push source IO work {key:?} is not registered")
                    })?;
                    let required = read.priority() == IoPriority::Required;
                    let work = new_io_work(read, required);
                    io_work.insert(key, Arc::clone(&work));
                    work
                };
                self.push_control.demand_sources[source.node as usize] = Some(BoundDemandIo {
                    key,
                    source_range,
                    work,
                });
            }
        }
        for &source_index in &self.push_control.source_matches {
            let source = &plan.sources()[source_index];
            let start = self.range.start.max(source.root_range.start);
            let end = self.range.end.min(source.root_range.end);
            if start < end {
                let rows = usize::try_from(end - start)
                    .map_err(|_| vortex_err!("push source span exceeds usize"))?;
                if is_deferred(source.role) {
                    let demand_io = self.push_control.demand_sources[source.node as usize]
                        .clone()
                        .ok_or_else(|| vortex_err!("deferred push source IO was not prebound"))?;
                    let parts = std::mem::take(
                        self.push_control
                            .source_activation
                            .parts
                            .get_mut(source.node as usize)
                            .ok_or_else(|| vortex_err!("push source is outside the node arena"))?,
                    );
                    self.push_control.projection_sources[source.node as usize] =
                        Some(PendingPushSource {
                            span: start..end,
                            role: source.role,
                            demand_io: Some(demand_io),
                            parts,
                            direct_rows: None,
                            known_rows: 0,
                        });
                    continue;
                }
                let selection = ActivationRows::selected(vortex_mask::Mask::new_true(rows));
                let runtime = self
                    .physical
                    .as_mut()
                    .ok_or_else(|| vortex_err!("push physical runtime is not installed"))?;
                let (pipeline, stage) = runtime.location(source.node);
                debug_assert_eq!(stage, 0, "push source must head its physical pipeline");
                runtime.enqueue_start(pipeline, start..end, selection);
                self.stats.push_source_activations += 1;
            }
        }
        self.stats.push_ready_events_max = self.stats.push_ready_events_max.max(
            u64::try_from(
                self.physical
                    .as_ref()
                    .map_or(0, |runtime| runtime.pending.len()),
            )
            .unwrap_or(u64::MAX),
        );
        Ok(())
    }

    #[cfg(test)]
    fn observe_demand_hint(
        &mut self,
        scheduler: &Scheduler,
        target: DemandTarget,
        coverage: Range<u64>,
        selection: vortex_mask::Mask,
    ) {
        let Ok(coverage_len) = usize::try_from(coverage.end.saturating_sub(coverage.start)) else {
            self.stats.demand_hints_dropped += 1;
            return;
        };
        if coverage.start >= coverage.end || selection.len() != coverage_len {
            self.stats.demand_hints_dropped += 1;
            return;
        }
        let spans = self
            .push_control
            .demand_state
            .entry(target)
            .or_insert_with(|| Vec::with_capacity(8));
        let previous_span_count = spans.len();
        if !refine_demand_spans(
            spans,
            &mut self.push_control.demand_refine_scratch,
            coverage.clone(),
            selection.clone(),
        ) {
            self.stats.demand_hints_dropped += 1;
            return;
        }
        self.push_control.demand_state_live = self
            .push_control
            .demand_state_live
            .saturating_sub(previous_span_count)
            .saturating_add(spans.len());
        self.stats.demand_state_live_max = self
            .stats
            .demand_state_live_max
            .max(u64::try_from(self.push_control.demand_state_live).unwrap_or(u64::MAX));
        scheduler.run.plan.overlapping_demand_source_indices(
            target,
            &coverage,
            &mut self.push_control.source_matches,
        );
        self.push_control.source_nodes.clear();
        self.push_control.source_nodes.extend(
            self.push_control
                .source_matches
                .iter()
                .map(|&index| scheduler.run.plan.sources()[index].node),
        );
        observe_prebound_demand(
            &mut self.stats,
            coverage,
            selection,
            &self.push_control.source_nodes,
            &self.push_control.projection_sources,
            &self.push_control.demand_sources,
            &mut self.push_control.demand_observation,
        );
    }

    #[cfg(test)]
    fn publish_demand_hint(
        &mut self,
        scheduler: &Scheduler,
        target: DemandTarget,
        coverage: Range<u64>,
        selection: vortex_mask::Mask,
    ) {
        self.stats.demand_hints_emitted += 1;
        match scheduler.run.demand_hints {
            DemandHintDelivery::Immediate | DemandHintDelivery::Delayed(0) => {
                self.observe_demand_hint(scheduler, target, coverage, selection);
            }
            DemandHintDelivery::Disabled => self.stats.demand_hints_dropped += 1,
            DemandHintDelivery::Delayed(remaining) => {
                self.push_control.demand_hints.push_back(PendingDemandHint {
                    due_transition: self
                        .stats
                        .push_pipeline_stage_calls
                        .saturating_add(u64::try_from(remaining).unwrap_or(u64::MAX)),
                    target,
                    coverage,
                    selection,
                });
                self.push_control.has_delayed_hints = true;
            }
        }
    }

    fn run_push(&mut self, scheduler: &Arc<Scheduler>) -> VortexResult<LocalPoll> {
        loop {
            let profile_started = push_profile_enabled().then(Instant::now);
            let physical = self
                .physical
                .as_mut()
                .ok_or_else(|| vortex_err!("push physical runtime is not installed"))?;
            let mut host = PushHost {
                scheduler: Some(scheduler),
                control: &mut self.push_control,
            };
            let poll = physical.poll(
                self.arena,
                &mut host,
                &self.io,
                &scheduler.run.cells,
                &scheduler.run.session,
                &mut self.stats,
            )?;
            if let Some(started) = profile_started {
                self.stats.push_profile_runtime_time += started.elapsed();
            }
            match poll {
                PipelinePoll::Effect(PipelineEffect::Batch { batch, terminal }) => {
                    if !terminal {
                        return Err(vortex_err!(
                            "physical runtime exposed a nonterminal root batch"
                        ));
                    }
                    if self.stats.time_to_first_batch.is_none() {
                        self.stats.time_to_first_batch = Some(scheduler.run.start.elapsed());
                    }
                    self.stats.push_root_batches += 1;
                    return self.finish_morsel(scheduler, Some(batch));
                }
                PipelinePoll::Effect(PipelineEffect::End) => self.push_root_done = true,
                #[cfg(test)]
                PipelinePoll::Effect(PipelineEffect::Demand {
                    target,
                    coverage,
                    selection,
                }) => self.publish_demand_hint(scheduler, target, coverage, selection),
                PipelinePoll::Waiting {
                    pipeline,
                    stage,
                    waits,
                } => {
                    self.stats.execute_io_blocks += 1;
                    return Ok(LocalPoll::PushBlocked {
                        pipeline,
                        stage,
                        waits,
                    });
                }
                PipelinePoll::Yield => return Ok(LocalPoll::Runnable),
                PipelinePoll::Idle => {
                    let runtime = self
                        .physical
                        .as_ref()
                        .ok_or_else(|| vortex_err!("idle push morsel lost its physical runtime"))?;
                    if (self.push_root_done || runtime.root_done)
                        && self.pipeline_waiting.is_empty()
                        && runtime.is_idle()
                    {
                        return self.finish_morsel(scheduler, None);
                    }
                    if !self.pipeline_waiting.is_empty() {
                        return Ok(LocalPoll::Idle);
                    }
                    let pending = self
                        .push_control
                        .projection_sources
                        .iter()
                        .enumerate()
                        .filter_map(|(node, source)| source.as_ref().map(|source| (node, source)))
                        .map(|(node, source)| {
                            format!(
                                "{node}:{}of{}",
                                source.known_rows,
                                source.span.end - source.span.start
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    return Err(vortex_err!(
                        "push morsel became idle before root completion; pending sources [{pending}]"
                    ));
                }
            }
        }
    }

    fn finish_morsel(
        &mut self,
        scheduler: &Scheduler,
        batch: Option<ArrayRef>,
    ) -> VortexResult<LocalPoll> {
        self.stats.demand_hints_dropped +=
            u64::try_from(self.push_control.demand_hints.len()).unwrap_or(u64::MAX);
        self.push_control.demand_hints.clear();
        if let Some(physical) = self.physical.as_mut() {
            physical.reset(self.range.clone());
        }
        retire_morsel(
            self.arena,
            scheduler.run.plan.root(),
            &scheduler.run.cells,
            &mut self.stats,
        );
        self.io.clear();
        if batch.is_some() && self.stats.time_to_first_batch.is_none() {
            self.stats.time_to_first_batch = Some(scheduler.run.start.elapsed());
        }
        let io_uses = self.stats.io_uses - self.morsel_io_uses_start;
        let io_requests = self.stats.io_requests - self.morsel_io_requests_start;
        let io_batches = self.stats.io_batches - self.morsel_io_batches_start;
        let io_blocks = self.stats.execute_io_blocks - self.morsel_io_blocks_start;
        self.stats
            .record_morsel_io(io_uses, io_requests, io_batches, io_blocks);
        self.active = false;
        Ok(LocalPoll::Complete {
            index: self.index,
            batch,
        })
    }
}

fn activate_pending_sources_into(
    pending: &mut [Option<PendingPushSource>],
    scratch: &mut SourceActivationScratch,
    source_order: &[NodeId],
    target: ActivationTarget,
    coverage: Range<u64>,
    rows: &ActivationRows,
    ready: &mut Vec<(NodeId, Range<u64>, ActivationRows)>,
) -> VortexResult<()> {
    ready.clear();
    scratch.rows.clear();
    let coverage_len = usize::try_from(coverage.end.saturating_sub(coverage.start))
        .map_err(|_| vortex_err!("demand coverage length exceeds usize"))?;
    if coverage.end < coverage.start || rows.logical().len() != coverage_len {
        return Err(vortex_err!("demand selection does not match its coverage"));
    }
    for &node in source_order {
        let Some(source) = pending.get_mut(node as usize).and_then(Option::as_mut) else {
            continue;
        };
        if source.role.activation_target() != target {
            return Err(vortex_err!(
                "pending push source {node} has role {:?}, not activation target {target:?}",
                source.role
            ));
        }
        let start = source.span.start.max(coverage.start);
        let end = source.span.end.min(coverage.end);
        if start >= end {
            continue;
        }
        let total = usize::try_from(source.span.end - source.span.start)
            .map_err(|_| vortex_err!("source span length exceeds usize"))?;
        if source.known_rows == 0
            && source.parts.is_empty()
            && start == source.span.start
            && end == source.span.end
        {
            let batch_start = usize::try_from(start - coverage.start)
                .map_err(|_| vortex_err!("demand slice start exceeds usize"))?;
            let batch_end = usize::try_from(end - coverage.start)
                .map_err(|_| vortex_err!("demand slice end exceeds usize"))?;
            source.direct_rows = Some(if coverage == source.span {
                rows.clone()
            } else {
                let batch_span = batch_start..batch_end;
                if let Some((_, cached)) = scratch
                    .rows
                    .iter()
                    .find(|(cached_span, _)| cached_span == &batch_span)
                {
                    cached.clone()
                } else {
                    let sliced = rows.slice(batch_span.clone());
                    scratch.rows.push((batch_span, sliced.clone()));
                    sliced
                }
            });
            source.known_rows = total;
            continue;
        }
        if source
            .parts
            .iter()
            .any(|(part, _)| part.start < end && start < part.end)
        {
            return Err(vortex_err!(
                "demand update {start}..{end} overlaps an earlier update for source {node}"
            ));
        }
        let batch_start = usize::try_from(start - coverage.start)
            .map_err(|_| vortex_err!("demand slice start exceeds usize"))?;
        let batch_end = usize::try_from(end - coverage.start)
            .map_err(|_| vortex_err!("demand slice end exceeds usize"))?;
        source
            .parts
            .push((start..end, rows.slice(batch_start..batch_end)));
        source.known_rows += batch_end - batch_start;
    }
    for &node in source_order {
        let Some(source) = pending.get(node as usize).and_then(Option::as_ref) else {
            continue;
        };
        let total = usize::try_from(source.span.end - source.span.start)
            .map_err(|_| vortex_err!("source span length exceeds usize"))?;
        let is_ready = source.known_rows == total;
        if !is_ready {
            continue;
        }
        let mut source = pending
            .get_mut(node as usize)
            .and_then(Option::take)
            .ok_or_else(|| vortex_err!("ready push source disappeared"))?;
        let total = usize::try_from(source.span.end - source.span.start)
            .map_err(|_| vortex_err!("source span length exceeds usize"))?;
        let rows = if let Some(rows) = source.direct_rows.take() {
            rows
        } else {
            source
                .parts
                .sort_unstable_by_key(|(coverage, _)| coverage.start);
            let mut selected_slices = Vec::new();
            let mut materialized_slices = Vec::new();
            let mut cursor = source.span.start;
            for (part, rows) in source.parts.drain(..) {
                if part.start != cursor {
                    return Err(vortex_err!("demand updates left a gap at row {cursor}"));
                }
                let offset = usize::try_from(part.start - source.span.start)
                    .map_err(|_| vortex_err!("demand part offset exceeds usize"))?;
                for (mask, slices) in [
                    (rows.logical(), &mut selected_slices),
                    (rows.materialized(), &mut materialized_slices),
                ] {
                    match mask.slices() {
                        vortex_mask::AllOr::All => {
                            slices.push((offset, offset + mask.len()));
                        }
                        vortex_mask::AllOr::None => {}
                        vortex_mask::AllOr::Some(part_slices) => slices.extend(
                            part_slices
                                .iter()
                                .map(|(start, end)| (offset + start, offset + end)),
                        ),
                    }
                }
                cursor = part.end;
            }
            ActivationRows::try_new(
                vortex_mask::Mask::from_slices(total, selected_slices),
                vortex_mask::Mask::from_slices(total, materialized_slices),
            )?
        };
        ready.push((node, source.span, rows));
        let pooled = scratch
            .parts
            .get_mut(node as usize)
            .ok_or_else(|| vortex_err!("push source {node} is outside the node arena"))?;
        *pooled = source.parts;
    }
    Ok(())
}

impl MorselScan {
    /// Configure a scan over a built plan.
    pub fn new(
        plan: Arc<ExecPlan>,
        segments: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> Self {
        let morsels = morsels(&plan, 0);
        Self::new_with_morsels(plan, segments, session, morsels)
    }

    pub(crate) fn new_with_morsels(
        plan: Arc<ExecPlan>,
        segments: Arc<dyn SegmentSource>,
        session: VortexSession,
        morsels: Vec<Range<u64>>,
    ) -> Self {
        Self {
            plan,
            segments,
            io: None,
            session,
            morsels: Arc::from(morsels),
            threads: 1,
            share_decodes: true,
            execution_mode: ExecutionMode::Pull,
            lookahead_morsels: 0,
            output_rows: usize::MAX,
            output_bytes: u64::MAX,
            demand_hints: DemandHintDelivery::Immediate,
            completion: None,
            sparse_morsels: false,
            external_driver: None,
        }
    }

    /// Set the number of driving threads and affinity-owned active morsels.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Override the morsel cut.
    pub fn with_morsels(mut self, morsels: Vec<Range<u64>>) -> Self {
        self.morsels = Arc::from(morsels);
        self
    }

    fn validate_morsels(&self) -> VortexResult<()> {
        let row_count = self.plan.row_count();
        if row_count == 0 {
            if self.morsels.is_empty() {
                return Ok(());
            }
            return Err(vortex_err!("an empty plan requires an empty morsel cut"));
        }
        if self.morsels.is_empty() {
            return Err(vortex_err!("morsel cut does not cover the plan"));
        }
        let mut expected_start = 0;
        for range in self.morsels.iter() {
            if range.start >= range.end {
                return Err(vortex_err!("morsel ranges must be non-empty"));
            }
            if (!self.sparse_morsels && range.start != expected_start)
                || (self.sparse_morsels && range.start < expected_start)
            {
                return Err(vortex_err!(
                    "morsel ranges must be sorted{}",
                    if self.sparse_morsels {
                        " and non-overlapping"
                    } else {
                        " and contiguous"
                    }
                ));
            }
            if range.end > row_count {
                return Err(vortex_err!("morsel range exceeds plan row count"));
            }
            expected_start = range.end;
        }
        if !self.sparse_morsels && expected_start != row_count {
            return Err(vortex_err!("morsel cut does not cover the plan"));
        }
        Ok(())
    }

    /// Enable or disable the leased shared decoded cells.
    pub fn with_share_decodes(mut self, share: bool) -> Self {
        self.share_decodes = share;
        self
    }

    pub(crate) fn with_io_service(mut self, io: Arc<IoService>) -> Self {
        self.io = Some(io);
        self
    }

    pub(crate) fn with_external_driver(mut self, driver: ExternalDriver) -> Self {
        self.external_driver = Some(driver);
        self
    }

    /// Select recursive pull or leaf-driven push value execution.
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }

    /// Keep this many future morsels visible to filtered background I/O in addition to the
    /// worker-active window. Unfiltered scans retain whole-plan lookahead.
    pub fn with_lookahead_morsels(mut self, morsels: usize) -> Self {
        self.lookahead_morsels = morsels;
        self
    }

    /// Configure delivery of optional scheduler-only demand hints.
    pub fn with_demand_hints(mut self, delivery: DemandHintDelivery) -> Self {
        self.demand_hints = delivery;
        self
    }

    /// Deliver each completed morsel to a sink, including morsels with no output.
    pub fn with_completion_sink(
        mut self,
        completion: impl Fn(usize, VortexResult<Option<ArrayRef>>) + Send + Sync + 'static,
    ) -> Self {
        self.completion = Some(Arc::new(completion));
        self
    }

    /// Permit a sorted, non-overlapping subset of the plan's rows as the morsel cut.
    pub fn with_sparse_morsels(mut self, sparse_morsels: bool) -> Self {
        self.sparse_morsels = sparse_morsels;
        self
    }

    /// Bound credited root output retained ahead of the consumer by selected rows and encoded
    /// bytes. A single oversized batch is admitted when no other output holds credit.
    ///
    /// Push execution coalesces internal root fragments before requesting final sink credit, so at
    /// most one additional uncredited morsel (bounded by the configured morsel rows) can be parked
    /// per active worker. This is a bounded streaming sink, not a strict pre-materialization memory
    /// limit.
    pub fn with_output_capacity(mut self, rows: usize, bytes: u64) -> Self {
        self.output_rows = rows.max(1);
        self.output_bytes = bytes.max(1);
        self
    }

    fn lease_counts(&self) -> HashMap<IoKey, usize> {
        let mut counts: HashMap<IoKey, usize> = HashMap::default();
        for (key, range) in self.plan.flat_uses() {
            let overlapping = overlapping_morsels(&self.morsels, &range);
            if overlapping > 0 {
                *counts.entry(key).or_default() += overlapping;
            }
        }
        counts
    }

    /// The morsels this scan will drive.
    pub fn morsel_ranges(&self) -> &[Range<u64>] {
        &self.morsels
    }

    /// Run the scan, returning batches in row order plus the run's counters.
    pub fn run(&self) -> VortexResult<(Vec<ArrayRef>, ScanStats)> {
        let (batches, stats, _) = self.run_timed()?;
        Ok((batches, stats))
    }

    /// Drive every configured morsel on the calling thread.
    ///
    /// This is intended for execution engines such as DuckDB that already own and schedule their
    /// scan threads. Segment futures remain asynchronous; only ordered output coordination uses a
    /// helper thread.
    pub fn run_on_current_thread(&self) -> VortexResult<(Vec<ArrayRef>, ScanStats)> {
        self.validate_morsels()?;
        vortex_ensure!(
            self.output_rows == usize::MAX && self.output_bytes == u64::MAX,
            "caller-thread scans require unbounded output credit"
        );

        let start = Instant::now();
        let cells = if self.share_decodes {
            SharedCells::with_leases(self.lease_counts())
        } else {
            SharedCells::disabled()
        };
        let run = Arc::new(WorkerRun {
            plan: Arc::clone(&self.plan),
            session: self.session.clone(),
            morsels: Arc::clone(&self.morsels),
            io: self
                .io
                .clone()
                .unwrap_or_else(|| IoService::new(Arc::clone(&self.segments))),
            cells,
            start,
            execution_mode: self.execution_mode,
            lookahead_morsels: self.lookahead_morsels,
            output_rows: self.output_rows,
            output_bytes: self.output_bytes,
            demand_hints: self.demand_hints,
            external_driver: self.external_driver.clone(),
        });
        let (scheduler, signals) = Scheduler::new(Arc::clone(&run), 1);
        scheduler.submit_exact_lookahead();
        let signals = signals
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("external morsel worker signal channel is missing"))?;
        let plan = Arc::clone(&self.plan);
        let worker_stats = EXTERNAL_ARENA.with_borrow_mut(|slot| {
            if slot
                .as_ref()
                .is_none_or(|(cached, _)| !Arc::ptr_eq(cached, &plan))
            {
                *slot = Some((Arc::clone(&plan), plan.instantiate()));
            }
            let Some((_, arena)) = slot.as_mut() else {
                unreachable!("external arena was initialized above")
            };
            scheduler.worker_loop(0, &signals, arena)
        });
        let stats = scheduler.finish(vec![worker_stats])?;
        let batches = scheduler.take_ordered_batches();
        if scheduler.remaining.load(Ordering::Acquire) == 0 {
            debug_assert_eq!(
                run.cells.live(),
                0,
                "every lease must be released by the end of the scan"
            );
        }
        Ok((batches, stats))
    }

    /// Start the scan and return an ordered blocking stream with the configured credited-output
    /// bound. Root-edge heads may add at most one parked batch per active worker.
    pub fn into_stream(self) -> VortexResult<MorselStream> {
        self.validate_morsels()?;
        let (output_tx, output_rx) = bounded::<CreditedBatch>(self.threads.max(1));
        let (completion_tx, completion_rx) = mpsc::channel();
        let cancellation = StreamCancellation::new();
        let scan_cancellation = Arc::clone(&cancellation);
        let handle = std::thread::Builder::new()
            .name("vortex-morsel-stream".to_owned())
            .spawn(move || {
                drop(completion_tx.send(self.run_timed_to(output_tx, Some(&scan_cancellation))));
            })
            .map_err(|err| vortex_err!("failed to spawn streaming morsel scan: {err}"))?;
        Ok(MorselStream {
            output: Some(output_rx),
            completion: completion_rx,
            result: None,
            error_yielded: false,
            handle: Some(handle),
            cancellation,
        })
    }

    /// Run the scan with worker creation and shutdown outside the measured interval.
    pub(crate) fn run_timed(&self) -> VortexResult<(Vec<ArrayRef>, ScanStats, Duration)> {
        self.validate_morsels()?;
        let (output_tx, output_rx) = bounded::<CreditedBatch>(self.threads.max(1));
        let collector = std::thread::spawn(move || {
            output_rx
                .into_iter()
                .map(CreditedBatch::receive)
                .collect::<Vec<_>>()
        });
        let (stats, wall) = self.run_timed_to(output_tx, None)?;
        let batches = collector
            .join()
            .map_err(|_| vortex_err!("output collector panicked"))?;
        Ok((batches, stats, wall))
    }

    fn run_timed_to(
        &self,
        output_tx: Sender<CreditedBatch>,
        cancellation: Option<&Arc<StreamCancellation>>,
    ) -> VortexResult<(ScanStats, Duration)> {
        let workers = MorselWorkerPool::new(self.threads, Arc::clone(&self.plan))?;
        let start = Instant::now();
        let cells = if self.share_decodes {
            SharedCells::with_leases(self.lease_counts())
        } else {
            SharedCells::disabled()
        };
        let run = Arc::new(WorkerRun {
            plan: Arc::clone(&self.plan),
            session: self.session.clone(),
            morsels: Arc::clone(&self.morsels),
            io: self
                .io
                .clone()
                .unwrap_or_else(|| IoService::new(Arc::clone(&self.segments))),
            cells,
            start,
            execution_mode: self.execution_mode,
            lookahead_morsels: self.lookahead_morsels,
            output_rows: self.output_rows,
            output_bytes: self.output_bytes,
            demand_hints: self.demand_hints,
            external_driver: self.external_driver.clone(),
        });

        let (scheduler, signals) = Scheduler::new(Arc::clone(&run), self.threads);
        if let Some(cancellation) = cancellation {
            cancellation.install(&scheduler);
        }
        let output_scheduler = Arc::clone(&scheduler);
        let completion = self.completion.clone();
        let coordinator = std::thread::spawn(move || {
            output_scheduler.stream_ordered(&output_tx, completion.as_ref());
        });
        scheduler.submit_exact_lookahead();
        let worker_stats = workers.run(Arc::clone(&scheduler), signals)?;
        let stats = scheduler.finish(worker_stats)?;
        coordinator
            .join()
            .map_err(|_| vortex_err!("ordered output coordinator panicked"))?;

        if scheduler.remaining.load(Ordering::Acquire) == 0 {
            debug_assert_eq!(
                run.cells.live(),
                0,
                "every lease must be released by the end of the scan"
            );
        }
        drop(scheduler);

        let wall = start.elapsed();
        drop(workers);
        Ok((stats, wall))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use parking_lot::Mutex;
    use vortex_array::IntoArray;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_layout::segments::SegmentFuture;
    use vortex_layout::segments::SegmentId;
    use vortex_layout::segments::SegmentSource;
    use vortex_utils::aliases::hash_map::HashMap;

    use super::BoundDemandIo;
    use super::DemandIoAction;
    use super::DemandObservationScratch;
    use super::GateTrace;
    use super::IoWork;
    use super::OutputCredits;
    use super::OutputWaitToken;
    use super::PendingPushSource;
    use super::PhysicalRuntime;
    use super::PipelineCall;
    use super::PipelineContinuation;
    use super::PipelineEffect;
    use super::PipelinePoll;
    use super::PipelineStage;
    use super::PipelineWaitToken;
    use super::PushControlState;
    use super::PushHost;
    use super::RootFragmentPoll;
    use super::SidebandAction;
    use super::SourceActivationScratch;
    use super::StageOutput;
    use super::accept_root_fragment;
    use super::activate_pending_sources_into;
    use super::apply_unissued_demand;
    use super::assignment_lookahead_target;
    use super::claim_lookahead_extension;
    use super::current_pipeline_wait;
    use super::drain_queued_io;
    use super::drain_queued_io_limit;
    use super::observe_prebound_demand;
    use super::overlapping_morsels;
    use super::refine_demand_spans;
    use super::take_ordered_output;
    use crate::build::PhysicalTopology;
    use crate::build::SourceRole;
    use crate::build::TestPipelineDefinition;
    use crate::cells::SharedCells;
    use crate::io::IoKey;
    use crate::io::IoPlane;
    use crate::io::IoPriority;
    use crate::io::IoService;
    use crate::node::ActivationRows;
    use crate::node::ActivationTarget;
    use crate::node::Arena;
    use crate::node::DemandTarget;
    use crate::node::ExecCx;
    use crate::node::ExecNode;
    use crate::node::ExecPoll;
    use crate::node::InputPort;
    use crate::node::Node;
    use crate::node::NodeState;
    use crate::node::PlanCx;
    use crate::node::PlanPoll;
    use crate::node::PushBatch;
    use crate::node::PushCx;
    use crate::node::RetireCx;
    use crate::node::Route;
    use crate::node::Value;
    use crate::node::WaitSet;
    use crate::stats::ScanStats;

    fn test_runtime(
        nodes: Vec<Node>,
        pipelines: Vec<TestPipelineDefinition>,
        credit_targets: Vec<Vec<Option<(u32, usize)>>>,
        root: u32,
    ) -> (Arena, PhysicalRuntime) {
        let node_count = nodes.len();
        let topology = PhysicalTopology::for_test(pipelines, credit_targets, node_count);
        let mut arena = Arena::new_compiled(nodes);
        arena.prepare_push_sidebands((0..node_count).map(|_| 0));
        let sidebands = arena.take_push_sidebands();
        let mut runtime = PhysicalRuntime::from_topology(topology, node_count, root, sidebands);
        runtime.reset(0..1);
        (arena, runtime)
    }

    #[test]
    fn compiled_outgoing_marks_cross_pipeline_boundary() -> VortexResult<()> {
        let port = InputPort::new(0)?;
        let topology = PhysicalTopology::for_test(
            vec![
                (vec![(0, None)], Some(Route { parent: 1, port })),
                (vec![(1, None)], None),
            ],
            vec![Vec::new(), vec![Some((0, 0))]],
            2,
        );

        let target = topology
            .outgoing(0)
            .ok_or_else(|| vortex_err!("pipeline sink has no compiled target"))?;
        assert_eq!((target.pipeline, target.stage, target.node), (1, 0, 1));
        assert_eq!(target.input, port);
        assert!(target.boundary);
        assert!(topology.outgoing(1).is_none());
        Ok(())
    }

    fn test_host(control: &mut PushControlState) -> PushHost<'_> {
        PushHost {
            scheduler: None,
            control,
        }
    }

    #[test]
    fn out_of_order_output_credit_allows_dynamic_morsel_progress() {
        let credits = OutputCredits::new(10, 10);
        let (tx, _rx) = crossbeam_channel::unbounded();
        let token = OutputWaitToken {
            generation: 1,
            epoch: 1,
        };

        assert_eq!(
            credits.acquire_or_park(6, 6, false, tx.clone(), token),
            Some(false)
        );
        assert_eq!(
            credits.acquire_or_park(4, 4, false, tx.clone(), token),
            Some(false)
        );
        assert_eq!(credits.acquire_or_park(1, 1, false, tx, token), None);

        let state = credits.state.lock();
        assert_eq!((state.rows, state.bytes), (10, 10));
        assert_eq!(state.waiters.len(), 1);
    }

    #[test]
    fn assigned_morsels_advance_lookahead_beyond_retired_frontier_without_duplicates() {
        let cursor = AtomicUsize::new(0);
        let workers = 4;
        let explicit_lookahead = 2;

        assert_eq!(claim_lookahead_extension(&cursor, 6, 16), Some(0..6));
        // A worker that finishes out of order can claim morsel 7 while ordered retirement is
        // still at zero. Keep a complete active-worker window plus explicit lookahead ahead of it.
        let assigned_target = assignment_lookahead_target(7, workers, explicit_lookahead);
        assert_eq!(assigned_target, 14);
        assert_eq!(
            claim_lookahead_extension(&cursor, assigned_target, 16),
            Some(6..14)
        );
        assert_eq!(cursor.load(Ordering::Acquire), 14);
        // A lagging assignment and the old retirement window cannot resubmit covered ranges.
        assert_eq!(
            claim_lookahead_extension(
                &cursor,
                assignment_lookahead_target(5, workers, explicit_lookahead),
                16,
            ),
            None
        );
        assert_eq!(claim_lookahead_extension(&cursor, 6, 16), None);
        assert_eq!(claim_lookahead_extension(&cursor, 20, 16), Some(14..16));
        assert_eq!(cursor.load(Ordering::Acquire), 16);
    }

    #[test]
    fn completion_boundary_drains_all_queued_io_and_empty_is_immediate() -> VortexResult<()> {
        let mut queued = 4;
        let mut polls = 0;
        let drained = drain_queued_io(|| {
            polls += 1;
            if queued == 0 {
                return Ok(false);
            }
            queued -= 1;
            Ok(true)
        })?;
        assert_eq!(drained, 4);
        assert_eq!(polls, 5);

        let mut empty_polls = 0;
        let drained = drain_queued_io(|| {
            empty_polls += 1;
            Ok(false)
        })?;
        assert_eq!(drained, 0);
        assert_eq!(empty_polls, 1);
        Ok(())
    }

    #[test]
    fn gate_boundary_drain_is_bounded_and_empty_is_immediate() -> VortexResult<()> {
        let mut queued = 7;
        let mut polls = 0;
        let drained = drain_queued_io_limit(3, || {
            polls += 1;
            queued -= 1;
            Ok(true)
        })?;
        assert_eq!(drained, 3);
        assert_eq!(polls, 3);
        assert_eq!(queued, 4);

        let mut empty_polls = 0;
        let drained = drain_queued_io_limit(3, || {
            empty_polls += 1;
            Ok(false)
        })?;
        assert_eq!(drained, 0);
        assert_eq!(empty_polls, 1);

        let mut zero_polls = 0;
        let drained = drain_queued_io_limit(0, || {
            zero_polls += 1;
            Ok(true)
        })?;
        assert_eq!(drained, 0);
        assert_eq!(zero_polls, 0);
        Ok(())
    }

    #[test]
    fn ordered_head_bypasses_full_output_capacity_once() {
        let credits = OutputCredits::new(8, 8);
        let (tx, _rx) = crossbeam_channel::unbounded();
        let token = OutputWaitToken {
            generation: 1,
            epoch: 1,
        };

        assert_eq!(
            credits.acquire_or_park(8, 8, false, tx.clone(), token),
            Some(false)
        );
        assert_eq!(
            credits.acquire_or_park(3, 3, true, tx.clone(), token),
            Some(true)
        );
        assert_eq!(credits.acquire_or_park(1, 1, true, tx.clone(), token), None);
        {
            let state = credits.state.lock();
            assert_eq!((state.rows, state.bytes), (11, 11));
            assert!(state.head_bypass);
        }

        credits.release(3, 3, true);
        assert_eq!(credits.acquire_or_park(1, 1, true, tx, token), Some(true));
    }

    #[test]
    fn buffered_outputs_are_taken_in_exact_morsel_and_coverage_order() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1i32]).into_array();
        let mut results = vec![
            (1, 10, array.clone(), false),
            (0, 5, array.clone(), false),
            (0, 0, array, false),
        ];

        assert_eq!(
            take_ordered_output(&mut results, 0)
                .ok_or_else(|| vortex_err!("first morsel output disappeared"))?
                .1,
            0
        );
        assert_eq!(
            take_ordered_output(&mut results, 0)
                .ok_or_else(|| vortex_err!("second morsel fragment disappeared"))?
                .1,
            5
        );
        assert!(take_ordered_output(&mut results, 0).is_none());
        assert_eq!(
            take_ordered_output(&mut results, 1)
                .ok_or_else(|| vortex_err!("later morsel output disappeared"))?
                .0,
            1
        );
        Ok(())
    }

    #[test]
    fn root_fragments_coalesce_to_one_terminal_morsel_without_external_credit() -> VortexResult<()>
    {
        let first = PrimitiveArray::from_iter([10i32, 20]).into_array();
        let last = PrimitiveArray::from_iter([30i32]).into_array();
        let dtype = first.dtype().clone();
        let empty = vortex_array::Canonical::empty(&dtype).into_array();
        let mut parts = Vec::with_capacity(8);
        let capacity = parts.capacity();
        let mut coverage_end = 100;
        let morsel = 100..109;
        let fragments = [
            PushBatch::try_new(
                100..102,
                vortex_mask::Mask::new_false(2),
                Value::Array(empty.clone()),
            )?,
            PushBatch::try_new(
                102..105,
                vortex_mask::Mask::from_iter([true, false, true]),
                Value::Array(first),
            )?,
            PushBatch::try_new(
                105..107,
                vortex_mask::Mask::from_iter([false, true]),
                Value::Array(last),
            )?,
            PushBatch::try_new(
                107..109,
                vortex_mask::Mask::new_false(2),
                Value::Array(empty),
            )?,
        ];

        let mut internal_credits = 0;
        let mut outputs = Vec::new();
        for (index, fragment) in fragments.into_iter().enumerate() {
            match accept_root_fragment(
                &mut parts,
                &mut coverage_end,
                &morsel,
                fragment,
                index == 3,
            )? {
                RootFragmentPoll::Continue => internal_credits += 1,
                RootFragmentPoll::Terminal(Some(array)) => outputs.push(array),
                RootFragmentPoll::Terminal(None) => {}
            }
        }

        assert_eq!(internal_credits, 3);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 3);
        assert!(parts.is_empty());
        assert_eq!(parts.capacity(), capacity);

        coverage_end = 200;
        let all_empty = PushBatch::try_new(
            200..204,
            vortex_mask::Mask::new_false(4),
            Value::Array(vortex_array::Canonical::empty(&dtype).into_array()),
        )?;
        assert!(matches!(
            accept_root_fragment(&mut parts, &mut coverage_end, &(200..204), all_empty, true,)?,
            RootFragmentPoll::Terminal(None)
        ));
        assert!(parts.is_empty());
        assert_eq!(parts.capacity(), capacity);
        Ok(())
    }

    #[test]
    fn consumed_credits_queue_in_port_order() -> VortexResult<()> {
        let (_arena, mut runtime) = test_runtime(
            vec![
                Node::dynamic(Box::new(UpstreamOnce {
                    calls: Arc::new(AtomicUsize::new(0)),
                })),
                Node::dynamic(Box::new(UpstreamOnce {
                    calls: Arc::new(AtomicUsize::new(0)),
                })),
                Node::dynamic(Box::new(UpstreamOnce {
                    calls: Arc::new(AtomicUsize::new(0)),
                })),
            ],
            vec![
                (vec![(0, None)], None),
                (vec![(1, None)], None),
                (vec![(2, None)], None),
            ],
            vec![Vec::new(), Vec::new(), vec![Some((0, 0)), Some((1, 0))]],
            2,
        );
        let io = IoPlane::new(IoService::new(Arc::new(UnusedSource)));
        let cells = SharedCells::disabled();
        let session = array_session();
        let mut stats = ScanStats::default();
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);

        let first = {
            let mut services = super::PipelineServices {
                io: &io,
                cells: &cells,
                session: &session,
                stats: &mut stats,
            };
            runtime.apply_sideband(
                2,
                super::StageSideband::Consumed(InputPort::new(0)?),
                &mut host,
                &mut services,
            )?
        };
        let second = {
            let mut services = super::PipelineServices {
                io: &io,
                cells: &cells,
                session: &session,
                stats: &mut stats,
            };
            runtime.apply_sideband(
                2,
                super::StageSideband::Consumed(InputPort::new(1)?),
                &mut host,
                &mut services,
            )?
        };
        assert!(matches!(first, SidebandAction::Continue));
        assert!(matches!(second, SidebandAction::Continue));
        assert_eq!(runtime.pending.len(), 2);
        assert_eq!(runtime.pending[0].pipeline, 0);
        assert!(matches!(runtime.pending[0].call, PipelineCall::Credit));
        assert_eq!(runtime.pending[1].pipeline, 1);
        assert!(matches!(runtime.pending[1].call, PipelineCall::Credit));
        assert_eq!(runtime.work_since_yield, 2);
        Ok(())
    }

    #[test]
    fn wide_source_activation_follows_plan_order_without_catalog_scans() -> VortexResult<()> {
        const SOURCES: u32 = 64;
        let plan_order = (0..SOURCES).collect::<Vec<_>>();
        let mut pending = (0..SOURCES).map(|_| None).collect::<Vec<_>>();
        for node in plan_order.iter().rev().copied() {
            pending[node as usize] = Some(PendingPushSource {
                span: 100..116,
                role: SourceRole::Projection,
                demand_io: None,
                parts: Vec::with_capacity(3),
                direct_rows: None,
                known_rows: 0,
            });
        }
        let expected = vortex_mask::Mask::from_iter([
            true, false, true, true, false, false, true, false, true, true, false, true, false,
            true, true, false,
        ]);
        let mut scratch = SourceActivationScratch {
            parts: (0..SOURCES).map(|_| Vec::new()).collect(),
            rows: Vec::with_capacity(SOURCES as usize),
        };

        let mut ready = Vec::new();
        activate_pending_sources_into(
            &mut pending,
            &mut scratch,
            &plan_order,
            ActivationTarget::Projection,
            100..116,
            &ActivationRows::selected(expected.clone()),
            &mut ready,
        )?;

        assert_eq!(
            ready.iter().map(|(node, ..)| *node).collect::<Vec<_>>(),
            plan_order
        );
        assert!(
            ready
                .iter()
                .all(|(_, span, rows)| span == &(100..116) && rows.logical() == &expected)
        );
        assert!(pending.iter().all(Option::is_none));
        assert_eq!(scratch.parts.len(), SOURCES as usize);
        assert!(scratch.parts.iter().all(|parts| parts.capacity() >= 3));
        assert!(scratch.rows.is_empty());
        Ok(())
    }

    #[test]
    fn full_source_inside_wider_gate_keeps_direct_activation_rows() -> VortexResult<()> {
        let mut pending = (0..2).map(|_| None).collect::<Vec<_>>();
        pending[1] = Some(PendingPushSource {
            span: 102..106,
            role: SourceRole::Projection,
            demand_io: None,
            parts: Vec::with_capacity(2),
            direct_rows: None,
            known_rows: 0,
        });
        let logical =
            vortex_mask::Mask::from_iter([false, true, true, false, true, false, true, true]);
        let materialized = logical.clone();
        let rows = ActivationRows::try_new(logical.clone(), materialized.clone())?;
        let mut scratch = SourceActivationScratch {
            parts: (0..2).map(|_| Vec::new()).collect(),
            rows: Vec::with_capacity(2),
        };
        let mut ready = Vec::new();

        activate_pending_sources_into(
            &mut pending,
            &mut scratch,
            &[1],
            ActivationTarget::Projection,
            100..108,
            &rows,
            &mut ready,
        )?;

        let (_, span, activated) = ready
            .pop()
            .ok_or_else(|| vortex_err!("contained source was not activated"))?;
        assert_eq!(span, 102..106);
        assert_eq!(activated.logical(), &logical.slice(2..6));
        assert_eq!(activated.materialized(), &materialized.slice(2..6));
        assert!(pending[1].is_none());
        assert!(scratch.parts[1].is_empty());
        assert!(scratch.parts[1].capacity() >= 2);
        assert_eq!(scratch.rows.len(), 1);
        Ok(())
    }

    #[test]
    fn wider_gate_reuses_activation_rows_for_sources_with_the_same_span() -> VortexResult<()> {
        let mut pending = (0..4).map(|_| None).collect::<Vec<_>>();
        for node in [3, 1] {
            pending[node] = Some(PendingPushSource {
                span: 102..106,
                role: SourceRole::Projection,
                demand_io: None,
                parts: Vec::new(),
                direct_rows: None,
                known_rows: 0,
            });
        }
        let rows = ActivationRows::selected(vortex_mask::Mask::from_iter([
            false, true, true, false, true, false, true, true,
        ]));
        let mut scratch = SourceActivationScratch {
            parts: (0..4).map(|_| Vec::new()).collect(),
            rows: Vec::with_capacity(4),
        };
        let scratch_capacity = scratch.rows.capacity();
        let mut ready = Vec::new();

        activate_pending_sources_into(
            &mut pending,
            &mut scratch,
            &[1, 3],
            ActivationTarget::Projection,
            100..108,
            &rows,
            &mut ready,
        )?;

        assert_eq!(
            ready.iter().map(|(node, ..)| *node).collect::<Vec<_>>(),
            [1, 3]
        );
        assert_eq!(scratch.rows.len(), 1);
        assert_eq!(scratch.rows.capacity(), scratch_capacity);
        match (ready[0].2.logical(), ready[1].2.logical()) {
            (vortex_mask::Mask::Values(lhs), vortex_mask::Mask::Values(rhs)) => {
                assert!(Arc::ptr_eq(lhs, rhs));
            }
            _ => return Err(vortex_err!("expected value masks")),
        }
        assert!(ready.iter().all(|(_, span, rows)| {
            span == &(102..106)
                && rows.logical() == &vortex_mask::Mask::from_iter([true, false, true, false])
        }));
        Ok(())
    }

    #[test]
    fn fragmented_activation_assembles_both_row_domains_atomically() -> VortexResult<()> {
        let mut pending = (0..8).map(|_| None).collect::<Vec<_>>();
        pending[7] = Some(PendingPushSource {
            span: 100..108,
            role: SourceRole::Predicate {
                slot: 1,
                mode: crate::nodes::ConjunctMode::Cascade,
            },
            demand_io: None,
            parts: Vec::new(),
            direct_rows: None,
            known_rows: 0,
        });
        let mut scratch = SourceActivationScratch {
            parts: (0..8).map(|_| Vec::new()).collect(),
            rows: Vec::with_capacity(8),
        };
        let mut ready = Vec::new();
        activate_pending_sources_into(
            &mut pending,
            &mut scratch,
            &[7],
            ActivationTarget::PredicateSlot(1),
            100..103,
            &ActivationRows::try_new(
                vortex_mask::Mask::from_iter([true, false, false]),
                vortex_mask::Mask::from_iter([true, true, false]),
            )?,
            &mut ready,
        )?;
        assert!(ready.is_empty());
        activate_pending_sources_into(
            &mut pending,
            &mut scratch,
            &[7],
            ActivationTarget::PredicateSlot(1),
            103..108,
            &ActivationRows::try_new(
                vortex_mask::Mask::from_iter([false, true, false, false, true]),
                vortex_mask::Mask::from_iter([true, true, false, true, true]),
            )?,
            &mut ready,
        )?;
        let (_, span, rows) = ready
            .pop()
            .ok_or_else(|| vortex_err!("source not activated"))?;
        assert_eq!(span, 100..108);
        assert_eq!(
            rows.logical(),
            &vortex_mask::Mask::from_iter([true, false, false, false, true, false, false, true])
        );
        assert_eq!(
            rows.materialized(),
            &vortex_mask::Mask::from_iter([true, true, false, true, true, false, true, true])
        );
        Ok(())
    }

    #[test]
    fn gate_tail_starts_first_source_and_queues_extras_in_plan_order() -> VortexResult<()> {
        let calls = Arc::new(AtomicUsize::new(0));
        let nodes = (0..3)
            .map(|_| {
                Node::dynamic(Box::new(UpstreamOnce {
                    calls: Arc::clone(&calls),
                }))
            })
            .collect();
        let (_arena, mut runtime) = test_runtime(
            nodes,
            vec![
                (vec![(0, None)], None),
                (vec![(1, None)], None),
                (vec![(2, None)], None),
            ],
            vec![Vec::new(), Vec::new(), Vec::new()],
            2,
        );
        let mut ready = vec![
            (
                0,
                0..1,
                ActivationRows::selected(vortex_mask::Mask::new_true(1)),
            ),
            (
                1,
                1..2,
                ActivationRows::selected(vortex_mask::Mask::new_true(1)),
            ),
            (
                2,
                2..3,
                ActivationRows::selected(vortex_mask::Mask::new_true(1)),
            ),
        ];

        let first = runtime
            .schedule_ready_sources(&mut ready)
            .ok_or_else(|| vortex_err!("first ready source was not tail-started"))?;
        assert_eq!(first.pipeline, 0);
        assert!(matches!(first.call, PipelineCall::Start { .. }));
        assert!(ready.is_empty());
        assert_eq!(
            runtime
                .pending
                .iter()
                .map(|activation| activation.pipeline)
                .collect::<Vec<_>>(),
            [1, 2]
        );

        runtime.pending.clear();
        runtime.work_since_yield = super::PUSH_INLINE_QUANTUM;
        let mut overflow = vec![(
            0,
            3..4,
            ActivationRows::selected(vortex_mask::Mask::new_true(1)),
        )];
        assert!(runtime.schedule_ready_sources(&mut overflow).is_none());
        assert_eq!(runtime.pending.len(), 1);
        assert_eq!(runtime.pending[0].pipeline, 0);
        Ok(())
    }

    #[test]
    fn projection_gate_rejects_wider_materialization() -> VortexResult<()> {
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        let mut stats = ScanStats::default();
        let rows = ActivationRows::try_new(
            vortex_mask::Mask::from_iter([true, false]),
            vortex_mask::Mask::new_true(2),
        )?;
        assert!(
            host.gate(&mut stats, ActivationTarget::Projection, 0..2, rows, 0)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn counts_overlapping_sorted_morsels() {
        let morsels = [0..10, 10..20, 20..30];

        assert_eq!(overlapping_morsels(&morsels, &(10..20)), 1);
        assert_eq!(overlapping_morsels(&morsels, &(5..25)), 3);
        assert_eq!(overlapping_morsels(&morsels, &(0..30)), 3);
        assert_eq!(overlapping_morsels(&morsels, &(30..40)), 0);
        assert_eq!(overlapping_morsels(&morsels, &(31..40)), 0);
    }

    #[test]
    fn rejects_stale_and_duplicate_pipeline_wakes() {
        let current = PipelineWaitToken {
            generation: 7,
            continuation_epoch: 3,
            pipeline: 11,
            stage: 2,
        };
        let location = PipelineStage {
            pipeline: current.pipeline,
            stage: current.stage,
        };
        let mut waiting = HashMap::default();
        waiting.insert(location, PipelineContinuation { token: current });

        assert!(current_pipeline_wait(7, &waiting, current));
        assert!(!current_pipeline_wait(
            8,
            &waiting,
            PipelineWaitToken {
                generation: 7,
                ..current
            }
        ));
        assert!(!current_pipeline_wait(
            7,
            &waiting,
            PipelineWaitToken {
                continuation_epoch: 2,
                ..current
            }
        ));
        assert!(!current_pipeline_wait(
            7,
            &waiting,
            PipelineWaitToken {
                pipeline: 12,
                ..current
            }
        ));
        assert!(!current_pipeline_wait(
            7,
            &waiting,
            PipelineWaitToken {
                stage: 1,
                ..current
            }
        ));
        waiting.remove(&location);
        assert!(!current_pipeline_wait(7, &waiting, current));
    }

    #[test]
    fn demand_refinement_is_partition_independent_and_rejects_growth() {
        let mut spans = Vec::new();
        let mut scratch = Vec::new();
        assert!(refine_demand_spans(
            &mut spans,
            &mut scratch,
            0..6,
            vortex_mask::Mask::from_iter([true, false, true, true, false, true]),
        ));
        assert!(refine_demand_spans(
            &mut spans,
            &mut scratch,
            0..3,
            vortex_mask::Mask::from_iter([true, false, false]),
        ));
        assert!(refine_demand_spans(
            &mut spans,
            &mut scratch,
            3..6,
            vortex_mask::Mask::from_iter([false, false, true]),
        ));
        assert_eq!(spans.len(), 2);

        let before = spans.clone();
        assert!(!refine_demand_spans(
            &mut spans,
            &mut scratch,
            1..5,
            vortex_mask::Mask::from_iter([true, true, false, true]),
        ));
        assert_eq!(spans, before);
    }

    #[test]
    fn monotone_demand_fragments_reuse_span_storage_without_fallback() {
        let mut spans = Vec::with_capacity(4);
        let initial_capacity = spans.capacity();
        let mut scratch = Vec::new();
        for start in [0, 2, 4, 6] {
            assert!(refine_demand_spans(
                &mut spans,
                &mut scratch,
                start..start + 2,
                vortex_mask::Mask::from_iter([true, false]),
            ));
        }
        assert_eq!(spans.len(), 4);
        assert_eq!(spans.capacity(), initial_capacity);
        assert!(scratch.is_empty());
        assert_eq!(scratch.capacity(), 0);
    }

    struct UnusedSource;

    impl SegmentSource for UnusedSource {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            futures::future::pending().boxed()
        }
    }

    struct CountingIssueSource {
        requests: Arc<AtomicUsize>,
    }

    impl SegmentSource for CountingIssueSource {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            self.requests.fetch_add(1, Ordering::Relaxed);
            futures::future::pending().boxed()
        }
    }

    fn unissued_test_work(service: &Arc<IoService>, segment: u32) -> VortexResult<IoWork> {
        let mut reads = service.register_reads(
            [IoKey::Segment(SegmentId::from(segment))],
            IoPriority::Speculative,
        );
        let read = reads
            .pop()
            .ok_or_else(|| vortex_err!("test read was already submitted"))?;
        Ok(IoWork {
            queued: AtomicBool::new(false),
            running: AtomicBool::new(false),
            required: AtomicBool::new(false),
            ready: Mutex::new(vec![0]),
            scheduled: vec![AtomicBool::new(true)],
            completed: vec![AtomicBool::new(false)],
            reads: vec![read],
        })
    }

    #[test]
    fn prebound_demand_preserves_identity_dedup_and_source_order() -> VortexResult<()> {
        let service = IoService::new(Arc::new(UnusedSource));
        let shared = Arc::new(unissued_test_work(&service, 21)?);
        let other = Arc::new(unissued_test_work(&service, 22)?);
        let same_span = Arc::new(unissued_test_work(&service, 23)?);
        let shared_key = shared.reads[0].key();
        let other_key = other.reads[0].key();
        let same_span_key = same_span.reads[0].key();
        let shared_binding = BoundDemandIo {
            key: shared_key,
            source_range: 100..102,
            work: Arc::clone(&shared),
        };
        let other_binding = BoundDemandIo {
            key: other_key,
            source_range: 102..104,
            work: Arc::clone(&other),
        };
        let same_span_binding = BoundDemandIo {
            key: same_span_key,
            source_range: 100..102,
            work: Arc::clone(&same_span),
        };
        let mut pending = (0..5).map(|_| None).collect::<Vec<_>>();
        pending[2] = Some(PendingPushSource {
            span: 100..102,
            role: SourceRole::Projection,
            demand_io: Some(shared_binding.clone()),
            parts: Vec::new(),
            direct_rows: None,
            known_rows: 0,
        });
        let mut sources = (0..5).map(|_| None).collect::<Vec<_>>();
        sources[1] = Some(shared_binding);
        sources[3] = Some(other_binding);
        sources[4] = Some(same_span_binding);
        let mut scratch = DemandObservationScratch {
            seen: Vec::with_capacity(4),
            span_selected: Vec::with_capacity(4),
        };
        let mut stats = ScanStats::default();

        observe_prebound_demand(
            &mut stats,
            100..104,
            vortex_mask::Mask::from_iter([true, false, false, false]),
            &[2, 1, 3, 4],
            &pending,
            &sources,
            &mut scratch,
        );

        assert!(Arc::ptr_eq(
            &pending[2]
                .as_ref()
                .and_then(|source| source.demand_io.as_ref())
                .ok_or_else(|| vortex_err!("pending source lost its bound IO work"))?
                .work,
            &shared
        ));
        assert_eq!(scratch.seen, [shared_key, other_key, same_span_key]);
        assert_eq!(scratch.span_selected.len(), 2);
        assert_eq!(stats.demand_io_candidates, 3);
        assert_eq!(stats.demand_io_promotions, 2);
        assert_eq!(stats.demand_io_suppressed, 1);
        assert!(shared.required.load(Ordering::Acquire));
        assert!(!other.required.load(Ordering::Acquire));
        assert!(same_span.required.load(Ordering::Acquire));
        Ok(())
    }

    #[test]
    fn selected_unissued_demand_waits_for_source_to_issue() -> VortexResult<()> {
        let requests = Arc::new(AtomicUsize::new(0));
        let service = IoService::new(Arc::new(CountingIssueSource {
            requests: Arc::clone(&requests),
        }));
        let work = unissued_test_work(&service, 11)?;

        assert_eq!(apply_unissued_demand(&work, true), DemandIoAction::Required);
        assert!(work.required.load(Ordering::Acquire));
        assert!(work.reads[0].is_unissued());
        assert_eq!(work.reads[0].priority(), IoPriority::Speculative);
        assert_eq!(requests.load(Ordering::Relaxed), 0);

        assert!(service.issue(&work.reads[0]));
        work.reads[0].promote();
        assert_eq!(work.reads[0].priority(), IoPriority::Required);
        assert_eq!(requests.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn all_false_unissued_demand_stays_suppressed() -> VortexResult<()> {
        let requests = Arc::new(AtomicUsize::new(0));
        let service = IoService::new(Arc::new(CountingIssueSource {
            requests: Arc::clone(&requests),
        }));
        let work = unissued_test_work(&service, 12)?;

        assert_eq!(
            apply_unissued_demand(&work, false),
            DemandIoAction::Suppressed
        );
        assert!(!work.required.load(Ordering::Acquire));
        assert!(work.reads[0].is_unissued());
        assert_eq!(requests.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn already_issued_demand_work_is_unchanged() -> VortexResult<()> {
        let requests = Arc::new(AtomicUsize::new(0));
        let service = IoService::new(Arc::new(CountingIssueSource {
            requests: Arc::clone(&requests),
        }));
        let work = unissued_test_work(&service, 13)?;
        assert!(service.issue(&work.reads[0]));

        assert_eq!(
            apply_unissued_demand(&work, true),
            DemandIoAction::Unchanged
        );
        assert!(!work.required.load(Ordering::Acquire));
        assert!(!work.reads[0].is_unissued());
        assert_eq!(requests.load(Ordering::Relaxed), 1);
        Ok(())
    }

    struct CursorNode {
        cursor: usize,
    }

    impl ExecNode for CursorNode {
        fn reset(&mut self, _range: Range<u64>) {
            self.cursor = 0;
        }

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_resume(
            &mut self,
            _cx: &mut PushCx<'_>,
            out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            if self.cursor < 2 {
                out.push_demand(
                    DemandTarget::PredicateSlot(self.cursor),
                    0..1,
                    vortex_mask::Mask::new_true(1),
                );
                self.cursor += 1;
                Ok(NodeState::Ready)
            } else {
                out.set_end();
                Ok(NodeState::NeedInput)
            }
        }

        fn push_credit(
            &mut self,
            _cx: &mut PushCx<'_>,
            out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            out.push_demand(
                DemandTarget::PredicateSlot(9),
                0..1,
                vortex_mask::Mask::new_true(1),
            );
            Ok(NodeState::NeedInput)
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    #[test]
    fn physical_pipeline_retains_output_cursor_ahead_of_credit() -> VortexResult<()> {
        let io = IoPlane::new(IoService::new(Arc::new(UnusedSource)));
        let cells = SharedCells::disabled();
        let session = array_session();
        let mut stats = ScanStats::default();
        let (mut arena, mut runtime) = test_runtime(
            vec![Node::dynamic(Box::new(CursorNode { cursor: 0 }))],
            vec![(vec![(0, None)], None)],
            vec![Vec::new()],
            0,
        );
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        runtime.enqueue_resume(0, 0);
        let mut order = Vec::new();
        loop {
            match runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)? {
                PipelinePoll::Effect(PipelineEffect::Demand {
                    target: DemandTarget::PredicateSlot(slot),
                    ..
                }) => order.push(slot),
                PipelinePoll::Effect(PipelineEffect::End) => {
                    order.push(2);
                    runtime.enqueue_credit(0, 0);
                }
                PipelinePoll::Idle => break,
                PipelinePoll::Effect(PipelineEffect::Batch { .. })
                | PipelinePoll::Effect(PipelineEffect::Demand { .. })
                | PipelinePoll::Waiting { .. }
                | PipelinePoll::Yield => return Err(vortex_err!("unexpected pipeline poll")),
            }
        }
        assert_eq!(order, [0, 1, 2, 9]);
        assert_eq!(stats.push_pipeline_stage_calls, 4);
        assert_eq!(stats.push_dispatch_spills, 0);

        runtime.reset(0..1);
        arena.reset_subtree(0, 0..1);
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        runtime.enqueue_resume(0, 0);
        let mut second_order = Vec::new();
        loop {
            match runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)? {
                PipelinePoll::Effect(PipelineEffect::Demand {
                    target: DemandTarget::PredicateSlot(slot),
                    ..
                }) => second_order.push(slot),
                PipelinePoll::Effect(PipelineEffect::End) => {
                    second_order.push(2);
                    runtime.enqueue_credit(0, 0);
                }
                PipelinePoll::Idle => break,
                _ => return Err(vortex_err!("unexpected second-morsel pipeline poll")),
            }
        }
        assert_eq!(second_order, order);
        Ok(())
    }

    struct DemandLoopNode {
        next: usize,
        remaining: usize,
    }

    struct GateBurstNode {
        emitted: bool,
    }

    impl ExecNode for GateBurstNode {
        fn reset(&mut self, _range: Range<u64>) {
            self.emitted = false;
        }

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_resume(
            &mut self,
            _cx: &mut PushCx<'_>,
            out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            if self.emitted {
                return Ok(NodeState::NeedInput);
            }
            for slot in 0..40 {
                out.push_gate(
                    ActivationTarget::PredicateSlot(slot),
                    u64::try_from(slot).unwrap_or(u64::MAX)
                        ..u64::try_from(slot + 1).unwrap_or(u64::MAX),
                    ActivationRows::selected(vortex_mask::Mask::new_false(1)),
                );
            }
            self.emitted = true;
            Ok(NodeState::NeedInput)
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    #[test]
    fn gate_burst_yields_and_resumes_exactly_once_in_plan_order() -> VortexResult<()> {
        let io = IoPlane::new(IoService::new(Arc::new(UnusedSource)));
        let cells = SharedCells::disabled();
        let session = array_session();
        let mut stats = ScanStats::default();
        let (mut arena, mut runtime) = test_runtime(
            vec![Node::dynamic(Box::new(GateBurstNode { emitted: false }))],
            vec![(vec![(0, None)], None)],
            vec![Vec::new()],
            0,
        );
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        runtime.enqueue_resume(0, 0);

        let mut yields = 0;
        loop {
            match runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)? {
                PipelinePoll::Yield => yields += 1,
                PipelinePoll::Idle => break,
                _ => return Err(vortex_err!("unexpected gate-burst pipeline poll")),
            }
        }

        assert!(yields >= 1);
        assert_eq!(stats.push_inline_gates, 40);
        assert_eq!(host.control.gate_trace.len(), 80);
        let (pairs, remainder) = host.control.gate_trace.as_chunks::<2>();
        assert!(remainder.is_empty());
        for (slot, pair) in pairs.iter().enumerate() {
            let target = ActivationTarget::PredicateSlot(slot);
            let coverage = u64::try_from(slot).unwrap_or(u64::MAX)
                ..u64::try_from(slot + 1).unwrap_or(u64::MAX);
            match pair {
                [
                    GateTrace::Hint(hint_target, hint_coverage, hint_rows),
                    GateTrace::Activate(activation_target, activation_coverage, activation_rows),
                ] => {
                    assert_eq!(*hint_target, target);
                    assert_eq!(*activation_target, target);
                    assert_eq!(hint_coverage, &coverage);
                    assert_eq!(activation_coverage, &coverage);
                    assert!(hint_rows.logical().all_false());
                    assert!(hint_rows.materialized().all_false());
                    assert!(activation_rows.logical().all_false());
                    assert!(activation_rows.materialized().all_false());
                }
                _ => return Err(vortex_err!("gate hint/activation order was not atomic")),
            }
        }
        Ok(())
    }

    impl ExecNode for DemandLoopNode {
        fn reset(&mut self, _range: Range<u64>) {}

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_resume(
            &mut self,
            _cx: &mut PushCx<'_>,
            out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            if self.remaining == 0 {
                return Ok(NodeState::NeedInput);
            }
            out.push_demand(
                DemandTarget::PredicateSlot(self.next),
                0..1,
                vortex_mask::Mask::new_true(1),
            );
            self.next += 1;
            self.remaining -= 1;
            Ok(if self.remaining == 0 {
                NodeState::NeedInput
            } else {
                NodeState::Ready
            })
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    #[test]
    fn physical_pipeline_fairness_persists_across_control_effects() -> VortexResult<()> {
        let io = IoPlane::new(IoService::new(Arc::new(UnusedSource)));
        let cells = SharedCells::disabled();
        let session = array_session();
        let mut stats = ScanStats::default();
        let (mut arena, mut runtime) = test_runtime(
            vec![Node::dynamic(Box::new(DemandLoopNode {
                next: 0,
                remaining: 80,
            }))],
            vec![(vec![(0, None)], None)],
            vec![Vec::new()],
            0,
        );
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        runtime.enqueue_resume(0, 0);
        let mut effects = 0;
        let mut yielded = false;
        loop {
            match runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)? {
                PipelinePoll::Effect(PipelineEffect::Demand {
                    target: DemandTarget::PredicateSlot(slot),
                    ..
                }) => {
                    assert_eq!(slot, effects);
                    effects += 1;
                }
                PipelinePoll::Yield => yielded = true,
                PipelinePoll::Idle => break,
                _ => return Err(vortex_err!("unexpected pipeline poll")),
            }
        }
        assert_eq!(effects, 80);
        assert!(yielded);
        assert!(stats.push_dispatch_spills > 0);
        Ok(())
    }

    struct UpstreamOnce {
        calls: Arc<AtomicUsize>,
    }

    struct PassiveSource {
        calls: Arc<AtomicUsize>,
    }

    impl ExecNode for PassiveSource {
        fn reset(&mut self, _range: Range<u64>) {}

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_resume(
            &mut self,
            _cx: &mut PushCx<'_>,
            out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            out.set_batch(
                PushBatch::try_new(
                    0..1,
                    vortex_mask::Mask::new_true(1),
                    Value::Mask(vortex_mask::Mask::new_true(1)),
                )?,
                true,
            );
            Ok(NodeState::NeedInput)
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    struct PassiveSink {
        calls: Arc<AtomicUsize>,
    }

    impl ExecNode for PassiveSink {
        fn reset(&mut self, _range: Range<u64>) {}

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_input(
            &mut self,
            _port: InputPort,
            _batch: PushBatch,
            _last_for_input: bool,
            _cx: &mut PushCx<'_>,
            _out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(NodeState::NeedInput)
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    #[test]
    fn passive_intra_pipeline_transfer_ignores_boundary_block_flag() -> VortexResult<()> {
        let io = IoPlane::new(IoService::new(Arc::new(UnusedSource)));
        let cells = SharedCells::disabled();
        let session = array_session();
        let mut stats = ScanStats::default();
        let source_calls = Arc::new(AtomicUsize::new(0));
        let sink_calls = Arc::new(AtomicUsize::new(0));
        let (mut arena, mut runtime) = test_runtime(
            vec![
                Node::dynamic(Box::new(PassiveSource {
                    calls: Arc::clone(&source_calls),
                })),
                Node::dynamic(Box::new(PassiveSink {
                    calls: Arc::clone(&sink_calls),
                })),
            ],
            vec![(vec![(0, None), (1, Some(InputPort::new(0)?))], None)],
            vec![Vec::new(), vec![Some((0, 0))]],
            1,
        );
        runtime.blocked[0] = true;
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        let mut services = super::PipelineServices {
            io: &io,
            cells: &cells,
            session: &session,
            stats: &mut stats,
        };

        assert!(
            runtime
                .invoke(
                    &mut arena,
                    &mut host,
                    0,
                    0,
                    PipelineCall::Resume,
                    &mut services,
                )?
                .is_none()
        );
        assert_eq!(source_calls.load(Ordering::Relaxed), 1);
        assert_eq!(sink_calls.load(Ordering::Relaxed), 1);
        assert_eq!(stats.push_fast_stage_transfers, 1);
        assert_eq!(stats.push_inline_transfers, 1);
        assert_eq!(stats.push_pipeline_boundary_resumes, 1);
        assert!(runtime.frames.is_empty());
        Ok(())
    }

    impl ExecNode for UpstreamOnce {
        fn reset(&mut self, _range: Range<u64>) {}

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_resume(
            &mut self,
            _cx: &mut PushCx<'_>,
            out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if call == 0 {
                out.set_batch(
                    PushBatch::try_new(
                        0..1,
                        vortex_mask::Mask::new_true(1),
                        Value::Mask(vortex_mask::Mask::new_true(1)),
                    )?,
                    true,
                );
                Ok(NodeState::Ready)
            } else {
                Ok(NodeState::NeedInput)
            }
        }

        fn push_credit(
            &mut self,
            _cx: &mut PushCx<'_>,
            _out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            Ok(NodeState::NeedInput)
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    struct TerminalRoot {
        calls: Arc<AtomicUsize>,
    }

    struct FragmentedRoot {
        next: usize,
        calls: Arc<AtomicUsize>,
    }

    impl FragmentedRoot {
        fn emit(&mut self, out: &mut StageOutput) -> VortexResult<NodeState> {
            let index = self.next;
            self.next += 1;
            self.calls.fetch_add(1, Ordering::Relaxed);
            let (selection, array) = if index == 1 {
                (
                    vortex_mask::Mask::new_false(1),
                    PrimitiveArray::from_iter(std::iter::empty::<i32>()).into_array(),
                )
            } else {
                (
                    vortex_mask::Mask::new_true(1),
                    PrimitiveArray::from_iter([[10, 20, 30, 40][index]]).into_array(),
                )
            };
            let terminal = index == 3;
            let coverage_start =
                u64::try_from(index).map_err(|_| vortex_err!("fragment index exceeds u64"))?;
            out.set_batch(
                PushBatch::try_new(
                    coverage_start..coverage_start + 1,
                    selection,
                    Value::Array(array),
                )?,
                terminal,
            );
            Ok(if terminal {
                NodeState::Done
            } else {
                NodeState::NeedInput
            })
        }
    }

    impl ExecNode for FragmentedRoot {
        fn reset(&mut self, _range: Range<u64>) {
            self.next = 0;
        }

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_resume(
            &mut self,
            _cx: &mut PushCx<'_>,
            out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            self.emit(out)
        }

        fn push_credit(
            &mut self,
            _cx: &mut PushCx<'_>,
            out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            self.emit(out)
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    impl ExecNode for TerminalRoot {
        fn reset(&mut self, _range: Range<u64>) {}

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_resume(
            &mut self,
            _cx: &mut PushCx<'_>,
            out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            out.set_batch(
                PushBatch::try_new(
                    0..1,
                    vortex_mask::Mask::new_true(1),
                    Value::Array(PrimitiveArray::from_iter([1i32]).into_array()),
                )?,
                true,
            );
            Ok(NodeState::Done)
        }

        fn push_credit(
            &mut self,
            _cx: &mut PushCx<'_>,
            _out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(NodeState::Done)
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    #[test]
    fn terminal_root_batch_does_not_require_a_credit_invocation() -> VortexResult<()> {
        let io = IoPlane::new(IoService::new(Arc::new(UnusedSource)));
        let cells = SharedCells::disabled();
        let session = array_session();
        let mut stats = ScanStats::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let (mut arena, mut runtime) = test_runtime(
            vec![Node::dynamic(Box::new(TerminalRoot {
                calls: Arc::clone(&calls),
            }))],
            vec![(vec![(0, None)], None)],
            vec![Vec::new()],
            0,
        );
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        runtime.enqueue_resume(0, 0);

        assert!(matches!(
            runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)?,
            PipelinePoll::Effect(PipelineEffect::Batch { terminal: true, .. })
        ));
        assert!(runtime.root_done);
        assert!(matches!(
            runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)?,
            PipelinePoll::Idle
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn physical_runtime_coalesces_root_fragments_and_credits_internally() -> VortexResult<()> {
        let io = IoPlane::new(IoService::new(Arc::new(UnusedSource)));
        let cells = SharedCells::disabled();
        let session = array_session();
        let mut stats = ScanStats::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let (mut arena, mut runtime) = test_runtime(
            vec![Node::dynamic(Box::new(FragmentedRoot {
                next: 0,
                calls: Arc::clone(&calls),
            }))],
            vec![(vec![(0, None)], None)],
            vec![Vec::new()],
            0,
        );
        runtime.reset(0..4);
        arena.reset_subtree(0, 0..4);
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        runtime.enqueue_resume(0, 0);

        let poll = runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)?;
        let PipelinePoll::Effect(PipelineEffect::Batch { batch, terminal }) = poll else {
            return Err(vortex_err!(
                "runtime did not expose one terminal root batch"
            ));
        };
        assert!(terminal);
        assert_eq!(batch.len(), 3);
        assert_eq!(calls.load(Ordering::Relaxed), 4);
        assert_eq!(stats.push_pipeline_boundary_resumes, 0);
        assert!(runtime.root_done);
        assert!(matches!(
            runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats,)?,
            PipelinePoll::Idle
        ));
        Ok(())
    }

    struct WaitOnce {
        waiting: bool,
    }

    impl ExecNode for WaitOnce {
        fn reset(&mut self, _range: Range<u64>) {}

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_input(
            &mut self,
            _port: InputPort,
            _batch: PushBatch,
            _last_for_input: bool,
            _cx: &mut PushCx<'_>,
            _out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            self.waiting = true;
            Ok(NodeState::Waiting(WaitSet::new()))
        }

        fn push_resume(
            &mut self,
            _cx: &mut PushCx<'_>,
            _out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            self.waiting = false;
            Ok(NodeState::NeedInput)
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    #[test]
    fn blocked_stage_freezes_its_upstream_frame() -> VortexResult<()> {
        let io = IoPlane::new(IoService::new(Arc::new(UnusedSource)));
        let cells = SharedCells::disabled();
        let session = array_session();
        let mut stats = ScanStats::default();
        let calls = Arc::new(AtomicUsize::new(0));
        let (mut arena, mut runtime) = test_runtime(
            vec![
                Node::dynamic(Box::new(UpstreamOnce {
                    calls: Arc::clone(&calls),
                })),
                Node::dynamic(Box::new(WaitOnce { waiting: false })),
            ],
            vec![(vec![(0, None), (1, Some(InputPort::new(0)?))], None)],
            vec![Vec::new(), vec![Some((0, 0))]],
            1,
        );
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        runtime.enqueue_resume(0, 0);
        assert!(matches!(
            runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)?,
            PipelinePoll::Waiting {
                pipeline: 0,
                stage: 1,
                ..
            }
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        runtime.enqueue_credit(0, 0);
        assert!(matches!(
            runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)?,
            PipelinePoll::Idle
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        runtime.enqueue_resume(0, 1);
        assert!(matches!(
            runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)?,
            PipelinePoll::Idle
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        Ok(())
    }

    struct BlockingBoundary {
        waiting: bool,
        inputs: Arc<AtomicUsize>,
    }

    impl ExecNode for BlockingBoundary {
        fn reset(&mut self, _range: Range<u64>) {}

        fn next_plan(&mut self, _cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
            Ok(PlanPoll::Complete)
        }

        fn execute(&mut self, _cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
            Ok(ExecPoll::Done)
        }

        fn push_input(
            &mut self,
            _port: InputPort,
            _batch: PushBatch,
            _last_for_input: bool,
            _cx: &mut PushCx<'_>,
            _out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            if self.waiting {
                return Err(vortex_err!("blocked boundary was re-entered"));
            }
            let input = self.inputs.fetch_add(1, Ordering::Relaxed);
            if input == 0 {
                self.waiting = true;
                Ok(NodeState::Waiting(WaitSet::new()))
            } else {
                Ok(NodeState::NeedInput)
            }
        }

        fn push_resume(
            &mut self,
            _cx: &mut PushCx<'_>,
            _out: &mut StageOutput,
        ) -> VortexResult<NodeState> {
            if !self.waiting {
                return Err(vortex_err!("boundary resumed without waiting"));
            }
            self.waiting = false;
            Ok(NodeState::NeedInput)
        }

        fn retire(&mut self, _cx: &mut RetireCx<'_>) {}

        fn children(&self) -> &[u32] {
            &[]
        }
    }

    #[test]
    fn second_producer_waits_for_blocked_boundary_resume() -> VortexResult<()> {
        let io = IoPlane::new(IoService::new(Arc::new(UnusedSource)));
        let cells = SharedCells::disabled();
        let session = array_session();
        let mut stats = ScanStats::default();
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));
        let inputs = Arc::new(AtomicUsize::new(0));
        let port_0 = InputPort::new(0)?;
        let port_1 = InputPort::new(1)?;
        let (mut arena, mut runtime) = test_runtime(
            vec![
                Node::dynamic(Box::new(UpstreamOnce {
                    calls: Arc::clone(&first_calls),
                })),
                Node::dynamic(Box::new(UpstreamOnce {
                    calls: Arc::clone(&second_calls),
                })),
                Node::dynamic(Box::new(BlockingBoundary {
                    waiting: false,
                    inputs: Arc::clone(&inputs),
                })),
            ],
            vec![
                (
                    vec![(0, None)],
                    Some(Route {
                        parent: 2,
                        port: port_0,
                    }),
                ),
                (
                    vec![(1, None)],
                    Some(Route {
                        parent: 2,
                        port: port_1,
                    }),
                ),
                (vec![(2, None)], None),
            ],
            vec![Vec::new(), Vec::new(), vec![Some((0, 0)), Some((1, 0))]],
            2,
        );
        let mut control = PushControlState::default();
        let mut host = test_host(&mut control);
        runtime.enqueue_resume(0, 0);
        assert!(matches!(
            runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)?,
            PipelinePoll::Waiting {
                pipeline: 2,
                stage: 0,
                ..
            }
        ));
        assert_eq!(inputs.load(Ordering::Relaxed), 1);

        runtime.enqueue_resume(1, 0);
        assert!(matches!(
            runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)?,
            PipelinePoll::Idle
        ));
        assert_eq!(inputs.load(Ordering::Relaxed), 1);
        assert_eq!(second_calls.load(Ordering::Relaxed), 1);

        runtime.enqueue_resume(2, 0);
        assert!(matches!(
            runtime.poll(&mut arena, &mut host, &io, &cells, &session, &mut stats)?,
            PipelinePoll::Idle
        ));
        assert_eq!(inputs.load(Ordering::Relaxed), 2);
        assert_eq!(first_calls.load(Ordering::Relaxed), 2);
        assert_eq!(second_calls.load(Ordering::Relaxed), 2);
        Ok(())
    }
}
