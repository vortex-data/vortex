//! Physical plan v2 — lowering API + Source/Transform/Sink runtime.
//!
//! This module provides the lowering model proposed in
//! `docs/design/proposals/passive-sync-execution.md`:
//!
//! - [`Operator::lower(ctx, tail)`][Operator] dispatches per-operator
//!   pipeline construction in continuation-passing style.
//! - The runtime side is split into three concrete traits
//!   ([`SourceNode`], [`TransformNode`], [`SinkNode`]); all three
//!   are poll-based synchronous state machines.
//! - Async / CPU offloads happen via [`SpawnRuntime`] exposed on
//!   each operator's ctx ([`SourceCtx`], [`TransformCtx`],
//!   [`SinkCtx`]). `spawn`, `spawn_io`, `spawn_cpu` return a
//!   [`WorkHandle<T>`] the operator polls in its state machine.
//! - Pipelines are linked by `PipelineBarrier`s for cross-pipeline
//!   coordination (build-then-probe, etc.).
//!
//! **No row demand, no cancellation.** The new model intentionally
//! drops requirement propagation; downstream consumers always
//! consume every row their upstream produces, and dropping a
//! `WorkHandle` abandons the result without cancelling the work.

mod abi;
mod driver_io;
mod error;
pub mod gather;
mod ids;
pub mod limit;
mod submitter;
mod lowering;
pub mod merge_join;
pub mod merge_join_resource;
pub mod operators;
pub mod parent_child_min;
mod plan;
mod pool;
pub mod runtime;
mod spawn;
pub mod sum_aggregate;
pub mod vortex_aggregate;
pub mod vortex_scan;

pub use abi::{
    Batch, DynSinkNode, DynSourceNode, DynTransformNode, LocalInitRuntime, OperatorPoll,
    Parallelism, PendingSend, SinkCtx, SinkDriver, SinkNode, SourceCtx, SourceDriver, SourceNode,
    TransformCtx, TransformDriver, TransformNode, TransformOutput, TypedSinkNode, TypedSourceNode,
    TypedTransformNode,
};
pub use driver_io::DriverIo;
pub use error::{BuildError, BuildResult, PlanValidationError};
pub use ids::{PipelineBarrier, PipelineId};
pub use lowering::{
    LoweredPlan, LoweringCtx, LoweringCtxExt, Pipeline, PipelineBuilder, PipelineSink,
    PipelineSource, PipelineTail, PipelineTransform,
};
pub use plan::{Operator, PhysicalPlan};
pub use spawn::{IoCost, Priority, SpawnRuntime, WorkHandle};
