//! Multi-lane streaming aggregate that emits partial-state batches.
//!
//! `PartialAggregate` is the producer side of the
//! `partial-then-merge` aggregate split. It mirrors [`Aggregate`]'s
//! input contract — same per-lane accumulator, same coerce-args /
//! demand-filter / encoding-aware accumulate path — but instead of
//! finalising locally it emits each lane's partial state as a 1-row
//! batch. A downstream [`MergeAggregate`] folds those partials into
//! the final scalar.
//!
//! ## Why split
//!
//! All-in-one [`Aggregate`] is single-lane: the accumulator runs on
//! one worker, so the heavy `try_accumulate` / kernel /
//! canonicalise-then-accumulate work doesn't scale across lanes.
//! When the input is large enough that the accumulator is the
//! bottleneck, splitting buys fan-out: each lane runs its own
//! accumulator over its share of the input, then a cheap
//! [`MergeAggregate`] folds the per-lane partials.
//!
//! ## Lane safety
//!
//! Construction returns an error when the supplied `AggregateFnRef`
//! is not lane-safe (see [`is_lane_safe`]). The lane-safe set today
//! is `Sum`, `Count`, `MinMax`, `Mean::combined()`, `NanCount` —
//! aggregates whose partial state combines associatively /
//! commutatively. Order- or globally-stateful aggregates (`First`,
//! `Last`, `IsConstant`, ...) cannot be split and must use
//! [`Aggregate`].
//!
//! ## Output shape
//!
//! Output domain has cardinality `Exact(L)` where `L` is the lane
//! count chosen by the scheduler — each lane emits exactly one
//! partial. Output dtype is the partial-state dtype reported by
//! `aggregate_fn.state_dtype(accumulator_dtype)` (e.g.
//! `struct(sum: f64, count: u64)` for `Mean::combined()` over f64).
//! Each lane wraps its flushed partial scalar in a 1-row constant
//! array via [`scalar_to_array`].
//!
//! [`Aggregate`]: super::Aggregate
//! [`MergeAggregate`]: super::MergeAggregate
//! [`is_lane_safe`]: super::aggregate_common::is_lane_safe
//! [`scalar_to_array`]: super::aggregate_common::scalar_to_array

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_session::VortexSession;

use super::aggregate_common::is_lane_safe;
use super::aggregate_common::scalar_to_array;

use crate::Batch;
use crate::Cardinality;
use crate::Domain;
use crate::DomainId;
use crate::DomainSpan;
use crate::EngineError;
use crate::EngineResult;
use crate::GlobalInitCtx;
use crate::InputPortId;
use crate::InputPortSpec;
use crate::LocalInitCtx;
use crate::Operator;
use crate::OperatorSpec;
use crate::OutputPortSpec;
use crate::RequirementCtx;
use crate::RequirementSet;
use crate::UpdateCtx;
use crate::WorkClass;
use crate::WorkConstraints;
use crate::WorkCost;
use crate::WorkCtx;
use crate::WorkKey;
use crate::WorkProposal;
use crate::WorkStatus;
use crate::WorkValue;

pub struct PartialAggregate {
    label: String,
    input_domain: Domain,
    /// Output domain ID; the actual cardinality is filled in from
    /// the lane count at preparation time, but for `Cardinality`
    /// purposes we mark it `Unknown` because the lane count isn't
    /// known until the scheduler picks one. Downstream operators
    /// shouldn't depend on partial-state cardinality being fixed.
    output_domain: Domain,
    /// Dtype the accumulator runs over (after `coerce_args`).
    accumulator_dtype: DType,
    /// Partial-state dtype reported by `state_dtype`. Each lane
    /// emits a 1-row constant array of this dtype.
    state_dtype: DType,
    aggregate_fn: AggregateFnRef,
    session: VortexSession,
}

impl PartialAggregate {
    pub fn new(
        label: impl Into<String>,
        input_domain: Domain,
        input_dtype: DType,
        aggregate_fn: AggregateFnRef,
        session: VortexSession,
    ) -> EngineResult<Self> {
        let label = label.into();
        if !is_lane_safe(&aggregate_fn) {
            return Err(EngineError::message(format!(
                "aggregate {aggregate_fn} is not lane-safe; cannot use \
                 PartialAggregate (use Aggregate instead)",
            )));
        }
        let accumulator_dtype = aggregate_fn
            .coerce_args(&input_dtype)
            .map_err(|e| EngineError::message(format!("partial coerce_args: {e}")))?;
        let state_dtype = aggregate_fn
            .state_dtype(&accumulator_dtype)
            .ok_or_else(|| {
                EngineError::message(format!(
                    "aggregate {aggregate_fn} has no partial-state dtype",
                ))
            })?;
        // Partial output cardinality = number of lanes. We don't
        // know that until the scheduler picks lane_count; declare
        // Unknown.
        let output_domain = Domain::new(
            DomainId::new(format!("partial_aggregate:{label}")),
            Cardinality::Unknown,
        );
        Ok(Self {
            label,
            input_domain,
            output_domain,
            accumulator_dtype,
            state_dtype,
            aggregate_fn,
            session,
        })
    }

    pub fn output_domain(&self) -> &Domain {
        &self.output_domain
    }

    pub fn state_dtype(&self) -> &DType {
        &self.state_dtype
    }
}

pub struct PartialAggregateGlobalState {
    /// Decremented by each lane when it has emitted its partial.
    /// The lane that drops it to zero seals the output.
    lanes_remaining: AtomicUsize,
}

pub struct PartialAggregateState {
    accumulator: AccumulatorRef,
    /// True once this lane has emitted its partial-state row.
    emitted: bool,
    /// True once the output has been sealed by the last lane out.
    sealed: bool,
    /// Output cursor for this lane's partial. With one row per lane
    /// and `lane_count` lanes total, lane `i` claims span `[i, i+1)`
    /// on the output domain.
    lane_index: u64,
}

impl Operator for PartialAggregate {
    type GlobalState = PartialAggregateGlobalState;
    type LocalState = PartialAggregateState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.input_domain.clone(), 1)],
            Some(OutputPortSpec::new(
                "out",
                self.output_domain.clone(),
                1,
            )),
        )
        .lanes(None)
    }

    fn init_global(&self, _ctx: &mut GlobalInitCtx<'_>) -> EngineResult<Self::GlobalState> {
        Ok(PartialAggregateGlobalState {
            lanes_remaining: AtomicUsize::new(0),
        })
    }

    fn init_local(
        &self,
        global: &Self::GlobalState,
        ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        let lane_count = ctx.lane_count();
        // First lane to init bumps lanes_remaining up to lane_count;
        // subsequent CASes are no-ops.
        let _ = global.lanes_remaining.compare_exchange(
            0,
            lane_count,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        let accumulator = self
            .aggregate_fn
            .accumulator(&self.accumulator_dtype)
            .map_err(|e| EngineError::message(format!("partial accumulator: {e}")))?;
        Ok(PartialAggregateState {
            accumulator,
            emitted: false,
            sealed: false,
            lane_index: ctx.lane().index as u64,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
        _output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        let len = match self.input_domain.cardinality() {
            Cardinality::Exact(len) => len,
            Cardinality::Unknown => u64::MAX / 2,
        };
        inputs[0].require_span(DomainSpan::new(0, len));
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        local: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        if local.sealed {
            return Ok(());
        }
        let port = InputPortId::from_index(0);
        let drainable = ctx.peek(port).is_some() || ctx.input_finished(port);
        let class = if drainable {
            WorkClass::Emit
        } else {
            WorkClass::Cpu
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            class,
            WorkValue::required(1),
            WorkCost::small_cpu(),
            WorkConstraints::output_capacity(),
        ));
        Ok(())
    }

    fn run(
        &self,
        global: &Self::GlobalState,
        local: &mut Self::LocalState,
        _work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus> {
        if local.sealed {
            return Ok(WorkStatus::Finished);
        }

        // Phase 1: drain the input and accumulate. Same shape as
        // `Aggregate::run` — the per-lane accumulator handles
        // try_accumulate, encoding-aware kernels, and the
        // canonicalise-then-accumulate fallback internally.
        if !local.emitted {
            while let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
                if batch.demand_all_false() {
                    continue;
                }
                let demand = batch.demand().clone();
                let array = batch.into_array();
                let array = if demand.all_true() {
                    array
                } else {
                    array.filter(demand).map_err(|e| {
                        EngineError::message(format!("partial demand filter: {e}"))
                    })?
                };
                let array = if array.dtype() == &self.accumulator_dtype {
                    array
                } else {
                    array.cast(self.accumulator_dtype.clone()).map_err(|e| {
                        EngineError::message(format!("partial input cast: {e}"))
                    })?
                };
                let mut exec = self.session.create_execution_ctx();
                local
                    .accumulator
                    .accumulate(&array, &mut exec)
                    .map_err(|e| EngineError::message(format!("partial accumulate: {e}")))?;
                if local.accumulator.is_saturated() {
                    break;
                }
            }
            let input_done = ctx.input_finished(InputPortId::from_index(0))
                || local.accumulator.is_saturated();
            if !input_done {
                return Ok(WorkStatus::Made);
            }

            if !ctx.has_capacity() {
                return Ok(WorkStatus::Made);
            }

            // Flush the partial state as a 1-row constant array on
            // this lane's slice of the output domain. Each lane gets
            // exactly one row at `lane_index`.
            let partial = local
                .accumulator
                .flush()
                .map_err(|e| EngineError::message(format!("partial flush: {e}")))?;
            let array = scalar_to_array(&partial);
            let batch = Batch::from_array(DomainSpan::new(local.lane_index, 1), array);
            ctx.push(batch)?;
            local.emitted = true;
        }

        // Phase 2: vote done. Last lane out seals the output.
        let was_last = global.lanes_remaining.fetch_sub(1, Ordering::SeqCst) == 1;
        if was_last {
            ctx.seal()?;
            ctx.trace(format!("{}: sealed output", self.label));
        }
        local.sealed = true;
        Ok(WorkStatus::Finished)
    }
}
