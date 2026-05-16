//! Single-lane operator that folds partial-state batches from
//! [`PartialAggregate`] into a finalised aggregate scalar.
//!
//! `MergeAggregate` is the consumer side of the
//! `partial-then-merge` aggregate split. It expects an input stream
//! whose dtype matches the partial-state dtype reported by
//! `aggregate_fn.state_dtype(accumulator_dtype)` (e.g.
//! `struct(sum: f64, count: u64)` for `Mean::combined()` over f64).
//! Each input row is extracted as a `Scalar` and stashed; on seal
//! the operator dispatches to [`merge_partials`] for the typed
//! `empty_partial` → `combine_partials` → `finalize_scalar` fold,
//! emits the finalised scalar as a 1-row batch, and seals.
//!
//! ## Why single-lane
//!
//! The partial-state stream is small by design: one partial per
//! upstream `PartialAggregate` lane (e.g. M shards × L lanes per
//! shard = M·L rows total). The fold itself is cheap — there is no
//! win from running it across multiple lanes, and doing so would
//! reintroduce the cross-lane combine step we just paid to fan out
//! upstream.
//!
//! ## Output shape
//!
//! Output domain has cardinality `Exact(1)`. Output dtype is
//! `aggregate_fn.return_dtype(accumulator_dtype)` — the same final
//! dtype `Aggregate` would produce. An optional capture slot mirrors
//! [`Aggregate::with_capture`] so tests can read the scalar without
//! parsing the 1-row output batch.
//!
//! [`PartialAggregate`]: super::PartialAggregate
//! [`Aggregate::with_capture`]: super::Aggregate::with_capture
//! [`merge_partials`]: super::aggregate_common::merge_partials

use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_session::VortexSession;

use super::aggregate_common::is_lane_safe;
use super::aggregate_common::merge_partials;
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

pub struct MergeAggregate {
    label: String,
    input_domain: Domain,
    output_domain: Domain,
    /// Dtype the underlying accumulator was built for (after
    /// `coerce_args`). Required input to `merge_partials`'
    /// `empty_partial` step — must match what the producing
    /// [`PartialAggregate`] used.
    accumulator_dtype: DType,
    aggregate_fn: AggregateFnRef,
    session: VortexSession,
    captured: Option<Arc<Mutex<Option<Scalar>>>>,
}

impl MergeAggregate {
    /// Build a merge for the given aggregate function. `input_dtype`
    /// is the original (pre-coerce) input dtype the producing
    /// `PartialAggregate` was bound with — this constructor
    /// re-derives the `accumulator_dtype` via `coerce_args` so the
    /// caller doesn't have to thread it through.
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
                "aggregate {aggregate_fn} is not lane-safe; \
                 nothing to merge",
            )));
        }
        let accumulator_dtype = aggregate_fn
            .coerce_args(&input_dtype)
            .map_err(|e| EngineError::message(format!("merge coerce_args: {e}")))?;
        let output_domain = Domain::new(
            DomainId::new(format!("merge_aggregate:{label}")),
            Cardinality::Exact(1),
        );
        Ok(Self {
            label,
            input_domain,
            output_domain,
            accumulator_dtype,
            aggregate_fn,
            session,
            captured: None,
        })
    }

    pub fn with_capture(mut self, slot: Arc<Mutex<Option<Scalar>>>) -> Self {
        self.captured = Some(slot);
        self
    }

    pub fn output_domain(&self) -> &Domain {
        &self.output_domain
    }
}

pub struct MergeAggregateState {
    /// Per-shard partial scalars accumulated from the input stream.
    /// Grows by at most one row per input batch; `M·L` rows total
    /// (shards × lanes-per-shard) — small by construction.
    partials: Vec<Scalar>,
    emitted: bool,
}

impl Operator for MergeAggregate {
    type GlobalState = ();
    type LocalState = MergeAggregateState;

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
    }

    fn init_global(&self, _ctx: &mut GlobalInitCtx<'_>) -> EngineResult<Self::GlobalState> {
        Ok(())
    }

    fn init_local(
        &self,
        _global: &Self::GlobalState,
        _ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        Ok(MergeAggregateState {
            partials: Vec::new(),
            emitted: false,
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
        // Need every partial-state row.
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
        if local.emitted {
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
        _global: &Self::GlobalState,
        local: &mut Self::LocalState,
        _work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus> {
        if local.emitted {
            return Ok(WorkStatus::Finished);
        }

        // Drain available partial-state batches; extract each row as
        // a Scalar and stash. Skip placeholder (all-false-demand)
        // batches — those carry no real partial state.
        while let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            if batch.demand_all_false() {
                continue;
            }
            let demand = batch.demand().clone();
            let array = batch.into_array();
            let len = array.len();
            let mut exec = self.session.create_execution_ctx();
            for i in 0..len {
                if !demand.value(i) {
                    continue;
                }
                let scalar = array
                    .execute_scalar(i, &mut exec)
                    .map_err(|e| EngineError::message(format!("merge execute_scalar: {e}")))?;
                local.partials.push(scalar);
            }
        }

        if !ctx.input_finished(InputPortId::from_index(0)) {
            return Ok(WorkStatus::Made);
        }
        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        }

        let partials = std::mem::take(&mut local.partials);
        let scalar = merge_partials(&self.aggregate_fn, &self.accumulator_dtype, partials)?;
        if let Some(slot) = &self.captured {
            *slot.lock() = Some(scalar.clone());
        }
        let array = scalar_to_array(&scalar);
        let batch = Batch::from_array(DomainSpan::new(0, 1), array);
        ctx.push(batch)?;
        ctx.seal()?;
        ctx.trace(format!("{}: emitted merged result", self.label));
        local.emitted = true;
        Ok(WorkStatus::Finished)
    }
}
