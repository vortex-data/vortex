//! Single-lane streaming aggregate, erased over [`AggregateFnRef`].
//!
//! `Aggregate` runs any of Vortex's registered aggregate functions
//! (`SUM`, `MIN`, `MAX`, `COUNT`, `MEAN`, etc.) over an input stream
//! of batches. The operator does not parameterise on the vtable —
//! that lets a planner build heterogeneous aggregate pipelines (e.g.
//! `SUM(a), AVG(b), MIN(c)`) without specialising the operator type
//! per query.
//!
//! ## Single-lane and self-contained
//!
//! `Aggregate` is single-lane: it owns one accumulator and finalises
//! the result locally. For aggregates whose partial state combines
//! associatively (`Sum`, `Count`, `MinMax`, `Mean`, `NanCount`) the
//! planner can split work across lanes by binding to
//! [`PartialAggregate`] feeding a fan-in feeding [`MergeAggregate`]
//! instead — `Aggregate` is the all-in-one shape used when there is
//! no parallelism to gain (single shard, tiny input) or when the
//! aggregate isn't lane-safe.
//!
//! Per-lane state is an [`AccumulatorRef`] obtained from
//! `aggregate_fn.accumulator(input_dtype)`; the accumulator handles
//! `try_accumulate` shortcuts, encoding-specific kernels, and the
//! canonicalise-then-accumulate fallback.
//!
//! [`PartialAggregate`]: super::PartialAggregate
//! [`MergeAggregate`]: super::MergeAggregate

use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_session::VortexSession;

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

pub struct Aggregate {
    label: String,
    input_domain: Domain,
    output_domain: Domain,
    /// The dtype this operator's accumulator was built for. Some
    /// aggregate functions (notably `Mean`) expect input coerced to
    /// a wider type — `coerce_args` returns that target dtype, and
    /// `Aggregate` casts incoming arrays to it before accumulating.
    accumulator_dtype: DType,
    aggregate_fn: AggregateFnRef,
    session: VortexSession,
    /// Optional sink for the finalised scalar. Tests use it to fetch
    /// the result without parsing the output batch.
    captured: Option<Arc<Mutex<Option<Scalar>>>>,
}

impl Aggregate {
    pub fn new(
        label: impl Into<String>,
        input_domain: Domain,
        input_dtype: DType,
        aggregate_fn: AggregateFnRef,
        session: VortexSession,
    ) -> EngineResult<Self> {
        let label = label.into();
        let output_domain = Domain::new(
            DomainId::new(format!("aggregate:{label}")),
            Cardinality::Exact(1),
        );
        let accumulator_dtype = aggregate_fn
            .coerce_args(&input_dtype)
            .map_err(|e| EngineError::message(format!("aggregate coerce_args: {e}")))?;
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

    /// Capture the finalised scalar into a shared slot. The output
    /// batch is still emitted regardless.
    pub fn with_capture(mut self, slot: Arc<Mutex<Option<Scalar>>>) -> Self {
        self.captured = Some(slot);
        self
    }

    pub fn output_domain(&self) -> &Domain {
        &self.output_domain
    }
}

pub struct AggregateState {
    accumulator: AccumulatorRef,
    /// True once we've pushed the final result and sealed our
    /// output. `update` stops proposing work after this.
    emitted: bool,
}

impl Operator for Aggregate {
    type GlobalState = ();
    type LocalState = AggregateState;

    fn spec(&self) -> OperatorSpec {
        let return_dtype = self.aggregate_fn.return_dtype(&self.accumulator_dtype);
        let output_cols = match return_dtype {
            Some(DType::Struct(fields, _)) => fields.nfields(),
            _ => 1,
        };
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.input_domain.clone(), 1)],
            Some(OutputPortSpec::new(
                "out",
                self.output_domain.clone(),
                output_cols,
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
        let accumulator = self
            .aggregate_fn
            .accumulator(&self.accumulator_dtype)
            .map_err(|e| EngineError::message(format!("aggregate accumulator: {e}")))?;
        Ok(AggregateState {
            accumulator,
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
        };
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

        // Drain available batches and accumulate. The accumulator
        // handles `try_accumulate` (metadata short-circuit),
        // encoding-aware kernels, and the canonicalise-then-accumulate
        // fallback internally.
        while let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            if batch.demand_all_false() {
                continue;
            }
            let demand = batch.demand().clone();
            let array = batch.into_array();
            let array = if demand.all_true() {
                array
            } else {
                array
                    .filter(demand)
                    .map_err(|e| EngineError::message(format!("aggregate demand filter: {e}")))?
            };
            let array = if array.dtype() == &self.accumulator_dtype {
                array
            } else {
                array
                    .cast(self.accumulator_dtype.clone())
                    .map_err(|e| EngineError::message(format!("aggregate input cast: {e}")))?
            };
            let mut exec = self.session.create_execution_ctx();
            local
                .accumulator
                .accumulate(&array, &mut exec)
                .map_err(|e| EngineError::message(format!("aggregate accumulate: {e}")))?;
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
        let scalar = local
            .accumulator
            .finish()
            .map_err(|e| EngineError::message(format!("aggregate finish: {e}")))?;
        if let Some(slot) = &self.captured {
            *slot.lock() = Some(scalar.clone());
        }
        let array = scalar_to_array(&scalar);
        let batch = Batch::from_array(DomainSpan::new(0, 1), array);
        ctx.push(batch)?;
        ctx.seal()?;
        ctx.trace(format!("{}: emitted aggregate result", self.label));
        local.emitted = true;
        Ok(WorkStatus::Finished)
    }
}

