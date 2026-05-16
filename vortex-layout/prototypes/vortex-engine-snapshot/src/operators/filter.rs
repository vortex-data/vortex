//! Native filter + projection operator.
//!
//! `Filter` applies a *predicate* expression and a *projection*
//! expression to incoming batches without going through any of
//! `LayoutReader`'s `*_evaluation` paths. For each batch it:
//!
//! 1. evaluates `predicate` over the batch to produce a `Mask`;
//! 2. filters the batch by that mask;
//! 3. applies `projection` to the filtered rows;
//! 4. pushes the projected batch downstream if non-empty.
//!
//! Output cardinality is unknown — depends entirely on filter
//! selectivity. Output spans are consecutive starting at 0; the
//! produced spans are informational since downstream consumers of
//! filtered streams (sinks, aggregates) consume in arrival order
//! without referencing the input domain.
//!
//! When `predicate` is `lit(true)` (or otherwise statically known
//! to match every row) and `projection` is `root()`, this operator
//! is a no-op pass-through; for now it still runs the kernels.

use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_mask::Mask;
use vortex_session::VortexSession;

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

pub struct Filter {
    label: String,
    input_domain: Domain,
    output_domain: Domain,
    input_dtype: DType,
    predicate: Expression,
    projection: Expression,
    session: VortexSession,
}

pub struct FilterState {
    /// Cumulative output rows pushed so far. Used to compute the
    /// next batch's start offset within the (unknown-size) output
    /// domain.
    output_cursor: u64,
    sealed: bool,
}

impl Filter {
    pub fn new(
        label: impl Into<String>,
        input_dtype: DType,
        input_row_count: u64,
        predicate: Expression,
        projection: Expression,
        session: VortexSession,
    ) -> Self {
        let label = label.into();
        let input_domain = Domain::new(
            DomainId::new(format!("filter_in:{label}")),
            Cardinality::Exact(input_row_count),
        );
        let output_domain = Domain::new(
            DomainId::new(format!("filter_out:{label}")),
            Cardinality::Unknown,
        );
        Self {
            label,
            input_domain,
            output_domain,
            input_dtype,
            predicate,
            projection,
            session,
        }
    }

    pub fn output_domain(&self) -> &Domain {
        &self.output_domain
    }
}

impl Operator for Filter {
    type GlobalState = ();
    type LocalState = FilterState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            vec![InputPortSpec::new("in", self.input_domain.clone(), 1)],
            Some(OutputPortSpec::new("out", self.output_domain.clone(), 1)),
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
        Ok(FilterState {
            output_cursor: 0,
            sealed: false,
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
        // Filter needs the full input row range — selectivity isn't
        // known at requirement-translation time. A future
        // refinement: if the filter is a pruning expression (e.g.
        // `col(x) == lit(c)`) and a zoned source is upstream, push
        // the predicate up via a resource.
        // The scheduler clears each `inputs[i]` slot before this call,
        // so we mutate in place — preserves the slot's `Vec<RowInterval>`
        // allocation across turns.
        if let Cardinality::Exact(rows) = self.input_domain.cardinality()
            && rows > 0
        {
            inputs[0].require_span(DomainSpan::new(0, rows));
        }
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        _local: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
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
        if local.sealed {
            return Ok(WorkStatus::Finished);
        }
        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        }

        if let Some(batch) = ctx.pop(InputPortId::from_index(0)) {
            // Placeholder batches (all-don't-care) contribute
            // nothing — the values are garbage so any predicate
            // result on them is meaningless. Skip outright.
            if batch.demand_all_false() {
                return Ok(WorkStatus::Made);
            }
            let demand = batch.demand().clone();
            let array = batch.into_array();
            let array_len = array.len();

            // Evaluate the predicate over the array. For demand-
            // false rows the result is garbage; we AND with demand
            // before filtering so they get nuked.
            let pred_array = array
                .clone()
                .apply(&self.predicate)
                .map_err(|e| EngineError::message(format!("filter predicate apply: {e}")))?;
            let mut exec = self.session.create_execution_ctx();
            let pred_mask: Mask = pred_array
                .execute::<Mask>(&mut exec)
                .map_err(|e| EngineError::message(format!("filter predicate execute: {e}")))?;
            let final_mask = if demand.all_true() {
                pred_mask
            } else {
                use std::ops::BitAnd;
                pred_mask.bitand(&demand)
            };

            if final_mask.all_false() {
                return Ok(WorkStatus::Made);
            }
            let filtered = if final_mask.all_true() {
                array
            } else {
                array
                    .filter(final_mask)
                    .map_err(|e| EngineError::message(format!("filter: {e}")))?
            };
            let projected = filtered
                .apply(&self.projection)
                .map_err(|e| EngineError::message(format!("filter projection apply: {e}")))?;
            let len = projected.len() as u64;
            if len == 0 {
                return Ok(WorkStatus::Made);
            }
            let span = DomainSpan::new(local.output_cursor, len);
            local.output_cursor += len;
            // Filter clears the demand mask: every output row is a
            // genuine match, so the output's demand is all-true.
            ctx.push(Batch::from_array(span, projected),
            )?;
            ctx.trace(format!(
                "{}: filtered {} → {} rows",
                self.label, array_len, len
            ));
            return Ok(WorkStatus::Made);
        }

        if ctx.input_finished(InputPortId::from_index(0)) {
            local.sealed = true;
            ctx.seal()?;
            // input_dtype kept for future requirement-translation
            // pass; reference here suppresses dead-field warnings.
            let _ = &self.input_dtype;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}
