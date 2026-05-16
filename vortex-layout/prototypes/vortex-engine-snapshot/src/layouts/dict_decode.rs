//! `DictDecode`: combiner for engine-visible Dict decomposition.
//!
//! Replaces the dict-internal traversal that lived inside the legacy
//! `DictLayoutOperator`. A `vortex.dict` layout has two children:
//! `values` (auxiliary, typically small, one row per unique value)
//! and `codes` (transparent, one integer index per output row).
//! `bind_into_graph` recursively binds each child to its own
//! subgraph, then wires both into a `DictDecode`:
//!
//! - input 0 (values): drained fully on first run; concatenated if
//!   the values subgraph emitted multiple batches.
//! - input 1 (codes): streamed; each batch is materialised with
//!   [`ArrayRef::take`] against the buffered values and pushed
//!   downstream with the same span.
//!
//! `DictDecode` carries the values dtype on its output port. Spans
//! on the codes input flow through unchanged: code row `i` produces
//! output row `i`, so requirements and ordering are preserved.

use std::task::Context;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;

use crate::Batch;
use crate::Domain;
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

const VALUES_PORT: usize = 0;
const CODES_PORT: usize = 1;

pub struct DictDecode {
    label: String,
    output_domain: Domain,
    values_domain: Domain,
    codes_domain: Domain,
    values_dtype: DType,
    output_columns: usize,
}

pub struct DictDecodeState {
    /// Buffered values, accumulated across however many batches the
    /// values subgraph emits, then concatenated once the values port
    /// seals.
    values_batches: Vec<ArrayRef>,
    /// Concatenated values, set once `values_port_finished` flips.
    values_array: Option<ArrayRef>,
    values_port_finished: bool,
    sealed: bool,
}

impl DictDecode {
    pub fn new(
        label: impl Into<String>,
        output_domain: Domain,
        values_domain: Domain,
        codes_domain: Domain,
        values_dtype: DType,
        output_columns: usize,
    ) -> Self {
        Self {
            label: label.into(),
            output_domain,
            values_domain,
            codes_domain,
            values_dtype,
            output_columns,
        }
    }
}

impl Operator for DictDecode {
    type GlobalState = ();
    type LocalState = DictDecodeState;

    fn spec(&self) -> OperatorSpec {
        let inputs = vec![
            InputPortSpec::new(
                "values",
                self.values_domain.clone(),
                self.output_columns,
            ),
            InputPortSpec::new("codes", self.codes_domain.clone(), 1),
        ];
        let outputs = Some(OutputPortSpec::new(
            "out",
            self.output_domain.clone(),
            self.output_columns,
        ));
        OperatorSpec::new(self.label.clone(), inputs, outputs)
    }

    fn init_global(&self, _ctx: &mut GlobalInitCtx<'_>) -> EngineResult<Self::GlobalState> {
        Ok(())
    }

    fn init_local(
        &self,
        _global: &Self::GlobalState,
        _ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        Ok(DictDecodeState {
            values_batches: Vec::new(),
            values_array: None,
            values_port_finished: false,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        output: &RequirementSet,
        inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        // Values: require everything. The values child is typically
        // tiny (one row per unique value) and we don't know which
        // codes will arrive until we've seen them. EV scheduling can
        // still bias against running it via cost — value selectivity
        // is structurally 1.0 (every distinct value may be
        // referenced somewhere).
        if let crate::Cardinality::Exact(rows) = self.values_domain.cardinality()
            && rows > 0
        {
            inputs[VALUES_PORT].require_span(crate::DomainSpan::new(0, rows));
        }
        // Codes: faithful identity translation — code row `i`
        // produces output row `i`, so the output requirement maps
        // 1:1 onto the codes input. Forwarding the output's
        // demand+selectivity unchanged means upstream pruning
        // (e.g. `ZoneMapOperator`'s per-zone `NotNeeded` once the
        // zone map is built, or its 0.5-prior `Required` while the
        // zone map is still being built) reaches the codes scan.
        if let crate::Cardinality::Exact(_rows) = self.codes_domain.cardinality() {
            inputs[CODES_PORT].clone_from(output);
        }
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        // Phase 1: still draining values. Propose only when values
        // input has something to consume or has sealed; the run
        // method's first phase doesn't emit downstream batches yet.
        if !state.values_port_finished {
            let values_port = InputPortId::from_index(VALUES_PORT);
            let drainable =
                ctx.peek(values_port).is_some() || ctx.input_finished(values_port);
            if !drainable {
                return Ok(());
            }
            ctx.propose(WorkProposal::new(
                WorkKey::from_byte(0),
                WorkClass::Cpu,
                // We know roughly how many values are coming —
                // `values_domain.cardinality()` — but here it's a
                // constant per operator, not a per-batch signal. A
                // tiny placeholder is fine since this proposal
                // exists only to drive the values-drain phase.
                WorkValue::candidate(0, 0),
                WorkCost::small_cpu(),
                WorkConstraints::none(),
            ));
            return Ok(());
        }
        // Phase 2: codes streaming. Only propose when there's a
        // codes batch to decode (or the input has sealed and we
        // need to emit the final seal).
        let codes_port = InputPortId::from_index(CODES_PORT);
        let peeked = ctx.peek(codes_port);
        let finished = ctx.input_finished(codes_port);
        if peeked.is_none() && !finished {
            return Ok(());
        }
        let useful_rows = peeked
            .as_ref()
            .map(|b| b.demand().true_count() as u64)
            .unwrap_or(0);
        let value = if useful_rows > 0 {
            WorkValue::required(useful_rows)
        } else {
            WorkValue::candidate(0, 0)
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            WorkClass::Emit,
            value,
            WorkCost::small_cpu(),
            WorkConstraints::output_capacity(),
        ));
        Ok(())
    }

    fn run(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        _work: WorkKey,
        ctx: &mut WorkCtx<'_>,
    ) -> EngineResult<WorkStatus> {
        if state.sealed {
            return Ok(WorkStatus::Finished);
        }

        // Phase 1: drain values input until sealed; accumulate
        // batches.
        if !state.values_port_finished {
            while let Some(batch) = ctx.pop(InputPortId::from_index(VALUES_PORT)) {
                state.values_batches.push(batch.into_array());
            }
            if ctx.input_finished(InputPortId::from_index(VALUES_PORT)) {
                state.values_port_finished = true;
                state.values_array = Some(concat_values(
                    &self.values_dtype,
                    std::mem::take(&mut state.values_batches),
                )?);
            } else {
                return Ok(WorkStatus::Made);
            }
        }
        let values = state
            .values_array
            .as_ref()
            .expect("values resolved by phase 1");

        // Phase 2: drain whatever codes batches are available; emit
        // decoded batches downstream.
        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        }
        if let Some(batch) = ctx.pop(InputPortId::from_index(CODES_PORT)) {
            let span = batch.span();
            // Codes batch is fully placeholder → output is also a
            // placeholder (no take, no work). Output dtype is the
            // values dtype.
            if batch.demand_all_false() {
                ctx.push(Batch::placeholder(span, self.values_dtype.clone()),
                )?;
                return Ok(WorkStatus::Made);
            }
            let demand = batch.demand().clone();
            let codes = batch.into_array();
            let decoded = values
                .take(codes)
                .map_err(|e| EngineError::message(format!("dict decode take: {e}")))?;
            ctx.push(Batch::with_demand(span, decoded, demand),
            )?;
            ctx.trace(format!("{}: decoded codes span {:?}", self.label, span));
            return Ok(WorkStatus::Made);
        }
        if ctx.input_finished(InputPortId::from_index(CODES_PORT)) {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

/// Concatenate values batches into one array of `dtype`. The
/// values subgraph typically emits a single batch, but a chunked
/// values layout (rare) can produce multiple.
fn concat_values(dtype: &DType, batches: Vec<ArrayRef>) -> EngineResult<ArrayRef> {
    if batches.len() == 1 {
        return Ok(batches.into_iter().next().unwrap());
    }
    if batches.is_empty() {
        return Err(EngineError::message(
            "dict decode: values subgraph emitted no batches",
        ));
    }
    let chunked = ChunkedArray::try_new(batches, dtype.clone())
        .map_err(|e| EngineError::message(format!("concat values: {e}")))?;
    Ok(chunked.into_array())
}
