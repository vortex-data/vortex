//! `ChunkConcat`: combiner for engine-visible Chunked decomposition.
//!
//! Replaces the per-chunk traversal that lived inside the legacy
//! `ChunkedLayoutOperator`. Each chunk of a `vortex.chunked` layout
//! becomes its own subgraph (recursively bound via
//! `bind_into_graph`), and `ChunkConcat` is the explicit boundary
//! that gathers their per-chunk-local outputs into a single ordered
//! stream over the parent layout's combined domain.
//!
//! Inputs: N (one per chunk). Each input is wired to the output of
//! that chunk's subgraph and carries spans in the *chunk-local*
//! domain `[0, chunk_rows[i])`.
//!
//! Output: 1, carrying spans in the combined parent domain
//! `[0, sum(chunk_rows))`. Translation is a simple offset add:
//! input `i` batch span `[a, b)` becomes `[chunk_offsets[i] + a,
//! chunk_offsets[i] + b)`.
//!
//! V1 emission policy: drain inputs in chunk order. Input 0 is fully
//! drained (and observed sealed) before input 1 is touched, etc.
//! This preserves the chunked layout's ordered output contract at
//! the cost of pipelining — concurrent chunk decode happens upstream
//! (each chunk subgraph is its own operator) but the gather itself
//! is sequential. Fan-in pipelining is a follow-up that lands when
//! the operator's `Ordered`-versus-`Unordered` output contract is
//! plumbed through.

use std::task::Context;

use crate::Batch;
use crate::Cardinality;
use crate::Domain;
use crate::DomainSpan;
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

pub struct ChunkConcat {
    label: String,
    output_domain: Domain,
    /// Per-chunk input domain (cardinality = chunk row count).
    chunk_domains: Vec<Domain>,
    /// Cumulative row offsets per chunk; `chunk_offsets[i]` is the
    /// global row at which chunk `i`'s rows start. Length = N.
    chunk_offsets: Vec<u64>,
    output_columns: usize,
}

pub struct ChunkConcatState {
    /// Index of the chunk we're currently draining. Advances when
    /// the current input is observed empty + sealed.
    cursor: usize,
    sealed: bool,
}

impl ChunkConcat {
    pub fn new(
        label: impl Into<String>,
        output_domain: Domain,
        chunk_domains: Vec<Domain>,
        chunk_offsets: Vec<u64>,
        output_columns: usize,
    ) -> Self {
        assert_eq!(
            chunk_domains.len(),
            chunk_offsets.len(),
            "ChunkConcat: chunk_domains and chunk_offsets must agree"
        );
        Self {
            label: label.into(),
            output_domain,
            chunk_domains,
            chunk_offsets,
            output_columns,
        }
    }

    pub fn n_chunks(&self) -> usize {
        self.chunk_domains.len()
    }
}

impl Operator for ChunkConcat {
    type GlobalState = ();
    type LocalState = ChunkConcatState;

    fn spec(&self) -> OperatorSpec {
        let inputs = self
            .chunk_domains
            .iter()
            .enumerate()
            .map(|(i, d)| InputPortSpec::new(format!("chunk[{i}]"), d.clone(), self.output_columns))
            .collect();
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
        Ok(ChunkConcatState {
            cursor: 0,
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
        // Translate the downstream requirement (in our combined
        // output domain) into per-chunk requirements. For each
        // chunk i covering output rows
        // `[chunk_offsets[i], chunk_offsets[i] + chunk_rows[i])`,
        // we walk the downstream intervals overlapping that range
        // and emit chunk-local intervals.
        //
        // Intervals not present (Unknown) leave that part of the
        // chunk's input as Unknown, so the per-chunk source waits
        // until the parent's requirement firms up. This is what
        // lets a `ZoneMapOperator` upstream gate the data
        // subgraph until the zones have been pruned.
        // Slots are pre-cleared by the scheduler; mutate in place
        // to preserve their `Vec<RowInterval>` allocations.
        for (i, slot) in inputs.iter_mut().enumerate() {
            let chunk_rows = match self.chunk_domains[i].cardinality() {
                Cardinality::Exact(r) => r,
                Cardinality::Unknown => continue,
            };
            if chunk_rows == 0 {
                continue;
            }
            let chunk_start = self.chunk_offsets[i];
            let chunk_end = chunk_start + chunk_rows;
            for iv in output.intervals() {
                let lo = iv.start.max(chunk_start);
                let hi = iv.end.min(chunk_end);
                if lo >= hi {
                    continue;
                }
                let local_span = DomainSpan::new(lo - chunk_start, hi - lo);
                match iv.demand {
                    crate::RowDemand::Needed => {
                        slot.require_span_with_selectivity(local_span, iv.selectivity);
                    }
                    crate::RowDemand::Candidate => {
                        slot.candidate_span_with_selectivity(local_span, iv.selectivity);
                    }
                    crate::RowDemand::NotNeeded => {
                        slot.not_needed_span(local_span);
                    }
                    crate::RowDemand::Unknown => {
                        // Leave Unknown — don't insert.
                    }
                }
            }
        }
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        // Only propose when there's something to forward: either the
        // chunk at `cursor` has a buffered batch, or it has sealed
        // (so we'd advance the cursor / emit final seal). Without
        // input-driven proposals every transform spammed a
        // `required(1)` proposal on every wake, churning the heap
        // even when no progress was possible.
        let cursor = state.cursor;
        if cursor >= self.chunk_domains.len() {
            return Ok(());
        }
        let port = InputPortId::from_index(cursor);
        let peeked = ctx.peek(port);
        let finished = ctx.input_finished(port);
        if peeked.is_none() && !finished {
            return Ok(());
        }
        // Derive the work's value from the batch we'd emit. The
        // `Required(0..N)` row band still triggers (rows that will
        // really land downstream), so the proposal is comparable
        // with source proposals on the same priority tier.
        let useful_rows = peeked
            .as_ref()
            .map(|b| b.demand().true_count() as u64)
            .unwrap_or(0);
        let value = if useful_rows > 0 {
            WorkValue::required(useful_rows)
        } else {
            // Nothing useful in this batch (all-placeholder, or
            // we're just emitting a seal). Tiny placeholder so the
            // proposal is admissible but doesn't compete with real
            // work.
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

        // Advance past any chunks that are already drained + sealed.
        while state.cursor < self.chunk_domains.len() {
            let port = InputPortId::from_index(state.cursor);
            if ctx.peek(port).is_some() {
                break;
            }
            if ctx.input_finished(port) {
                state.cursor += 1;
                continue;
            }
            // No batch yet, not finished; wait.
            return Ok(WorkStatus::Made);
        }

        if state.cursor >= self.chunk_domains.len() {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }

        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        }

        let port = InputPortId::from_index(state.cursor);
        let Some(batch) = ctx.pop(port) else {
            return Ok(WorkStatus::Made);
        };
        let local = batch.span();
        let global = DomainSpan::new(self.chunk_offsets[state.cursor] + local.start(), local.len());
        // Forward the demand mask unchanged: chunk row `i` maps to
        // exactly one output row, so the demand flows through.
        let demand = batch.demand().clone();
        let translated = Batch::with_demand(global, batch.into_array(), demand);
        ctx.push(translated)?;
        ctx.trace(format!(
            "{}: forwarded chunk[{}] span {:?}",
            self.label, state.cursor, global
        ));
        Ok(WorkStatus::Made)
    }
}
