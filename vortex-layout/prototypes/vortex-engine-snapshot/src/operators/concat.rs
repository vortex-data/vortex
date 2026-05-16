//! `Concat`: ordered fan-in of N inputs into one output.
//!
//! Each input is a separate domain (typically one source subgraph
//! per file). Output is a fresh synthetic domain whose rows are
//! the in-order concatenation of input rows: all of input 0, then
//! all of input 1, …, then all of input N-1.
//!
//! Compared to [`crate::operators::Union`]:
//!
//! - **Order:** stable. Output[i] comes from `input_k[j]` where the
//!   `(k, j)` pair is determined by row index — `i = sum(len(input_0..k)) + j`.
//!   `Union` makes no such guarantee.
//! - **Concurrency:** `Concat` is single-lane on purpose. Multiple
//!   lanes would race to emit and break ordering. `Union` is
//!   multi-lane (one per input) since it doesn't care.
//! - **Buffering:** none beyond what the channels already do. We
//!   process one input at a time and emit each input's batches
//!   straight through.
//!
//! The single-lane property means `Concat` is structurally slower
//! than `Union` for unordered workloads — use `Union` when the
//! consumer doesn't care about row order, and `Concat` only when
//! it does. The choice belongs at the bind layer (see
//! `BindContext::ordering` and `BindCapabilities::output_ordering`).

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

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

pub struct Concat {
    label: String,
    input_domains: Vec<Domain>,
    output_domain: Domain,
    output_columns: usize,
    /// Optional sort key claimed on the output. If `Some(K)`,
    /// every input port also declares `required_sort = Some(K)`
    /// — the bind layer enforces that each producer satisfies the
    /// claim. The caller is responsible for wiring producers in
    /// key-monotonic order so that drain in source-index order
    /// actually preserves K.
    sort_key: Option<crate::SortKey>,
}

pub struct ConcatGlobalState {
    /// Running cursor into the output domain. Each emitted batch
    /// claims `[cursor, cursor + batch.span.len())` via
    /// `fetch_add(len)`. Atomic only for parity with `Union`'s API
    /// shape — `Concat` is single-lane so contention is not a
    /// concern.
    cursor: AtomicU64,
}

pub struct ConcatState {
    /// Index of the input port currently being drained. Advances
    /// monotonically. Reaching `input_domains.len()` means done.
    cursor_input: usize,
    sealed: bool,
}

impl Concat {
    pub fn new(
        label: impl Into<String>,
        input_domains: Vec<Domain>,
        output_domain: Domain,
        output_columns: usize,
    ) -> Self {
        assert!(
            !input_domains.is_empty(),
            "Concat requires at least one input"
        );
        Self {
            label: label.into(),
            input_domains,
            output_domain,
            output_columns,
            sort_key: None,
        }
    }

    /// Declare that this `Concat` preserves a sort key across its
    /// inputs. Inputs are required to claim the same key; bind
    /// validates. The caller must ensure producers are wired in
    /// key-monotonic order — `Concat` itself can't check that.
    pub fn with_sort_key(mut self, sort_key: crate::SortKey) -> Self {
        self.sort_key = Some(sort_key);
        self
    }

    /// Build a Concat whose output domain has cardinality equal to
    /// the sum of input cardinalities (or `Unknown` if any input
    /// is `Unknown`). Mirrors `Union::with_summed_domain`.
    pub fn with_summed_domain(
        label: impl Into<String>,
        input_domains: Vec<Domain>,
        output_domain_id: crate::DomainId,
        output_columns: usize,
    ) -> Self {
        let cardinality = sum_cardinalities(&input_domains);
        let output_domain = Domain::new(output_domain_id, cardinality);
        Self::new(label, input_domains, output_domain, output_columns)
    }

    pub fn n_inputs(&self) -> usize {
        self.input_domains.len()
    }
}

impl Operator for Concat {
    type GlobalState = ConcatGlobalState;
    type LocalState = ConcatState;

    fn spec(&self) -> OperatorSpec {
        let inputs = self
            .input_domains
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let mut spec =
                    InputPortSpec::new(format!("in[{i}]"), d.clone(), self.output_columns);
                if let Some(sk) = &self.sort_key {
                    spec = spec.with_required_sort(sk.clone());
                }
                spec
            })
            .collect();
        let mut output_spec = OutputPortSpec::new(
            "out",
            self.output_domain.clone(),
            self.output_columns,
        );
        if let Some(sk) = &self.sort_key {
            output_spec = output_spec.with_sort_key(sk.clone());
        }
        // Serial — one lane, runs through inputs in order.
        OperatorSpec::new(self.label.clone(), inputs, Some(output_spec))
    }

    fn init_global(&self, _ctx: &mut GlobalInitCtx<'_>) -> EngineResult<Self::GlobalState> {
        Ok(ConcatGlobalState {
            cursor: AtomicU64::new(0),
        })
    }

    fn init_local(
        &self,
        _global: &Self::GlobalState,
        _ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        Ok(ConcatState {
            cursor_input: 0,
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
        // Same conservative shape as Union: every input row may
        // be needed downstream. Refining per-input requires
        // translating the output requirement back into per-input
        // sub-ranges using the cumulative-length witness, which is
        // future work.
        for (i, slot) in inputs.iter_mut().enumerate() {
            if let Cardinality::Exact(rows) = self.input_domains[i].cardinality()
                && rows > 0
            {
                slot.require_span(DomainSpan::new(0, rows));
            }
        }
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
        // Only propose when the *current* input has a buffered
        // batch or has sealed; we don't speculatively propose
        // against later inputs. That keeps the heap small and
        // matches the operator's single-cursor execution model.
        if local.cursor_input >= self.input_domains.len() {
            return Ok(());
        }
        let port = InputPortId::from_index(local.cursor_input);
        let peeked = ctx.peek(port);
        let finished = ctx.input_finished(port);
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
        global: &Self::GlobalState,
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
        while local.cursor_input < self.input_domains.len() {
            let port = InputPortId::from_index(local.cursor_input);
            if let Some(batch) = ctx.pop(port) {
                let local_span = batch.span();
                let global_start = global
                    .cursor
                    .fetch_add(local_span.len(), Ordering::Relaxed);
                let global_span = DomainSpan::new(global_start, local_span.len());
                let demand = batch.demand().clone();
                let translated =
                    Batch::with_demand(global_span, batch.into_array(), demand);
                ctx.push(translated)?;
                return Ok(WorkStatus::Made);
            }
            if ctx.input_finished(port) {
                // Drained this input; advance to the next. Loop
                // around so we can immediately try popping from
                // the next input on this same `run` call if it has
                // data buffered.
                local.cursor_input += 1;
                continue;
            }
            // Current input has no batch and isn't sealed — wait
            // for more data.
            return Ok(WorkStatus::Made);
        }
        // All inputs drained.
        ctx.seal()?;
        ctx.trace(format!("{}: sealed output", self.label));
        local.sealed = true;
        Ok(WorkStatus::Finished)
    }
}

fn sum_cardinalities(domains: &[Domain]) -> Cardinality {
    let mut sum: u64 = 0;
    for d in domains {
        match d.cardinality() {
            Cardinality::Exact(rows) => sum = sum.saturating_add(rows),
            Cardinality::Unknown => return Cardinality::Unknown,
        }
    }
    Cardinality::Exact(sum)
}
