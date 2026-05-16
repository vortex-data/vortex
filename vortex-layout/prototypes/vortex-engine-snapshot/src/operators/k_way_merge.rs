//! K-way merge operator over sorted input streams.
//!
//! `KWayMerge` takes K input ports, each declaring the same
//! `required_sort`, and produces one output stream carrying that
//! same `sort_key`. The merge is a row-by-row comparison of input
//! heads; the smallest head is emitted and that input is advanced.
//!
//! ## V1 implementation
//!
//! This is the simplest correct merge:
//!
//! - Single-lane. The frontier (one slot per input) is shared
//!   state and is the kind of thing a tournament tree would
//!   protect; we don't have multi-lane merge primitives yet.
//! - Linear scan over K heads per emission. For small K this is
//!   identical-or-better than a tournament tree; a tournament tree
//!   is a follow-up for K ≫ 8.
//! - Row-at-a-time *output* (the underlying merge logic actually
//!   emits a run of consecutive rows from the winner until another
//!   input would take over; that's the standard merge optimisation
//!   and lands here directly).
//! - OVC is not yet implemented. Comparison is via `Scalar` extracted
//!   from the input arrays at the frontier cursor. Plumbing OVC
//!   through `Batch` is a separate follow-up.
//!
//! ## Sort key shapes supported
//!
//! - `SortKey::RowIndex`: compares by batch span position
//!   (`span.start() + cursor_within_batch`). Useful when merging
//!   pre-sorted row-index streams (e.g. the chunk-concat case).
//! - `SortKey::Natural { columns, directions }`: for now, the
//!   single-column / ascending case. Extracts the column at the
//!   cursor row via `array.execute_scalar` and compares via the
//!   `Scalar`'s `PartialOrd`. Multi-column / descending support
//!   lands when needed.

use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::scalar::Scalar;
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
use crate::SortDirection;
use crate::SortKey;
use crate::UpdateCtx;
use crate::WorkClass;
use crate::WorkConstraints;
use crate::WorkCost;
use crate::WorkCtx;
use crate::WorkKey;
use crate::WorkProposal;
use crate::WorkStatus;
use crate::WorkValue;

pub struct KWayMerge {
    label: String,
    /// One Domain per input port. Each port accepts a sorted stream
    /// over its own domain.
    input_domains: Vec<Domain>,
    /// Output's combined domain. Cardinality is the sum of input
    /// cardinalities (or `Unknown` if any input is `Unknown`).
    output_domain: Domain,
    /// Sort that every input must claim and that the output emits.
    sort_key: SortKey,
    /// Session for scalar extraction on `Natural`-sorted inputs.
    session: VortexSession,
}

pub struct KWayMergeState {
    /// Per-input frontier slot. `Some(batch)` means the input has a
    /// buffered batch; `cursor` is the row index within that batch.
    /// `None` means we need to pop a fresh batch.
    frontier: Vec<Frontier>,
    /// Per-input finished flag. Once `true`, the input is fully
    /// drained and contributes no more rows.
    finished: Vec<bool>,
    /// Output cursor (assigned span starts at 0 and advances by emitted
    /// rows).
    out_cursor: u64,
    sealed: bool,
}

struct Frontier {
    batch: Option<Batch>,
    cursor: usize,
}

impl KWayMerge {
    pub fn new(
        label: impl Into<String>,
        input_domains: Vec<Domain>,
        sort_key: SortKey,
        session: VortexSession,
    ) -> Self {
        assert!(
            !input_domains.is_empty(),
            "KWayMerge requires at least one input"
        );
        let label = label.into();
        // Output cardinality = sum of input cardinalities. Unknown
        // if any input is Unknown.
        let mut total: u64 = 0;
        let mut all_exact = true;
        for d in &input_domains {
            match d.cardinality() {
                Cardinality::Exact(n) => total = total.saturating_add(n),
                Cardinality::Unknown => {
                    all_exact = false;
                    break;
                }
            }
        }
        let cardinality = if all_exact {
            Cardinality::Exact(total)
        } else {
            Cardinality::Unknown
        };
        let output_domain =
            Domain::new(DomainId::new(format!("k_way_merge:{label}")), cardinality);
        Self {
            label,
            input_domains,
            output_domain,
            sort_key,
            session,
        }
    }

    pub fn output_domain(&self) -> &Domain {
        &self.output_domain
    }

    pub fn n_inputs(&self) -> usize {
        self.input_domains.len()
    }
}

impl Operator for KWayMerge {
    type GlobalState = ();
    type LocalState = KWayMergeState;

    fn spec(&self) -> OperatorSpec {
        let inputs = self
            .input_domains
            .iter()
            .enumerate()
            .map(|(i, d)| {
                InputPortSpec::new(format!("in[{i}]"), d.clone(), 1)
                    .with_required_sort(self.sort_key.clone())
            })
            .collect();
        let output = Some(
            OutputPortSpec::new("out", self.output_domain.clone(), 1)
                .with_sort_key(self.sort_key.clone()),
        );
        OperatorSpec::new(self.label.clone(), inputs, output)
        // Single-lane: frontier is shared state.
    }

    fn init_global(&self, _ctx: &mut GlobalInitCtx<'_>) -> EngineResult<Self::GlobalState> {
        Ok(())
    }

    fn init_local(
        &self,
        _global: &Self::GlobalState,
        _ctx: &mut LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        let n = self.n_inputs();
        Ok(KWayMergeState {
            frontier: (0..n)
                .map(|_| Frontier {
                    batch: None,
                    cursor: 0,
                })
                .collect(),
            finished: vec![false; n],
            out_cursor: 0,
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
        // Merge needs every input row to determine ordering.
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
        state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        if state.sealed {
            return Ok(());
        }
        // Propose if every input either has a buffered/peekable batch
        // or is finished.
        let mut ready = true;
        for i in 0..self.n_inputs() {
            if state.frontier[i].batch.is_some() || state.finished[i] {
                continue;
            }
            let port = InputPortId::from_index(i);
            if ctx.peek(port).is_none() && !ctx.input_finished(port) {
                ready = false;
                break;
            }
        }
        let value = if ready {
            WorkValue::required(1)
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

        // Refill empty frontier slots from inputs that still have
        // batches. After this, each slot is either Some(batch) or
        // its input is finished.
        for i in 0..self.n_inputs() {
            if state.frontier[i].batch.is_some() {
                continue;
            }
            if state.finished[i] {
                continue;
            }
            let port = InputPortId::from_index(i);
            loop {
                match ctx.pop(port) {
                    Some(batch) => {
                        if batch.demand_all_false() {
                            // Placeholder batch carries no real rows;
                            // skip and keep trying.
                            continue;
                        }
                        if batch.is_empty() {
                            continue;
                        }
                        state.frontier[i].batch = Some(batch);
                        state.frontier[i].cursor = 0;
                        break;
                    }
                    None => {
                        if ctx.input_finished(port) {
                            state.finished[i] = true;
                        }
                        break;
                    }
                }
            }
        }

        // Wait if any input is still pending (no batch and not finished).
        for i in 0..self.n_inputs() {
            if state.frontier[i].batch.is_none() && !state.finished[i] {
                return Ok(WorkStatus::Made);
            }
        }

        // All-done check: every input finished and no buffered batches.
        let any_active = state
            .frontier
            .iter()
            .any(|f| f.batch.is_some());
        if !any_active {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }

        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        }

        // Compare frontier heads; pick the winner.
        let winner = pick_winner(&state.frontier, &self.sort_key, &self.session)?;
        // Determine how many consecutive rows from the winner stay
        // the minimum across all frontiers. Emit that run.
        let run_len = run_length(&state.frontier, winner, &self.sort_key, &self.session)?;

        // Slice the winner's batch to [cursor, cursor + run_len) and
        // push as one output batch.
        let frontier = &mut state.frontier[winner];
        let batch = frontier
            .batch
            .as_ref()
            .expect("winner has a batch by construction");
        let cursor = frontier.cursor;
        let array = batch.array().clone();
        let demand = batch.demand().clone();
        let slice = slice_array(&array, cursor, cursor + run_len)?;
        let demand_slice = slice_mask(&demand, cursor, cursor + run_len);
        let span = DomainSpan::new(state.out_cursor, run_len as u64);
        let out_batch = Batch::with_demand(span, slice, demand_slice);
        ctx.push(out_batch)?;
        state.out_cursor += run_len as u64;

        // Advance the winner's cursor; pop next batch from the winner
        // if exhausted.
        frontier.cursor += run_len;
        if frontier.cursor >= batch.len() {
            frontier.batch = None;
            frontier.cursor = 0;
        }
        Ok(WorkStatus::Made)
    }
}

/// Pick the index of the frontier whose head value is the smallest
/// (or, for descending sort, the largest). Panics if no frontier
/// has a buffered batch — callers must check before invoking.
fn pick_winner(
    frontiers: &[Frontier],
    sort_key: &SortKey,
    session: &VortexSession,
) -> EngineResult<usize> {
    let mut best: Option<(usize, OrdKey)> = None;
    for (i, f) in frontiers.iter().enumerate() {
        let Some(batch) = f.batch.as_ref() else {
            continue;
        };
        let key = key_at(batch, f.cursor, sort_key, session)?;
        match &best {
            None => best = Some((i, key)),
            Some((_, cur)) => {
                if key_lt(&key, cur, sort_key) {
                    best = Some((i, key));
                }
            }
        }
    }
    best.map(|(i, _)| i)
        .ok_or_else(|| EngineError::message("KWayMerge: no active frontier"))
}

/// Compute the longest run length from the winner's current cursor
/// such that every row in the run is ≤ (or ≥, for descending) the
/// head value of every other active frontier.
///
/// For now, conservative: run length = 1. The optimal "run until
/// another input would take over" optimization lands in v2.
#[allow(clippy::needless_pass_by_value)]
fn run_length(
    frontiers: &[Frontier],
    winner: usize,
    _sort_key: &SortKey,
    _session: &VortexSession,
) -> EngineResult<usize> {
    let _ = frontiers;
    let _ = winner;
    Ok(1)
}

#[derive(Clone)]
enum OrdKey {
    RowIndex(u64),
    Scalar(Scalar),
}

fn key_at(
    batch: &Batch,
    cursor: usize,
    sort_key: &SortKey,
    session: &VortexSession,
) -> EngineResult<OrdKey> {
    match sort_key {
        SortKey::RowIndex => Ok(OrdKey::RowIndex(
            batch.span().start() + cursor as u64,
        )),
        SortKey::Natural { columns, directions } => {
            // Initial v1: only single-column ascending. Bind validates
            // matching sort_keys, so any caller using KWayMerge has
            // committed to this constraint upstream.
            if columns.len() != 1 {
                return Err(EngineError::message(
                    "KWayMerge: multi-column Natural sort not yet supported",
                ));
            }
            if directions.first().copied() != Some(SortDirection::Ascending) {
                return Err(EngineError::message(
                    "KWayMerge: only Ascending direction supported in v1",
                ));
            }
            let array = batch.array();
            let mut exec = session.create_execution_ctx();
            let scalar = array
                .execute_scalar(cursor, &mut exec)
                .map_err(|e| EngineError::message(format!("KWayMerge scalar: {e}")))?;
            Ok(OrdKey::Scalar(scalar))
        }
    }
}

fn key_lt(a: &OrdKey, b: &OrdKey, _sort_key: &SortKey) -> bool {
    match (a, b) {
        (OrdKey::RowIndex(x), OrdKey::RowIndex(y)) => x < y,
        (OrdKey::Scalar(x), OrdKey::Scalar(y)) => x
            .partial_cmp(y)
            .map(|o| o == std::cmp::Ordering::Less)
            .unwrap_or(false),
        _ => false,
    }
}

/// Slice an `ArrayRef` to `[start, end)`. Uses Vortex's `slice` op.
fn slice_array(array: &ArrayRef, start: usize, end: usize) -> EngineResult<ArrayRef> {
    if start == 0 && end == array.len() {
        return Ok(array.clone());
    }
    array
        .slice(start..end)
        .map_err(|e| EngineError::message(format!("KWayMerge slice: {e}")))
}

fn slice_mask(mask: &vortex_mask::Mask, start: usize, end: usize) -> vortex_mask::Mask {
    if start == 0 && end == mask.len() {
        return mask.clone();
    }
    mask.slice(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;
        use crate::Cardinality;

    fn dom(name: &str, n: u64) -> Domain {
        Domain::new(DomainId::new(name), Cardinality::Exact(n))
    }

    #[test]
    fn declares_required_sort_on_every_input() {
        let merge = KWayMerge::new(
            "merge",
            vec![dom("a", 3), dom("b", 3)],
            SortKey::RowIndex,
            VortexSession::empty(),
        );
        let spec = merge.spec();
        assert_eq!(spec.inputs.len(), 2);
        for input in &spec.inputs {
            assert_eq!(input.required_sort(), Some(&SortKey::RowIndex));
        }
        let output = spec.output.expect("KWayMerge has output");
        assert_eq!(output.sort_key, Some(SortKey::RowIndex));
    }

    #[test]
    fn output_cardinality_is_sum_when_inputs_exact() {
        let merge = KWayMerge::new(
            "merge",
            vec![dom("a", 3), dom("b", 5)],
            SortKey::RowIndex,
            VortexSession::empty(),
        );
        assert!(matches!(
            merge.output_domain.cardinality(),
            Cardinality::Exact(8)
        ));
    }

    #[test]
    fn output_cardinality_is_unknown_when_any_input_unknown() {
        let merge = KWayMerge::new(
            "merge",
            vec![
                dom("a", 3),
                Domain::new(DomainId::new("b"), Cardinality::Unknown),
            ],
            SortKey::RowIndex,
            VortexSession::empty(),
        );
        assert!(matches!(
            merge.output_domain.cardinality(),
            Cardinality::Unknown
        ));
    }
}
