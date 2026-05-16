//! Source operator for `vortex.flat` layouts.
//!
//! `FlatLayout` is the leaf encoding: one chunk of array data stored
//! as a single segment on disk. This operator reads the segment
//! *natively* — it does not delegate to `LayoutReader::projection_evaluation`.
//! Concretely, on first `run()` the operator:
//!
//! 1. issues `segment_source.request(segment_id)` to fetch the
//!    segment buffer;
//! 2. parses the segment as a `SerializedArray` (using the layout's
//!    pre-stored flatbuffer if present, otherwise the segment's own
//!    header);
//! 3. decodes to an `ArrayRef` of the layout's dtype.
//!
//! The decoded array is cached in `LocalState`, so subsequent
//! `run()` calls slice it for output batches without re-fetching.
//! The output requirement drives which sub-range the cursor emits
//! next; rows marked `NotNeeded` are skipped.
//!
//! No `LayoutReader` is involved. The flat operator owns its
//! `FlatLayout` typed view, the segment source, the array context,
//! and the session needed to drive decode.
//!
//! Output spans are *layout-local* — `[0, layout.row_count())`.
//! Combiners above (`StructAssembler`, `ChunkConcat`, `DictDecode`)
//! translate to global coords as needed.
//!
//! ## Bind-time row pruning
//!
//! Callers of `bind_into_graph` may pass a sub-range of the
//! layout's rows (`row_range`); the flat operator only emits rows
//! within that intersection. Today the operator still loads the
//! full segment — segments are atomic I/O units — but in-memory
//! slicing keeps the emit narrow. Per-batch output sub-ranges are
//! further limited by the output port's `RequirementSet`.
//!
//! ## Limitations
//!
//! - Single-segment loads only (which is the entire point of Flat).
//!   Multi-segment leaves don't exist for this encoding.
//! - The whole array is materialised on first emit. A streaming
//!   "partial decode" path is a follow-up; today's segments are
//!   small enough relative to memory budgets that one-shot decode
//!   is fine.

use std::ops::Range;
use std::sync::Arc;
use std::task::Context;

use vortex_array::ArrayRef;
use vortex_array::serde::SerializedArray;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::segments::SegmentSource;
use vortex_session::VortexSession;

use super::array_estimated_bytes;
use super::column_count_of;
use crate::Batch;
use crate::Cardinality;
use crate::Domain;
use crate::DomainId;
use crate::DomainSpan;
use crate::EngineError;
use crate::EngineResult;
use crate::MemoryReason;
use crate::Operator;
use crate::OperatorSpec;
use crate::OutputPortSpec;
use crate::RequirementCtx;
use crate::RequirementSet;
use crate::RowDemand;
use crate::UpdateCtx;
use crate::WorkClass;
use crate::WorkConstraints;
use crate::WorkCost;
use crate::WorkCtx;
use crate::WorkKey;
use crate::WorkProposal;
use crate::WorkStatus;
use crate::WorkValue;

pub struct FlatLayoutOperator {
    label: String,
    domain: Domain,
    output_columns: usize,
    /// Bind-time row sub-range over the layout's natural rows.
    /// The operator emits no rows outside this range. Defaults to
    /// `0..layout.row_count()`.
    row_range: Range<u64>,
    layout: FlatLayout,
    segment_source: Arc<dyn SegmentSource>,
    session: VortexSession,
    runtime: Arc<CurrentThreadRuntime>,
    max_batch_rows: u64,
}

pub struct FlatLayoutOperatorState {
    cursor: u64,
    /// One-shot decoded array, lazily populated on first `run()`
    /// via `decode_segment`. After this is `Some`, every emit
    /// slices this array; no further I/O happens.
    decoded: Option<ArrayRef>,
    sealed: bool,
}

impl FlatLayoutOperator {
    pub fn new(
        label: impl Into<String>,
        layout: FlatLayout,
        row_range: Range<u64>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        runtime: Arc<CurrentThreadRuntime>,
    ) -> Self {
        let label = label.into();
        let output_columns = column_count_of(layout.dtype());
        let len = row_range.end.saturating_sub(row_range.start);
        let domain = Domain::new(
            DomainId::new(format!("flat:{label}")),
            Cardinality::Exact(len),
        );
        Self {
            label,
            domain,
            output_columns,
            row_range,
            layout,
            segment_source,
            session,
            runtime,
            max_batch_rows: u64::MAX,
        }
    }

    pub fn with_max_batch_rows(mut self, rows: u64) -> Self {
        self.max_batch_rows = rows.max(1);
        self
    }

    pub fn domain(&self) -> &Domain {
        &self.domain
    }
}

impl Operator for FlatLayoutOperator {
    type GlobalState = ();
    type LocalState = FlatLayoutOperatorState;

    fn spec(&self) -> OperatorSpec {
        OperatorSpec::new(
            self.label.clone(),
            Vec::new(),
            Some(OutputPortSpec::new(
                "out",
                self.domain.clone(),
                self.output_columns,
            )),
        )
    }

    fn init_global(
        &self,
        _ctx: &mut crate::GlobalInitCtx<'_>,
    ) -> EngineResult<Self::GlobalState> {
        Ok(())
    }

    fn init_local(
        &self,
        _global: &Self::GlobalState,
        _ctx: &mut crate::LocalInitCtx<'_>,
    ) -> EngineResult<Self::LocalState> {
        Ok(FlatLayoutOperatorState {
            cursor: 0,
            decoded: None,
            sealed: false,
        })
    }

    fn propagate_requirements(
        &self,
        _global: &Self::GlobalState,
        _state: &mut Self::LocalState,
        _output: &RequirementSet,
        _inputs: &mut [RequirementSet],
        _ctx: &RequirementCtx<'_>,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn update(
        &self,
        _global: &Self::GlobalState,
        state: &mut Self::LocalState,
        ctx: &mut UpdateCtx<'_>,
    ) -> EngineResult<()> {
        // EV-aware proposal: derive `WorkValue` from the merged
        // output requirement so that selectivity-discounted demand
        // (e.g. `ZoneMapOperator` publishing a 0.5 prior on the data
        // subgraph while its zone map is still being built) ranks
        // below fully-selective demand (e.g. the zones subgraph
        // itself, where every zone is required for the prune
        // decision).
        //
        // Walking remaining rows `[cursor, local_len)`:
        //   * Required + selectivity FULL  → `required_rows` bucket.
        //     Triggers the scheduler's `row_band` (a static +1B in
        //     `score_operator_proposal`) and outranks any
        //     selectivity-discounted proposal regardless of EV.
        //   * Required (or Candidate) + selectivity < FULL →
        //     `candidate_rows × p_x256`. No row band; EV is the
        //     selectivity-weighted expected useful rows.
        //   * NotNeeded                    → contributes nothing.
        //   * Unknown / no interval        → contributes nothing
        //     (the run loop will wait until propagation refines).
        //
        // Cost stays `small_cpu()` for now; a full bytes-based cost
        // model is a follow-up.
        let local_len = self.row_range.end.saturating_sub(self.row_range.start);
        let cursor = state.cursor;
        let requirement = ctx.output_requirement();
        let mut required_rows: u64 = 0;
        let mut weighted_candidate_rows: u64 = 0;
        for iv in requirement.intervals() {
            if iv.end <= cursor {
                continue;
            }
            let lo = iv.start.max(cursor).min(local_len);
            let hi = iv.end.min(local_len);
            if lo >= hi {
                continue;
            }
            let overlap = hi - lo;
            match iv.demand {
                RowDemand::Needed if iv.selectivity == crate::Selectivity::FULL => {
                    required_rows = required_rows.saturating_add(overlap);
                }
                RowDemand::Needed | RowDemand::Candidate => {
                    let p = u64::from(iv.selectivity.p_x256());
                    weighted_candidate_rows =
                        weighted_candidate_rows.saturating_add(overlap.saturating_mul(p) / 255);
                }
                RowDemand::NotNeeded | RowDemand::Unknown => {}
            }
        }
        let value = if required_rows > 0 {
            // Mix in any candidate contribution as well so a
            // partly-required, partly-candidate range still credits
            // the candidate side; `required_rows > 0` is what the
            // scheduler's row band keys off of.
            WorkValue {
                required_rows,
                candidate_rows: weighted_candidate_rows,
                p_needed_x256: 255,
                memory_release_bytes: 0,
            }
        } else if weighted_candidate_rows > 0 {
            // Encode pre-weighted rows as `candidate_rows` with
            // `p_needed_x256 = 255` so EV in
            // `score_operator_proposal` reads `weighted × 255`.
            WorkValue::candidate(weighted_candidate_rows, 255)
        } else {
            // No demand at all yet — propose a tiny placeholder so the
            // operator stays in the heap and re-fires once propagation
            // catches up.
            WorkValue::candidate(0, 0)
        };
        ctx.propose(WorkProposal::new(
            WorkKey::from_byte(0),
            WorkClass::Cpu,
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

        let local_len = self.row_range.end.saturating_sub(self.row_range.start);
        if local_len == 0 {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }

        let requirement = ctx.output_requirement();

        // Decide what to do next, starting at `state.cursor`.
        // Walk the requirement's interval list (O(K), not O(N)):
        //
        //   - `Required` / `Candidate` interval covering cursor →
        //     emit a real batch (decode segment first, then slice +
        //     demand=true).
        //   - `NotNeeded` interval covering cursor → emit a
        //     placeholder batch (no I/O, demand=false).
        //   - Gap (Unknown — no interval covers cursor) → wait.
        //     Propagation hasn't yet decided whether those rows are
        //     needed; yield and try next turn.
        //   - Cursor walks to local_len through any combination of
        //     Required/NotNeeded intervals → seal.
        let mut decision: Option<EmitDecision> = None;
        for iv in requirement.intervals() {
            if iv.end <= state.cursor {
                continue;
            }
            let iv_start = iv.start.max(state.cursor).min(local_len);
            let iv_end = iv.end.min(local_len);
            if iv_start >= iv_end {
                continue;
            }
            // Gap before this interval (Unknown rows at cursor).
            if iv_start > state.cursor {
                return Ok(WorkStatus::Made);
            }
            // iv covers cursor.
            let cap = state
                .cursor
                .saturating_add(self.max_batch_rows)
                .min(local_len);
            let end = iv_end.min(cap);
            match iv.demand {
                RowDemand::Needed | RowDemand::Candidate => {
                    decision = Some(EmitDecision::Real {
                        start: state.cursor,
                        end,
                    });
                    break;
                }
                RowDemand::NotNeeded => {
                    decision = Some(EmitDecision::Placeholder {
                        start: state.cursor,
                        end,
                    });
                    break;
                }
                RowDemand::Unknown => {
                    // Explicit Unknown interval — same as gap.
                    return Ok(WorkStatus::Made);
                }
            }
        }

        let Some(decision) = decision else {
            // No interval reached cursor: either we walked past the
            // last interval (cursor >= local_len → seal) or the
            // requirement is empty (Unknown → wait).
            if state.cursor >= local_len {
                state.sealed = true;
                ctx.seal()?;
                return Ok(WorkStatus::Finished);
            }
            return Ok(WorkStatus::Made);
        };
        if !ctx.has_capacity() {
            return Ok(WorkStatus::Made);
        }

        let (start, end, batch) = match decision {
            EmitDecision::Real { start, end } => {
                if state.decoded.is_none() {
                    state.decoded = Some(self.decode_segment()?);
                }
                let decoded = state.decoded.as_ref().expect("decoded by guard");
                let abs_start = (self.row_range.start + start) as usize;
                let abs_end = (self.row_range.start + end) as usize;
                let array = if abs_start == 0 && abs_end == decoded.len() {
                    decoded.clone()
                } else {
                    decoded
                        .slice(abs_start..abs_end)
                        .map_err(|e| EngineError::message(format!("slice: {e}")))?
                };
                // No per-batch memory reservation: the reservation
                // handle was unused, and computing `nbytes()` per
                // batch (recursive walk over the array tree)
                // dominated profiles. The channel's running
                // byte-counter (Channel::retained_bytes_total) is the
                // actual gating mechanism; per-batch arbiter
                // accounting is dead code today.
                ctx.trace(format!(
                    "{}: emitted real rows [{}, {}) (abs [{}, {}))",
                    self.label, start, end, abs_start, abs_end
                ));
                (
                    start,
                    end,
                    Batch::from_array(DomainSpan::new(start, end - start), array),
                )
            }
            EmitDecision::Placeholder { start, end } => {
                ctx.trace(format!(
                    "{}: emitted placeholder rows [{}, {}) (NotNeeded — no I/O)",
                    self.label, start, end
                ));
                let span = DomainSpan::new(start, end - start);
                (start, end, Batch::placeholder(span, self.layout.dtype().clone()))
            }
        };

        ctx.push(batch)?;
        state.cursor = end;

        if state.cursor >= local_len {
            state.sealed = true;
            ctx.seal()?;
            return Ok(WorkStatus::Finished);
        }
        Ok(WorkStatus::Made)
    }
}

/// Outcome of the requirement walk: either emit real values for
/// `[start, end)` (decode the segment, slice, demand=true) or emit
/// a placeholder (constant array, demand=false, no I/O).
enum EmitDecision {
    Real { start: u64, end: u64 },
    Placeholder { start: u64, end: u64 },
}

impl FlatLayoutOperator {
    /// Fetch the layout's single segment and decode it to an
    /// `ArrayRef`. Mirrors `vortex_layout::layouts::flat::FlatReader`'s
    /// internal path but bypasses the `LayoutReader` trait so the
    /// engine has direct ownership of the I/O.
    ///
    /// Synchronous: the engine's V1 driver runs each scheduler turn
    /// on the calling thread, so we `block_on` the segment fetch.
    /// When the driver carries an explicit `DriverIo` (see
    /// `docs/design/drivers.md`), this becomes async with the I/O
    /// flowing through that capability.
    fn decode_segment(&self) -> EngineResult<ArrayRef> {
        use vortex_io::runtime::BlockingRuntime;

        let row_count = usize::try_from(self.layout.row_count())
            .map_err(|e| EngineError::message(format!("row_count overflow: {e}")))?;
        let segment_id = self.layout.segment_id();
        let array_tree = self.layout.array_tree().cloned();
        let ctx = self.layout.array_ctx().clone();
        let dtype = self.layout.dtype().clone();
        let session = self.session.clone();
        let segment_source = Arc::clone(&self.segment_source);

        self.runtime
            .block_on(async move {
                let segment = segment_source
                    .request(segment_id)
                    .await
                    .map_err(|e| EngineError::message(format!("segment fetch: {e}")))?;
                let parts = if let Some(tree) = array_tree {
                    SerializedArray::from_flatbuffer_and_segment(tree, segment).map_err(|e| {
                        EngineError::message(format!("from_flatbuffer_and_segment: {e}"))
                    })?
                } else {
                    SerializedArray::try_from(segment)
                        .map_err(|e| EngineError::message(format!("SerializedArray: {e}")))?
                };
                parts
                    .decode(&dtype, row_count, &ctx, &session)
                    .map_err(|e| EngineError::message(format!("decode: {e}")))
            })
    }
}
