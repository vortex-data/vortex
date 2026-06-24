// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scan2 vtable support for flat layouts: one segment, parsed lazily, decoded on
//! demand.
//!
//! A flat leaf exposes no evidence producers — it has no statistics or
//! index — and keeps the default selection path: its segment decodes whole, so a
//! selected read is the dense parse followed by a lazy filter, which
//! vortex pushes through the encodings.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scan::plan::OwnedRowScope;
use vortex_scan::plan::PrepareCtx;
use vortex_scan::plan::PreparedRead;
use vortex_scan::plan::PreparedReadRef;
use vortex_scan::plan::PreparedStateKey;
use vortex_scan::plan::PushCtx;
use vortex_scan::plan::ReadStep;
use vortex_scan::plan::ReadTask;
use vortex_scan::plan::ReadTaskOutput;
use vortex_scan::plan::ScanPlan;
use vortex_scan::plan::ScanPlanRef;
use vortex_scan::plan::ScanState;
use vortex_scan::plan::ScanStateRef;
use vortex_scan::plan::StateCtx;
use vortex_scan::plan::default_try_push_expr;
use vortex_scan::plan::downcast_state;
use vortex_scan::plan::request::ScanRequest;
use vortex_scan::read::ReadRequestKey;
use vortex_scan::read::ReadResults;
use vortex_scan::read::ScanIoPhase;
use vortex_scan::read::ScanRead;
use vortex_session::VortexSession;

use crate::layout_v2::Layout;
use crate::layout_v2::LayoutRef;
use crate::layout_v2::LayoutScanPlanCtx;
use crate::layouts_v2::flat::Flat;
use crate::segments::SegmentFutureCache;
use crate::segments::SegmentRequest;
use crate::segments::SegmentRequestKey;
use crate::segments::SegmentSource;

pub(crate) fn new_scan_plan(
    layout: Layout<Flat>,
    _req: &mut ScanRequest,
    ctx: &LayoutScanPlanCtx,
) -> VortexResult<ScanPlanRef> {
    Ok(Arc::new(FlatScanPlan {
        layout: layout.to_layout(),
        session: ctx.session().clone(),
        segment_source: Arc::clone(ctx.segment_source()),
        segment_future_cache: Arc::clone(ctx.segment_future_cache()),
    }))
}

/// Reads a flat layout: fetches its segment once per query, parses it
/// into a (lazy) array, and slices per request.
pub struct FlatScanPlan {
    layout: LayoutRef,
    session: VortexSession,
    segment_source: Arc<dyn SegmentSource>,
    segment_future_cache: Arc<SegmentFutureCache>,
}

/// Per-query cache of the parsed (still lazy) array.
#[derive(Default)]
pub struct FlatScanState {
    array: Mutex<Option<ArrayRef>>,
}

struct FlatPreparedRead {
    node: Arc<FlatScanPlan>,
    state: Arc<FlatScanState>,
}

struct FlatReadTask {
    read: Arc<FlatPreparedRead>,
    range: Range<u64>,
    rows: OwnedRowScope,
    phase: ScanIoPhase,
}

impl FlatScanPlan {
    fn array(&self, results: &ReadResults, state: &FlatScanState) -> VortexResult<ArrayRef> {
        if let Some(hit) = state.array.lock().clone() {
            return Ok(hit);
        }

        let mut guard = state.array.lock();
        if let Some(hit) = guard.clone() {
            return Ok(hit);
        }

        let array = decode_flat(&self.layout, results, &self.session)?;
        *guard = Some(array.clone());
        Ok(array)
    }
}

impl ScanPlan for FlatScanPlan {
    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<ScanStateRef> {
        Ok(Arc::new(FlatScanState::default()))
    }

    fn try_push_expr(
        self: Arc<Self>,
        expr: &Expression,
        _cx: &mut PushCtx,
    ) -> VortexResult<Option<ScanPlanRef>> {
        default_try_push_expr(self, expr)
    }

    fn prepare_read(self: Arc<Self>, cx: &mut PrepareCtx) -> VortexResult<Option<PreparedReadRef>> {
        let flat = self.layout.as_opt::<Flat>().ok_or_else(|| {
            vortex_err!("expected flat layout, got {}", self.layout.encoding_id())
        })?;
        let key = PreparedStateKey::new::<FlatScanState>(*flat.data().segment_id() as usize);
        let state = cx.shared_state(key, || Ok(FlatScanState::default()))?;
        Ok(Some(Arc::new(FlatPreparedRead { node: self, state })))
    }

    /// A flat leaf releases only once *wholly* behind the frontier: a
    /// partially-covered flat is the working set, and dropping it would
    /// thrash the segment fetch.
    fn release(&self, frontier: u64, state: &ScanState) -> VortexResult<()> {
        let state = downcast_state::<FlatScanState>(state)?;
        if frontier >= self.layout.row_count() {
            *state.array.lock() = None;
        }
        Ok(())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "flat")
    }
}

impl PreparedRead for FlatPreparedRead {
    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        phase: ScanIoPhase,
    ) -> VortexResult<Box<dyn ReadTask>> {
        Ok(Box::new(FlatReadTask {
            read: self,
            range,
            rows,
            phase,
        }))
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.node.release(frontier, &self.state)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

impl FlatPreparedRead {
    fn segment_read(&self, phase: ScanIoPhase) -> VortexResult<ScanRead> {
        let Some(flat) = self.node.layout.as_opt::<Flat>() else {
            vortex_bail!(
                "expected flat layout, got {}",
                self.node.layout.encoding_id()
            );
        };
        self.node
            .segment_future_cache
            .register(
                self.node.segment_source.as_ref(),
                vec![SegmentRequest::new(
                    flat.data().segment_id(),
                    self.node
                        .segment_source
                        .segment_info(flat.data().segment_id())?,
                    phase,
                )],
            )
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("flat segment read registration returned no reads"))
    }
}

impl ReadTask for FlatReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        let Self {
            read,
            range,
            rows,
            phase,
        } = *self;
        let segment_read = read.segment_read(phase)?;
        Ok(ReadStep::new(
            vec![segment_read],
            Vec::new(),
            move |_, _, results| {
                let array = read.node.array(&results, &read.state)?;
                let rows = rows.as_scope();
                let dense = slice_to_range(array, &range)?;
                if rows.selection.len() != dense.len() {
                    vortex_bail!(
                        "selection length {} does not match read range length {}",
                        rows.selection.len(),
                        dense.len()
                    );
                }
                if rows.demand.len() != dense.len() {
                    vortex_bail!(
                        "demand length {} does not match read range length {}",
                        rows.demand.len(),
                        dense.len()
                    );
                }
                if rows.selection.all_true() {
                    return Ok(ReadTaskOutput::Ready(dense));
                }
                Ok(ReadTaskOutput::Ready(dense.filter(rows.selection.clone())?))
            },
        ))
    }
}

pub(crate) fn decode_flat(
    layout: &LayoutRef,
    results: &ReadResults,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    let Some(flat) = layout.as_opt::<Flat>() else {
        vortex_bail!("expected flat layout, got {}", layout.encoding_id());
    };
    let row_count = usize::try_from(layout.row_count())
        .map_err(|_| vortex_err!("layout row count exceeds usize"))?;
    let key = ReadRequestKey::from(SegmentRequestKey::new(flat.data().segment_id()));
    let segment = results.get(key)?;
    let parts = if let Some(tree) = flat.data().array_tree() {
        SerializedArray::from_flatbuffer_and_segment(tree.clone(), segment)?
    } else {
        SerializedArray::try_from(segment)?
    };
    parts.decode(layout.dtype(), row_count, flat.data().array_ctx(), session)
}

pub(crate) fn slice_to_range(array: ArrayRef, range: &Range<u64>) -> VortexResult<ArrayRef> {
    let start = usize::try_from(range.start).map_err(|_| vortex_err!("row range exceeds usize"))?;
    let end = usize::try_from(range.end).map_err(|_| vortex_err!("row range exceeds usize"))?;
    if start == 0 && end == array.len() {
        return Ok(array);
    }
    SliceArray::try_new(array, start..end)?
        .into_array()
        .optimize()
}
