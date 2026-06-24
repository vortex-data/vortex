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
use vortex_array::expr::Expression;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scan::read::ScanRead;
use vortex_session::VortexSession;

use crate::layout_v2::Flat;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutRef;
use crate::scan::plan::OwnedRowScope;
use crate::scan::plan::PrepareCtx;
use crate::scan::plan::PreparedRead;
use crate::scan::plan::PreparedReadRef;
use crate::scan::plan::PreparedStateKey;
use crate::scan::plan::PushCtx;
use crate::scan::plan::ReadContext;
use crate::scan::plan::ReadStep;
use crate::scan::plan::ReadTask;
use crate::scan::plan::ReadTaskOutput;
use crate::scan::plan::RowScope;
use crate::scan::plan::ScanPlan;
use crate::scan::plan::ScanPlanRef;
use crate::scan::plan::ScanState;
use crate::scan::plan::ScanStateRef;
use crate::scan::plan::StateCtx;
use crate::scan::plan::default_try_push_expr;
use crate::scan::plan::downcast_state;
use crate::scan::plan::request::ScanRequest;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;

pub(crate) fn new_scan_plan(
    layout: Layout<Flat>,
    _req: &mut ScanRequest,
    _session: &VortexSession,
) -> VortexResult<ScanPlanRef> {
    Ok(Arc::new(FlatScanPlan {
        layout: layout.to_layout(),
    }))
}

/// Reads a flat layout: fetches its segment once per query, parses it
/// into a (lazy) array, and slices per request.
pub struct FlatScanPlan {
    layout: LayoutRef,
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
    reads: Vec<ScanRead>,
    prefetch_reads: Vec<ScanRead>,
}

impl FlatScanPlan {
    fn array(&self, io: &ReadContext, state: &FlatScanState) -> VortexResult<ArrayRef> {
        if let Some(hit) = state.array.lock().clone() {
            return Ok(hit);
        }

        let mut guard = state.array.lock();
        if let Some(hit) = guard.clone() {
            return Ok(hit);
        }

        let array = decode_flat(&self.layout, io)?;
        *guard = Some(array.clone());
        Ok(array)
    }
}

impl ScanPlan for FlatScanPlan {
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
    fn segment_requests(
        &self,
        _range: Range<u64>,
        _rows: RowScope<'_>,
        cx: &mut SegmentPlanCtx,
    ) -> VortexResult<SegmentRequests> {
        let Some(flat) = self.node.layout.as_opt::<Flat>() else {
            vortex_bail!(
                "expected flat layout, got {}",
                self.node.layout.encoding_id()
            );
        };
        Ok(SegmentRequests::exact(vec![
            cx.request_for_segment(flat.data().segment_id())?,
        ]))
    }

    fn create_task(
        self: Arc<Self>,
        range: Range<u64>,
        rows: OwnedRowScope,
        reads: Vec<ScanRead>,
        prefetch_reads: Vec<ScanRead>,
        _cx: &mut SegmentPlanCtx,
    ) -> VortexResult<Box<dyn ReadTask>> {
        Ok(Box::new(FlatReadTask {
            read: self,
            range,
            rows,
            reads,
            prefetch_reads,
        }))
    }

    fn release(&self, frontier: u64) -> VortexResult<()> {
        self.node.release(frontier, &self.state)
    }

    fn fmt_prepared(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

impl ReadTask for FlatReadTask {
    fn into_step(self: Box<Self>) -> VortexResult<ReadStep> {
        let Self {
            read,
            range,
            rows,
            reads,
            prefetch_reads,
        } = *self;
        Ok(ReadStep::new(reads, prefetch_reads, move |io, _, _| {
            let array = read.node.array(io, &read.state)?;
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
        }))
    }
}

pub(crate) fn decode_flat(layout: &LayoutRef, io: &ReadContext) -> VortexResult<ArrayRef> {
    let Some(flat) = layout.as_opt::<Flat>() else {
        vortex_bail!("expected flat layout, got {}", layout.encoding_id());
    };
    let row_count = usize::try_from(layout.row_count())
        .map_err(|_| vortex_err!("layout row count exceeds usize"))?;
    let segment = io.segment(flat.data().segment_id())?;
    let parts = if let Some(tree) = flat.data().array_tree() {
        SerializedArray::from_flatbuffer_and_segment(tree.clone(), segment)?
    } else {
        SerializedArray::try_from(segment)?
    };
    parts.decode(
        layout.dtype(),
        row_count,
        flat.data().array_ctx(),
        io.session(),
    )
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
