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

use futures::future::BoxFuture;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::serde::SerializedArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::layout_v2::Flat;
use crate::layout_v2::Layout;
use crate::layout_v2::LayoutRef;
use crate::scan::v2::node::ExpandCtx;
use crate::scan::v2::node::FileReader;
use crate::scan::v2::node::PlanCtx;
use crate::scan::v2::node::ReadPlan;
use crate::scan::v2::node::ReadPlanRef;
use crate::scan::v2::node::RowScope;
use crate::scan::v2::node::ScanNode;
use crate::scan::v2::node::ScanNodeRef;
use crate::scan::v2::node::ScanStateRef;
use crate::scan::v2::node::StateCtx;
use crate::scan::v2::node::downcast_state;
use crate::scan::v2::request::NodeRequest;
use crate::segments::SegmentPlanCtx;
use crate::segments::SegmentRequests;

pub(crate) fn new_scan_node(
    layout: Layout<Flat>,
    _req: &mut NodeRequest,
    _cx: &ExpandCtx,
) -> VortexResult<ScanNodeRef> {
    Ok(Arc::new(FlatScanNode {
        layout: layout.to_layout(),
    }))
}

/// Reads a flat layout: fetches its segment once per query, parses it
/// into a (lazy) array, and slices per request.
pub struct FlatScanNode {
    layout: LayoutRef,
}

/// Per-query cache of the parsed (still lazy) array. Concurrent decodes
/// are benign: the segment fetch is deduplicated by the shared segment
/// source, and last-write-wins on the parsed array.
#[derive(Default)]
pub struct FlatScanState {
    array: Mutex<Option<ArrayRef>>,
}

struct FlatReadPlan {
    node: Arc<FlatScanNode>,
}

impl ScanNode for FlatScanNode {
    type State = FlatScanState;

    fn init_state(&self, _cx: &mut StateCtx<'_>) -> VortexResult<FlatScanState> {
        Ok(FlatScanState::default())
    }

    fn plan_read(self: Arc<Self>, _cx: &mut PlanCtx) -> VortexResult<Option<ReadPlanRef>> {
        Ok(Some(Arc::new(FlatReadPlan { node: self })))
    }

    /// A flat leaf releases only once *wholly* behind the frontier: a
    /// partially-covered flat is the working set, and dropping it would
    /// thrash the segment fetch.
    fn release(&self, frontier: u64, state: &FlatScanState) -> VortexResult<()> {
        if frontier >= self.layout.row_count() {
            *state.array.lock() = None;
        }
        Ok(())
    }

    fn fmt_chain(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "flat")
    }
}

impl ReadPlan for FlatReadPlan {
    type State = ScanStateRef;

    fn init_state(&self, cx: &mut StateCtx<'_>) -> VortexResult<Self::State> {
        let node: ScanNodeRef = Arc::<FlatScanNode>::clone(&self.node);
        cx.init_node(&node)
    }

    fn read_scoped<'a>(
        &'a self,
        range: Range<u64>,
        rows: RowScope<'a>,
        io: &'a FileReader,
        state: &'a Self::State,
        _local: &'a mut ExecutionCtx,
    ) -> BoxFuture<'a, VortexResult<ArrayRef>> {
        let state = match downcast_state::<FlatScanNode>(state.as_ref()) {
            Ok(state) => state,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        Box::pin(async move {
            let array = if let Some(hit) = state.array.lock().clone() {
                hit
            } else {
                let decoded = decode_flat(&self.node.layout, io).await?;
                *state.array.lock() = Some(decoded.clone());
                decoded
            };
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
                return Ok(dense);
            }
            dense.filter(rows.selection.clone())
        })
    }

    fn segment_requests(
        &self,
        _range: Range<u64>,
        _rows: RowScope<'_>,
        _state: &Self::State,
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

    fn release(&self, frontier: u64, state: &Self::State) -> VortexResult<()> {
        self.node
            .release(frontier, downcast_state::<FlatScanNode>(state.as_ref())?)
    }

    fn fmt_plan(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt_chain(f)
    }
}

pub(crate) async fn decode_flat(layout: &LayoutRef, io: &FileReader) -> VortexResult<ArrayRef> {
    let Some(flat) = layout.as_opt::<Flat>() else {
        vortex_bail!("expected flat layout, got {}", layout.encoding_id());
    };
    let row_count = usize::try_from(layout.row_count())
        .map_err(|_| vortex_err!("layout row count exceeds usize"))?;
    let segment = io.segments().request(flat.data().segment_id()).await?;
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
    array.slice(start..end)
}
