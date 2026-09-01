// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::serde::SerializedArray;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::segments::SegmentId;
use vortex_session::registry::ReadContext;

use crate::io::IoBatch;
use crate::io::IoKey;
use crate::io::IoTicket;
use crate::io::IoUse;
use crate::io::ProducerId;
use crate::node::ExecCx;
use crate::node::ExecNode;
use crate::node::ExecPoll;
use crate::node::NodeId;
use crate::node::PlanCx;
use crate::node::PlanItem;
use crate::node::PlanPoll;
use crate::node::RetireCx;
use crate::node::Value;
use crate::node::ValueBatch;
use crate::node::Wait;
use crate::node::WaitSet;

/// The only node that touches the world: one stored segment, decoded and sliced.
///
/// `next_plan` names the segment exactly once per morsel. If the shared cell for the segment
/// already holds a decoded value, planning skips issuing the read — the morsel's own lease keeps
/// that value alive until it retires. Otherwise `execute` clones the scheduler-resolved ticket,
/// decodes, publishes into the cell, then slices to the morsel's local range and applies demand.
/// Retire releases the lease whether the value was used or not; the last release drops the cell.
pub struct FlatExec {
    segment: SegmentId,
    dtype: DType,
    read_ctx: ReadContext,
    array_tree: Option<ByteBuffer>,
    /// Rows in the whole segment.
    segment_rows: u64,
    /// Root-coordinate offset of this segment's row zero, for stamping `source_range`.
    root_offset: u64,
    estimated_bytes: usize,
    producer: ProducerId,

    // Per-morsel state.
    range: Range<u64>,
    ticket: Option<IoTicket>,
    planned: bool,
    done: bool,
}

impl FlatExec {
    /// Build a flat node over a flat layout.
    pub fn new(layout: &FlatLayout, root_offset: u64, producer: ProducerId) -> Self {
        let segment_rows = layout.row_count();
        let estimated_bytes = estimate_bytes(layout.dtype(), segment_rows);
        Self {
            segment: layout.segment_id(),
            dtype: layout.dtype().clone(),
            read_ctx: layout.array_ctx().clone(),
            array_tree: layout.array_tree().cloned(),
            segment_rows,
            root_offset,
            estimated_bytes,
            producer,
            range: 0..0,
            ticket: None,
            planned: false,
            done: false,
        }
    }

    fn decode(&self, cx: &mut ExecCx<'_>) -> VortexResult<Option<ArrayRef>> {
        if let Some(shared) = cx.shared_decoded(IoKey::Segment(self.segment)) {
            return Ok(Some(shared));
        }

        let ticket = self
            .ticket
            .ok_or_else(|| crate::io::unplanned_ticket(self.producer))?;
        let Some(bytes) = cx.ready(ticket)? else {
            return Ok(None);
        };

        let parts = match self.array_tree.as_ref() {
            Some(tree) => SerializedArray::from_flatbuffer_and_segment(tree.clone(), bytes)?,
            None => SerializedArray::try_from(bytes)?,
        };
        let rows = usize::try_from(self.segment_rows)
            .map_err(|_| vortex_err!("segment row count exceeds usize"))?;
        let session = cx.session().clone();
        let array = parts.decode(&self.dtype, rows, &self.read_ctx, &session)?;
        cx.stats().decodes += 1;
        cx.publish_decoded(IoKey::Segment(self.segment), &array);
        Ok(Some(array))
    }
}

impl ExecNode for FlatExec {
    fn reset(&mut self, range: Range<u64>) {
        debug_assert!(
            range.end <= self.segment_rows,
            "flat range {range:?} exceeds segment rows {}",
            self.segment_rows
        );
        self.range = range;
        self.ticket = None;
        self.planned = false;
        self.done = false;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        if self.planned || self.range.is_empty() {
            return Ok(PlanPoll::Complete);
        }
        if cx.out_of_budget() {
            return Ok(PlanPoll::Item(PlanItem::Plan));
        }

        // A decoded value already published by another morsel makes the read unnecessary. The
        // lease this morsel holds (counted before the scan started) pins the value until retire,
        // so skipping the read here can never leave execute empty-handed.
        if cx.decoded_available(IoKey::Segment(self.segment)) {
            self.planned = true;
            return Ok(PlanPoll::Complete);
        }

        // The extent is the whole stored unit: two morsels straddling this segment name the same
        // cell and share one read.
        let mut batch = IoBatch::new();
        batch.push(IoUse {
            key: IoKey::Segment(self.segment),
            extent: 0..self.segment_rows,
            source_range: self.root_offset..self.root_offset + self.segment_rows,
            producer: self.producer,
            estimated_bytes: self.estimated_bytes,
        });
        let tickets = cx.register(batch.clone())?;
        self.ticket = tickets.first().copied();
        self.planned = true;
        Ok(PlanPoll::Item(PlanItem::Io(batch)))
    }

    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        if self.done {
            return Ok(ExecPoll::Done);
        }
        let Some(mut array) = self.decode(cx)? else {
            let ticket = self
                .ticket
                .ok_or_else(|| crate::io::unplanned_ticket(self.producer))?;
            return Ok(ExecPoll::Blocked(
                [Wait::Io(ticket)].into_iter().collect::<WaitSet>(),
            ));
        };

        let start = usize::try_from(self.range.start).vortex_expect("flat range start fits usize");
        let end = usize::try_from(self.range.end).vortex_expect("flat range end fits usize");
        if start > 0 || end < array.len() {
            array = array.slice(start..end)?;
        }

        let demand = cx.demand();
        if !demand.all_true() {
            array = array.filter(demand.clone())?;
        }
        self.done = true;

        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.root_offset + self.range.start..self.root_offset + self.range.end,
            value: Value::Array(array),
        }))
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        if self.planned {
            cx.release_use(IoKey::Segment(self.segment));
        }
        self.ticket = None;
        self.planned = false;
    }

    fn children(&self) -> &[NodeId] {
        &[]
    }
}

/// A rough per-row byte estimate, used only for admission accounting.
///
/// The layout does not carry segment byte sizes, so this is a width estimate rather than a
/// measurement; P2's cost model replaces it with the footer's real segment extents.
fn estimate_bytes(dtype: &DType, rows: u64) -> usize {
    let per_row = match dtype {
        DType::Bool(_) => 1,
        DType::Primitive(ptype, _) => ptype.byte_width(),
        DType::Decimal(..) => 16,
        DType::Utf8(_) | DType::Binary(_) => 16,
        _ => 8,
    };
    usize::try_from(rows)
        .unwrap_or(usize::MAX)
        .saturating_mul(per_row)
}
