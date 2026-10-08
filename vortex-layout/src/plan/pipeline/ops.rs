// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! How each plan operator of this crate compiles into a [`PipelineGraph`](super::PipelineGraph).
//!
//! What an operator's exec node decides on its first compute is decided here, while the graph is
//! built: a plan with a cached result compiles to a known piece or a transform, and only one
//! that still has to collect its inputs compiles to a sink.

use std::mem;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Shared;
use vortex_array::arrays::SharedArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::ConcatPlan;
use crate::plan::EvalPlan;
use crate::plan::FilterPlan;
use crate::plan::PackPlan;
use crate::plan::SegmentScan;
use crate::plan::SegmentScanPlan;
use crate::plan::SharePlan;
use crate::plan::TakePlan;
use crate::plan::ZonedPlan;
use crate::plan::exec::ExecContext;
use crate::plan::exec::Piece;
use crate::plan::exec::Port;
use crate::plan::exec::Selection;
use crate::plan::exec::assemble;
use crate::plan::exec::decode_segment;
use crate::plan::exec::empty_piece;
use crate::plan::exec::expand_zones;
use crate::plan::exec::join;
use crate::plan::exec::keep_selected;
use crate::plan::exec::prune_zones;
use crate::plan::exec::pruned_zones;
use crate::plan::exec::row_indices;
use crate::plan::exec::slice_rows;
use crate::plan::pipeline::GraphBuilder;
use crate::plan::pipeline::Sink;
use crate::plan::pipeline::Source;
use crate::plan::pipeline::Transform;
use crate::segments::SegmentId;

/// Joins the arrays of `pieces` in row order.
fn in_row_order(dtype: &DType, mut pieces: Vec<Piece>) -> VortexResult<ArrayRef> {
    pieces.sort_by_key(|piece| piece.rows.start);
    join(dtype, pieces.into_iter().map(|piece| piece.array).collect())
}

/// A segment scan is a source producing every row of its range, or a known piece when the
/// segment is already decoded.
pub(crate) fn segment_scan(
    plan: &SegmentScanPlan,
    rows: Range<u64>,
    cx: &mut GraphBuilder<'_>,
) -> VortexResult<()> {
    if let Some(array) = cx.context().decoded().get(plan.segment_id()) {
        let array = slice_rows(plan, array, &rows)?;
        cx.piece(Piece { rows, array });
        return Ok(());
    }
    let source = SegmentSource {
        plan: plan.clone(),
        rows: rows.clone(),
    };
    cx.source(rows, Box::new(source));
    Ok(())
}

struct SegmentSource {
    plan: SegmentScanPlan,
    rows: Range<u64>,
}

impl Source for SegmentSource {
    fn segment(&self) -> Option<SegmentId> {
        Some(self.plan.segment_id())
    }

    fn produce(
        &mut self,
        bytes: Option<BufferHandle>,
        ctx: &ExecContext,
    ) -> VortexResult<ArrayRef> {
        let bytes = bytes.ok_or_else(|| vortex_err!("SegmentScan ran without its segment"))?;
        let array = decode_segment(&self.plan, ctx.session(), bytes)?;
        ctx.decoded().insert(self.plan.segment_id(), array.clone());
        slice_rows(&self.plan, array, &self.rows)
    }
}

/// A filter is a transform keeping the selected rows of each piece, and nothing at all when
/// every row is selected.
pub(crate) fn filter(
    plan: &FilterPlan,
    rows: Range<u64>,
    mask: Mask,
    cx: &mut GraphBuilder<'_>,
) -> VortexResult<()> {
    if mask.all_false() {
        cx.piece(empty_piece(plan.dtype(), rows));
        return Ok(());
    }
    let child = plan.child_plan()?;
    if mask.all_true() {
        return cx.child(&child, rows, mask);
    }
    let keep = KeepSelected {
        selection: Selection::try_new(rows.clone(), mask.clone())?,
        // The rows of a segment are not a predicate's result, even when they are boolean.
        predicate: plan.dtype().is_boolean() && !child.is::<SegmentScan>(),
        session: cx.context().session().clone(),
    };
    cx.transform(Arc::new(keep), |cx| cx.child(&child, rows, mask))
}

struct KeepSelected {
    selection: Selection,
    predicate: bool,
    session: VortexSession,
}

impl Transform for KeepSelected {
    fn apply(&self, piece: Piece) -> VortexResult<Piece> {
        let mask = self.selection.slice(&piece.rows);
        Ok(Piece {
            rows: piece.rows,
            array: keep_selected(piece.array, mask, self.predicate, &self.session)?,
        })
    }
}

/// An eval is a transform applying its expression to each piece.
pub(crate) fn eval(
    plan: &EvalPlan,
    rows: Range<u64>,
    mask: Mask,
    cx: &mut GraphBuilder<'_>,
) -> VortexResult<()> {
    let child = plan.child_plan()?;
    let apply = Apply {
        plan: plan.clone(),
        session: cx.context().session().clone(),
    };
    cx.transform(Arc::new(apply), |cx| cx.child(&child, rows, mask))
}

struct Apply {
    plan: EvalPlan,
    session: VortexSession,
}

impl Transform for Apply {
    fn apply(&self, piece: Piece) -> VortexResult<Piece> {
        Ok(Piece {
            rows: piece.rows,
            array: self.plan.apply(piece.array, &self.session)?,
        })
    }
}

/// A concat adds nothing of its own: each chunk overlapping the rows compiles with its pieces
/// rebased into the concatenated row domain. A chunk with nothing selected is never compiled; its
/// rows are a known empty piece.
pub(crate) fn concat(
    plan: &ConcatPlan,
    rows: Range<u64>,
    mask: Mask,
    cx: &mut GraphBuilder<'_>,
) -> VortexResult<()> {
    let selection = Selection::try_new(rows.clone(), mask)?;
    let offsets = plan.row_offsets();
    for (index, start) in offsets.iter().copied().enumerate() {
        let end = offsets
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| plan.row_count());
        let local = rows.start.max(start)..rows.end.min(end);
        if local.start >= local.end {
            continue;
        }
        let mask = selection.slice(&local);
        if mask.all_false() {
            cx.piece(empty_piece(plan.dtype(), local));
            continue;
        }
        let child = plan.child_required(index)?;
        cx.rebase(start, |cx| {
            cx.child(&child, local.start - start..local.end - start, mask)
        })?;
    }
    Ok(())
}

/// A pack is a sink with a port for each field.
pub(crate) fn pack(
    plan: &PackPlan,
    rows: Range<u64>,
    mask: Mask,
    cx: &mut GraphBuilder<'_>,
) -> VortexResult<()> {
    let nports = plan.children().len();
    let selection = Selection::try_new(rows.clone(), mask.clone())?;
    if nports == 0 {
        cx.piece(assemble(plan, &selection, Vec::new())?);
        return Ok(());
    }
    if mask.all_false() {
        cx.piece(empty_piece(plan.dtype(), rows));
        return Ok(());
    }
    let sink = PackSink {
        plan: plan.clone(),
        selection,
        ports: vec![Vec::new(); nports],
    };
    cx.sink(rows.clone(), Box::new(sink), |cx| {
        for port in 0..nports {
            let child = plan.child_required(port)?;
            cx.port(port, |cx| cx.child(&child, rows.clone(), mask.clone()))?;
        }
        Ok(())
    })
}

struct PackSink {
    plan: PackPlan,
    selection: Selection,
    ports: Vec<Vec<Piece>>,
}

impl Sink for PackSink {
    fn push(&mut self, port: Port, piece: Piece) -> VortexResult<()> {
        self.ports[port].push(piece);
        Ok(())
    }

    fn finish(&mut self) -> VortexResult<ArrayRef> {
        let ports = mem::take(&mut self.ports);
        Ok(assemble(&self.plan, &self.selection, ports)?.array)
    }
}

const CODES: Port = 0;
const VALUES: Port = 1;

/// A take whose values the plan already holds is a transform over its codes. Otherwise it is a
/// sink over the codes of the selected rows and the whole of the values.
pub(crate) fn take(
    plan: &TakePlan,
    rows: Range<u64>,
    mask: Mask,
    cx: &mut GraphBuilder<'_>,
) -> VortexResult<()> {
    if mask.all_false() {
        cx.piece(empty_piece(plan.dtype(), rows));
        return Ok(());
    }
    let codes = plan.codes()?;
    if let Some(values) = plan.cached_values() {
        return cx.transform(Arc::new(Lookup { values }), |cx| {
            cx.child(&codes, rows, mask)
        });
    }
    let values = plan.values()?;
    let len = usize::try_from(values.row_count())?;
    let sink = TakeSink {
        plan: plan.clone(),
        codes_dtype: codes.dtype().clone(),
        values_dtype: values.dtype().clone(),
        codes: Vec::new(),
        values: Vec::new(),
    };
    cx.sink(rows.clone(), Box::new(sink), |cx| {
        cx.port(CODES, |cx| cx.child(&codes, rows, mask))?;
        cx.port(VALUES, |cx| {
            cx.child(&values, 0..len as u64, Mask::new_true(len))
        })
    })
}

/// Looks up each code of a piece in dictionary values that are already known.
struct Lookup {
    values: ArrayRef,
}

impl Transform for Lookup {
    fn apply(&self, piece: Piece) -> VortexResult<Piece> {
        Ok(Piece {
            rows: piece.rows,
            array: DictArray::try_new(piece.array, self.values.clone())?.into_array(),
        })
    }
}

struct TakeSink {
    plan: TakePlan,
    codes_dtype: DType,
    values_dtype: DType,
    codes: Vec<Piece>,
    values: Vec<Piece>,
}

impl Sink for TakeSink {
    fn push(&mut self, port: Port, piece: Piece) -> VortexResult<()> {
        match port {
            CODES => self.codes.push(piece),
            VALUES => self.values.push(piece),
            port => vortex_bail!("Take has no port {port}"),
        }
        Ok(())
    }

    fn finish(&mut self) -> VortexResult<ArrayRef> {
        let values = in_row_order(&self.values_dtype, mem::take(&mut self.values))?;
        // The values are kept on the plan as a shared array, so however many executions use them
        // they are canonicalized once.
        let values = if values.is::<Shared>() {
            values
        } else {
            SharedArray::new(values).into_array()
        };
        let values = self.plan.cache_values(values);
        let codes = in_row_order(&self.codes_dtype, mem::take(&mut self.codes))?;
        Ok(DictArray::try_new(codes, values)?.into_array())
    }
}

/// The selected rows of a shared value.
fn shared_rows(value: &ArrayRef, selection: &Selection) -> VortexResult<ArrayRef> {
    let rows = selection.rows();
    let array = value.slice(usize::try_from(rows.start)?..usize::try_from(rows.end)?)?;
    if selection.mask().all_true() {
        return Ok(array);
    }
    array.filter(selection.mask().clone())
}

/// A share whose value the plan already holds is a known piece. Otherwise it is a sink over the
/// whole of its child.
pub(crate) fn share(
    plan: &SharePlan,
    rows: Range<u64>,
    mask: Mask,
    cx: &mut GraphBuilder<'_>,
) -> VortexResult<()> {
    let selection = Selection::try_new(rows.clone(), mask)?;
    if let Some(value) = plan.cached() {
        let array = shared_rows(&value, &selection)?;
        cx.piece(Piece { rows, array });
        return Ok(());
    }
    let child = plan.child_plan()?;
    let len = usize::try_from(plan.row_count())?;
    let sink = ShareSink {
        plan: plan.clone(),
        selection,
        pieces: Vec::new(),
    };
    cx.sink(rows, Box::new(sink), |cx| {
        cx.port(0, |cx| cx.child(&child, 0..len as u64, Mask::new_true(len)))
    })
}

struct ShareSink {
    plan: SharePlan,
    selection: Selection,
    pieces: Vec<Piece>,
}

impl Sink for ShareSink {
    fn push(&mut self, _port: Port, piece: Piece) -> VortexResult<()> {
        self.pieces.push(piece);
        Ok(())
    }

    fn finish(&mut self) -> VortexResult<ArrayRef> {
        let value = in_row_order(self.plan.dtype(), mem::take(&mut self.pieces))?;
        let value = self.plan.cache(SharedArray::new(value).into_array());
        shared_rows(&value, &self.selection)
    }
}

/// A data plan compiles as its data child. A pruning plan whose zones the plan already holds is
/// a known piece; otherwise it is a sink over the whole zone table.
pub(crate) fn zoned(
    plan: &ZonedPlan,
    rows: Range<u64>,
    mask: Mask,
    cx: &mut GraphBuilder<'_>,
) -> VortexResult<()> {
    if !plan.is_pruning() {
        let Some(data) = plan.data_plan()? else {
            vortex_bail!("Zoned plan has no data child");
        };
        return cx.child(&data, rows, mask);
    }
    if mask.all_false() {
        cx.piece(empty_piece(plan.dtype(), rows));
        return Ok(());
    }
    let selection = Selection::try_new(rows.clone(), mask)?;
    let session = cx.context().session().clone();
    let expression = plan
        .pruning_expression()
        .ok_or_else(|| vortex_err!("Zoned pruning plan has no expression"))?;
    if let Some(pruned) = pruned_zones(plan)?.get(expression, &session)? {
        cx.piece(expand_zones(plan, &selection, &pruned)?);
        return Ok(());
    }
    let zones = plan.zones_plan()?;
    let count = usize::try_from(zones.row_count())?;
    let sink = ZoneSink {
        plan: plan.clone(),
        selection,
        session,
        zones: Vec::new(),
    };
    cx.sink(rows, Box::new(sink), |cx| {
        cx.port(0, |cx| {
            cx.child(&zones, 0..count as u64, Mask::new_true(count))
        })
    })
}

struct ZoneSink {
    plan: ZonedPlan,
    selection: Selection,
    session: VortexSession,
    zones: Vec<Piece>,
}

impl Sink for ZoneSink {
    fn push(&mut self, _port: Port, piece: Piece) -> VortexResult<()> {
        self.zones.push(piece);
        Ok(())
    }

    fn finish(&mut self) -> VortexResult<ArrayRef> {
        let pruned = prune_zones(&self.plan, &self.session, mem::take(&mut self.zones))?;
        Ok(expand_zones(&self.plan, &self.selection, &pruned)?.array)
    }
}

/// A row index is a known piece: the global index of every selected row.
pub(crate) fn row_idx(rows: Range<u64>, mask: Mask, cx: &mut GraphBuilder<'_>) -> VortexResult<()> {
    let selection = Selection::try_new(rows, mask)?;
    let piece = row_indices(&selection, cx.context().row_offset())?;
    cx.piece(piece);
    Ok(())
}
