// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::ops::Range;

use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::PackPlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Input;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;
use crate::plan::exec::piece::empty_piece;
use crate::plan::exec::piece::join;

/// Assembles struct pieces from fields that arrive with different row alignments and in any order.
///
/// Each port buffers its pieces keyed by first row. A step emits one struct piece for every
/// maximal row range that all ports cover, slicing pieces at the boundaries and keeping the
/// remainders for later ranges.
pub(crate) struct PackNode {
    plan: PackPlan,
    selection: Selection,
    ports: Vec<BTreeMap<u64, Piece>>,
    open: usize,
    started: bool,
}

impl PackNode {
    pub(crate) fn new(plan: PackPlan, selection: Selection) -> Self {
        let nports = plan.children().len();
        Self {
            plan,
            selection,
            ports: vec![BTreeMap::new(); nports],
            open: nports,
            started: false,
        }
    }

    /// Row ranges covered by every port.
    fn aligned(&self) -> Vec<Range<u64>> {
        let mut ports = self.ports.iter();
        let Some(first) = ports.next() else {
            return Vec::new();
        };
        ports.fold(coverage(first), |acc, port| {
            intersect(&acc, &coverage(port))
        })
    }

    /// Removes the parts of each port's pieces covering `rows`, keeping any remainders.
    fn take(&mut self, rows: &Range<u64>) -> VortexResult<Vec<Vec<Piece>>> {
        let mut taken = Vec::with_capacity(self.ports.len());
        for port in &mut self.ports {
            let overlapping = port
                .range(..rows.end)
                .filter(|(_, piece)| piece.rows.end > rows.start)
                .map(|(start, _)| *start)
                .collect::<Vec<_>>();
            let mut parts = Vec::with_capacity(overlapping.len());
            for start in overlapping {
                let Some(piece) = port.remove(&start) else {
                    continue;
                };
                if piece.rows.start < rows.start {
                    let before = piece.slice_rows(&self.selection, piece.rows.start..rows.start)?;
                    port.insert(before.rows.start, before);
                }
                if piece.rows.end > rows.end {
                    let after = piece.slice_rows(&self.selection, rows.end..piece.rows.end)?;
                    port.insert(after.rows.start, after);
                }
                let inner = rows.start.max(piece.rows.start)..rows.end.min(piece.rows.end);
                parts.push(piece.slice_rows(&self.selection, inner)?);
            }
            taken.push(parts);
        }
        Ok(taken)
    }

    fn assemble(&self, rows: Range<u64>, ports: Vec<Vec<Piece>>) -> VortexResult<Piece> {
        let fields = self.plan.fields();
        let len = self.selection.count(&rows);
        let mut arrays = Vec::with_capacity(ports.len());
        for (index, parts) in ports.into_iter().enumerate() {
            let dtype = fields
                .field_by_index(index)
                .unwrap_or(DType::Bool(Nullability::NonNullable));
            arrays.push(join(&dtype, parts.into_iter().map(|p| p.array).collect())?);
        }
        let validity = if self.plan.dtype().is_nullable() {
            match arrays.pop() {
                Some(validity) => Validity::Array(validity),
                None => vortex_bail!("Nullable Pack is missing its validity port"),
            }
        } else {
            Validity::NonNullable
        };
        let array = StructArray::try_new_with_dtype(arrays, fields.clone(), len, validity)?;
        Ok(Piece {
            rows,
            array: array.into_array(),
        })
    }
}

impl ExecNode for PackNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if !self.started {
            self.started = true;
            let rows = self.selection.rows().clone();
            if self.selection.mask().all_false() || self.ports.is_empty() {
                let piece = if self.ports.is_empty() {
                    self.assemble(rows, Vec::new())?
                } else {
                    empty_piece(self.plan.dtype(), rows)
                };
                cx.emit(piece);
                cx.close();
                return Ok(NodeState::Done);
            }
            for port in 0..self.ports.len() {
                cx.spawn(
                    port,
                    self.plan.child_required(port)?,
                    self.selection.rows().clone(),
                    self.selection.mask().clone(),
                );
            }
        }

        for (port, input) in cx.take_inputs() {
            match input {
                Input::Piece(piece) => {
                    self.ports[port].insert(piece.rows.start, piece);
                }
                Input::Closed => self.open -= 1,
            }
        }
        for rows in self.aligned() {
            let ports = self.take(&rows)?;
            cx.emit(self.assemble(rows, ports)?);
        }
        if self.open == 0 {
            if self.ports.iter().any(|port| !port.is_empty()) {
                vortex_bail!("Pack fields closed with unaligned rows left over");
            }
            cx.close();
            return Ok(NodeState::Done);
        }
        Ok(NodeState::Waiting)
    }
}

/// Merges a port's pieces into maximal contiguous row ranges.
fn coverage(port: &BTreeMap<u64, Piece>) -> Vec<Range<u64>> {
    let mut ranges: Vec<Range<u64>> = Vec::new();
    for piece in port.values() {
        match ranges.last_mut() {
            Some(last) if last.end == piece.rows.start => last.end = piece.rows.end,
            _ => ranges.push(piece.rows.clone()),
        }
    }
    ranges
}

/// Intersects two sorted, disjoint range lists.
fn intersect(lhs: &[Range<u64>], rhs: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < lhs.len() && j < rhs.len() {
        let start = lhs[i].start.max(rhs[j].start);
        let end = lhs[i].end.min(rhs[j].end);
        if start < end {
            out.push(start..end);
        }
        if lhs[i].end < rhs[j].end {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}
