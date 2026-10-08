// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;

use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::plan::PackPlan;
use crate::plan::exec::Event;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;
use crate::plan::exec::piece::empty_piece;
use crate::plan::exec::piece::join;

/// Assembles one struct piece from fields that arrive with different row alignments and in any
/// order.
///
/// Each port keeps its pieces until every port has closed, then each field is joined in row order
/// and the struct is emitted once. Whether the struct is ready is a count of open ports, so a
/// piece costs the same to take however many are held and whatever order they arrive in.
pub(crate) struct PackNode {
    plan: PackPlan,
    selection: Selection,
    ports: Vec<Vec<Piece>>,
    open: usize,
    started: bool,
}

impl PackNode {
    pub(crate) fn new(plan: PackPlan, selection: Selection) -> Self {
        let nports = plan.children().len();
        Self {
            plan,
            selection,
            ports: vec![Vec::new(); nports],
            open: nports,
            started: false,
        }
    }
}

/// Builds the struct of `plan` over `selection` from the pieces each of its ports produced.
pub(crate) fn assemble(
    plan: &PackPlan,
    selection: &Selection,
    ports: Vec<Vec<Piece>>,
) -> VortexResult<Piece> {
    let fields = plan.fields();
    let rows = selection.rows().clone();
    let mut arrays = Vec::with_capacity(ports.len());
    for (index, mut pieces) in ports.into_iter().enumerate() {
        pieces.sort_by_key(|piece| piece.rows.start);
        let mut covered = rows.start;
        for piece in &pieces {
            if piece.rows.start != covered {
                vortex_bail!(
                    "Pack field {index} closed without rows {:?}",
                    covered..piece.rows.start
                );
            }
            covered = piece.rows.end;
        }
        if covered != rows.end {
            vortex_bail!(
                "Pack field {index} closed without rows {:?}",
                covered..rows.end
            );
        }
        let dtype = fields
            .field_by_index(index)
            .unwrap_or(DType::Bool(Nullability::NonNullable));
        arrays.push(join(
            &dtype,
            pieces.into_iter().map(|piece| piece.array).collect(),
        )?);
    }
    let validity = if plan.dtype().is_nullable() {
        match arrays.pop() {
            Some(validity) => Validity::Array(validity),
            None => vortex_bail!("Nullable Pack is missing its validity port"),
        }
    } else {
        Validity::NonNullable
    };
    let len = selection.mask().true_count();
    let array = StructArray::try_new_with_dtype(arrays, fields.clone(), len, validity)?;
    Ok(Piece {
        rows,
        array: array.into_array(),
    })
}

impl ExecNode for PackNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if !self.started {
            self.started = true;
            let rows = self.selection.rows().clone();
            if self.selection.mask().all_false() || self.ports.is_empty() {
                let piece = if self.ports.is_empty() {
                    assemble(&self.plan, &self.selection, Vec::new())?
                } else {
                    empty_piece(self.plan.dtype(), rows)
                };
                cx.emit(piece);
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

        for event in cx.events() {
            match event {
                Event::Piece(port, piece) => self.ports[port].push(piece),
                Event::Closed(port) => {
                    if self.open == 0 {
                        vortex_bail!("Pack field {port} closed twice");
                    }
                    self.open -= 1;
                }
                event => return Err(event.unexpected("Pack")),
            }
        }
        if self.open > 0 {
            return Ok(NodeState::Wait);
        }
        let ports = mem::take(&mut self.ports);
        cx.emit(assemble(&self.plan, &self.selection, ports)?);
        Ok(NodeState::Done)
    }
}
