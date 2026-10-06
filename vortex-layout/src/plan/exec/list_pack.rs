// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::ListPackPlan;
use crate::plan::exec::Event;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::Selection;
use crate::plan::exec::StepCx;
use crate::plan::exec::empty_piece;
use crate::plan::exec::join;

const OFFSETS: usize = 0;
const VALIDITY: usize = 1;
const ELEMENTS: usize = 2;

/// Reads list offsets first, then the element range they reference. Holes in the parent
/// selection are filtered after assembly, so parent and element row coordinates never mix.
pub(crate) struct ListPackNode {
    plan: ListPackPlan,
    selection: Selection,
    session: VortexSession,
    rows: Range<u64>,
    mask: Mask,
    pieces: [Vec<Piece>; 3],
    open: [bool; 3],
    offsets: Option<ArrayRef>,
    started: bool,
}

impl ListPackNode {
    pub(crate) fn new(plan: ListPackPlan, selection: Selection, session: VortexSession) -> Self {
        Self {
            plan,
            rows: selection.rows().clone(),
            mask: selection.mask().clone(),
            selection,
            session,
            pieces: Default::default(),
            open: [false; 3],
            offsets: None,
            started: false,
        }
    }

    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<bool> {
        let Some(first) = self.mask.first() else {
            cx.emit(empty_piece(
                self.plan.dtype(),
                self.selection.rows().clone(),
            ));
            return Ok(false);
        };
        let last = self
            .mask
            .last()
            .ok_or_else(|| vortex_err!("Missing last selected list"))?;
        self.rows =
            self.rows.start + u64::try_from(first)?..self.rows.start + u64::try_from(last)? + 1;
        self.mask = self.mask.slice(first..last + 1);
        self.open[OFFSETS] = true;
        cx.spawn(
            OFFSETS,
            self.plan.offsets()?,
            self.rows.start..self.rows.end + 1,
            Mask::new_true(self.mask.len() + 1),
        );
        if let Some(validity) = self.plan.validity()? {
            self.open[VALIDITY] = true;
            cx.spawn(
                VALIDITY,
                validity,
                self.rows.clone(),
                Mask::new_true(self.mask.len()),
            );
        }
        Ok(true)
    }

    fn take_pieces(&mut self, port: usize) -> Vec<ArrayRef> {
        let mut pieces = mem::take(&mut self.pieces[port]);
        pieces.sort_by_key(|piece| piece.rows.start);
        pieces.into_iter().map(|piece| piece.array).collect()
    }

    fn start_elements(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()> {
        let offsets = join(self.plan.offsets()?.dtype(), self.take_pieces(OFFSETS))?;
        vortex_ensure!(
            offsets.len() == self.mask.len() + 1,
            "Incomplete list offsets"
        );
        let mut ctx = self.session.create_execution_ctx();
        let start = offsets
            .execute_scalar(0, &mut ctx)?
            .as_primitive()
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("Invalid first list offset"))?;
        let end = offsets
            .execute_scalar(offsets.len() - 1, &mut ctx)?
            .as_primitive()
            .as_::<u64>()
            .ok_or_else(|| vortex_err!("Invalid last list offset"))?;
        let elements = self.plan.elements()?;
        vortex_ensure!(
            start <= end && end <= elements.row_count(),
            "List element range {start}..{end} is out of bounds"
        );
        let offsets = if start == 0 {
            offsets
        } else {
            let base = ConstantArray::new(start, offsets.len())
                .into_array()
                .cast(offsets.dtype().clone())?;
            offsets.binary(base, Operator::Sub)?
        };
        self.offsets = Some(offsets);
        self.open[ELEMENTS] = true;
        cx.spawn(
            ELEMENTS,
            elements,
            start..end,
            Mask::new_true(usize::try_from(end - start)?),
        );
        Ok(())
    }
}

impl ExecNode for ListPackNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if !self.started {
            self.started = true;
            if !self.start(cx)? {
                return Ok(NodeState::Done);
            }
        }
        for event in cx.events() {
            match event {
                Event::Piece(port, piece) if port < self.pieces.len() => {
                    self.pieces[port].push(piece)
                }
                Event::Closed(port) if port < self.open.len() => {
                    vortex_ensure!(self.open[port], "ListPack port {port} closed twice");
                    self.open[port] = false;
                }
                event => return Err(event.unexpected("ListPack")),
            }
        }
        if self.offsets.is_none() && !self.open[OFFSETS] {
            self.start_elements(cx)?;
        }
        if self.open.iter().any(|open| *open) {
            return Ok(NodeState::Wait);
        }
        let elements = join(self.plan.elements()?.dtype(), self.take_pieces(ELEMENTS))?;
        let validity = match self.plan.validity()? {
            Some(plan) => Validity::Array(join(plan.dtype(), self.take_pieces(VALIDITY))?),
            None => Validity::NonNullable,
        };
        let offsets = self
            .offsets
            .take()
            .ok_or_else(|| vortex_err!("Missing list offsets"))?;
        let array = ListArray::try_new(elements, offsets, validity)?
            .into_array()
            .filter(self.mask.clone())?;
        cx.emit(Piece {
            rows: self.selection.rows().clone(),
            array,
        });
        Ok(NodeState::Done)
    }
}
