// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Ready;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;
use crate::plan::exec::selection::join;

const OFFSETS: usize = 0;
const VALIDITY: usize = 1;
const ELEMENTS: usize = 2;

/// Reads list offsets first, then the element range they reference.
///
/// The node runs in two phases. It waits for the offsets to close, since the element range is
/// only known once they are whole, and spawns the elements. It then waits for every port to
/// close and assembles the lists once. Holes in the parent selection are filtered after
/// assembly, so parent and element row coordinates never mix.
pub(crate) struct ListPackNode {
    plan: ListPackPlan,
    session: VortexSession,
    /// The selected lists, narrowed to the first through the last selected one.
    rows: Range<u64>,
    mask: Mask,
    /// The offsets rebased to the element range, once read.
    offsets: Option<ArrayRef>,
}

impl ListPackNode {
    pub(crate) fn new(plan: ListPackPlan, selection: Selection, session: VortexSession) -> Self {
        Self {
            plan,
            rows: selection.rows().clone(),
            mask: selection.mask().clone(),
            session,
            offsets: None,
        }
    }

    fn spawn_elements(&mut self, cx: &mut StepCx<'_>) -> VortexResult<()> {
        let offsets = join(self.plan.offsets()?.dtype(), cx.input(OFFSETS).take_all())?;
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
        cx.spawn(
            ELEMENTS,
            elements,
            start..end,
            Mask::new_true(usize::try_from(end - start)?),
        );
        Ok(())
    }

    fn assemble(&mut self, cx: &mut StepCx<'_>) -> VortexResult<ArrayRef> {
        let elements = join(self.plan.elements()?.dtype(), cx.input(ELEMENTS).take_all())?;
        let validity = match self.plan.validity()? {
            Some(plan) => Validity::Array(join(plan.dtype(), cx.input(VALIDITY).take_all())?),
            None => Validity::NonNullable,
        };
        let offsets = self
            .offsets
            .take()
            .ok_or_else(|| vortex_err!("Missing list offsets"))?;
        ListArray::try_new(elements, offsets, validity)?
            .into_array()
            .filter(self.mask.clone())
    }
}

impl ExecNode for ListPackNode {
    fn ready(&self) -> Ready {
        if self.offsets.is_none() {
            Ready::Closed(&[OFFSETS])
        } else {
            Ready::AllClosed
        }
    }

    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        let Some(first) = self.mask.first() else {
            return Ok(NodeState::Done);
        };
        let last = self
            .mask
            .last()
            .ok_or_else(|| vortex_err!("Missing last selected list"))?;
        self.rows =
            self.rows.start + u64::try_from(first)?..self.rows.start + u64::try_from(last)? + 1;
        self.mask = self.mask.slice(first..last + 1);
        cx.spawn(
            OFFSETS,
            self.plan.offsets()?,
            self.rows.start..self.rows.end + 1,
            Mask::new_true(self.mask.len() + 1),
        );
        if let Some(validity) = self.plan.validity()? {
            cx.spawn(
                VALIDITY,
                validity,
                self.rows.clone(),
                Mask::new_true(self.mask.len()),
            );
        }
        Ok(NodeState::Wait)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.offsets.is_none() {
            self.spawn_elements(cx)?;
            return Ok(NodeState::Wait);
        }
        let lists = self.assemble(cx)?;
        cx.emit(lists);
        Ok(NodeState::Done)
    }
}
