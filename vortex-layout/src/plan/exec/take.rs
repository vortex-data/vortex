// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::plan::TakePlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Ready;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;
use crate::plan::exec::selection::join;

const CODES: usize = 0;
const VALUES: usize = 1;

/// Looks up each selected row's code in the full set of values.
///
/// The codes run over the selection; the values run over their whole domain. The node does not
/// run until the values have closed, so the first compute joins them once, and every compute
/// wraps the codes arrays that have arrived as dictionaries over the joined values.
pub(crate) struct TakeNode {
    plan: TakePlan,
    selection: Selection,
    values: Option<ArrayRef>,
}

impl TakeNode {
    pub(crate) fn new(plan: TakePlan, selection: Selection) -> Self {
        Self {
            plan,
            selection,
            values: None,
        }
    }
}

impl ExecNode for TakeNode {
    fn ready(&self) -> Ready {
        Ready::Closed(&[VALUES])
    }

    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if self.selection.mask().all_false() {
            return Ok(NodeState::Done);
        }
        cx.spawn(
            CODES,
            self.plan.codes()?,
            self.selection.rows().clone(),
            self.selection.mask().clone(),
        );
        let values = self.plan.values()?;
        let len = usize::try_from(values.row_count())?;
        cx.spawn(VALUES, values, 0..len as u64, Mask::new_true(len));
        Ok(NodeState::Wait)
    }

    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        let values = match &self.values {
            Some(values) => values.clone(),
            None => {
                let values = join(self.plan.values()?.dtype(), cx.input(VALUES).take_all())?;
                self.values = Some(values.clone());
                values
            }
        };
        for codes in cx.input(CODES).take_all() {
            cx.emit(DictArray::try_new(codes, values.clone())?.into_array());
        }
        if cx.input(CODES).finished() {
            return Ok(NodeState::Done);
        }
        Ok(NodeState::Wait)
    }
}
