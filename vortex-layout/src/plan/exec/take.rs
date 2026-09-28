// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DictArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::plan::TakePlan;
use crate::plan::exec::ExecNode;
use crate::plan::exec::Input;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;
use crate::plan::exec::piece::empty_piece;
use crate::plan::exec::piece::join;

const CODES: usize = 0;
const VALUES: usize = 1;

/// Looks up each selected row's code in the full set of values.
///
/// The codes run over the selection; the values run over their whole domain. Codes pieces that
/// arrive before every value has are held, then each becomes a dictionary piece over the joined
/// values.
///
/// The joined values are kept on the plan, so later executions of it skip the values subtree.
/// Boolean values, which the optimizer produces by pushing a predicate onto the dictionary, are
/// evaluated before they are kept, so the predicate runs once per dictionary rather than once
/// per execution.
pub(crate) struct TakeNode {
    plan: TakePlan,
    selection: Selection,
    started: bool,
    codes: Vec<Piece>,
    codes_open: bool,
    values: Vec<Piece>,
    values_open: bool,
    joined: Option<ArrayRef>,
}

impl TakeNode {
    pub(crate) fn new(plan: TakePlan, selection: Selection) -> Self {
        Self {
            plan,
            selection,
            started: false,
            codes: Vec::new(),
            codes_open: true,
            values: Vec::new(),
            values_open: true,
            joined: None,
        }
    }

    /// Spawns the codes over the selection and the values over their whole domain. Returns false
    /// when nothing is selected, after emitting an empty piece.
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<bool> {
        let rows = self.selection.rows().clone();
        if self.selection.mask().all_false() {
            cx.emit(empty_piece(self.plan.dtype(), rows));
            cx.close();
            return Ok(false);
        }
        cx.spawn(
            CODES,
            self.plan.codes()?,
            rows,
            self.selection.mask().clone(),
        );
        if let Some(values) = self.plan.cached_values() {
            self.joined = Some(values);
            self.values_open = false;
            return Ok(true);
        }
        let values = self.plan.values()?;
        let len = usize::try_from(values.row_count())?;
        cx.spawn(VALUES, values, 0..len as u64, Mask::new_true(len));
        Ok(true)
    }
}

impl ExecNode for TakeNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        if !self.started {
            self.started = true;
            if !self.start(cx)? {
                return Ok(NodeState::Done);
            }
        }
        for (port, input) in cx.take_inputs() {
            match (port, input) {
                (CODES, Input::Piece(piece)) => self.codes.push(piece),
                (CODES, Input::Closed) => self.codes_open = false,
                (VALUES, Input::Piece(piece)) => self.values.push(piece),
                (VALUES, Input::Closed) => self.values_open = false,
                (port, _) => vortex_bail!("Take has no input port {port}"),
            }
        }
        if self.joined.is_none() && !self.values_open {
            self.values.sort_by_key(|piece| piece.rows.start);
            let values = std::mem::take(&mut self.values)
                .into_iter()
                .map(|piece| piece.array)
                .collect();
            let mut values = join(self.plan.values()?.dtype(), values)?;
            if values.dtype().is_boolean() {
                let mut ctx = cx.session().create_execution_ctx();
                values = values.execute::<Canonical>(&mut ctx)?.into_array();
            }
            self.joined = Some(self.plan.cache_values(values));
        }
        let Some(values) = &self.joined else {
            return Ok(NodeState::Waiting);
        };
        for piece in self.codes.drain(..) {
            cx.emit(Piece {
                rows: piece.rows,
                array: DictArray::try_new(piece.array, values.clone())?.into_array(),
            });
        }
        if self.codes_open {
            return Ok(NodeState::Waiting);
        }
        cx.close();
        Ok(NodeState::Done)
    }
}
