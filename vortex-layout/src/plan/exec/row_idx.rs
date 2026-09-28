// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::Piece;
use crate::plan::exec::StepCx;
use crate::plan::exec::piece::Selection;

/// Produces the global row index of every selected row.
///
/// Row-index plans sit in the graph's root row domain, so a row's global index is the graph's
/// row offset plus its plan row.
pub(crate) struct RowIdxNode {
    selection: Selection,
}

impl RowIdxNode {
    pub(crate) fn new(selection: Selection) -> Self {
        Self { selection }
    }
}

impl ExecNode for RowIdxNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        let rows = self.selection.rows().clone();
        let offset = cx.row_offset();
        let indices = Buffer::from_iter(rows.start + offset..rows.end + offset).into_array();
        let array = if self.selection.mask().all_true() {
            indices
        } else {
            indices.filter(self.selection.mask().clone())?
        };
        cx.emit(Piece { rows, array });
        cx.close();
        Ok(NodeState::Done)
    }
}
