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
    /// The global row index of the graph's first plan row.
    row_offset: u64,
}

impl RowIdxNode {
    pub(crate) fn new(selection: Selection, row_offset: u64) -> Self {
        Self {
            selection,
            row_offset,
        }
    }
}

impl ExecNode for RowIdxNode {
    fn compute(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        let rows = self.selection.rows().clone();
        let offset = self.row_offset;
        let indices = Buffer::from_iter(rows.start + offset..rows.end + offset).into_array();
        let array = if self.selection.mask().all_true() {
            indices
        } else {
            indices.filter(self.selection.mask().clone())?
        };
        cx.emit(Piece { rows, array });
        Ok(NodeState::Done)
    }
}
