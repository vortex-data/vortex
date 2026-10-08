// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::plan::exec::ExecNode;
use crate::plan::exec::NodeState;
use crate::plan::exec::StepCx;
use crate::plan::exec::selection::Selection;

/// Produces the global row index of every selected row.
///
/// Row-index plans sit in the graph's root row domain, so a row's global index is the graph's
/// row offset plus its plan row. The node has no input and emits once, at start.
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
    fn start(&mut self, cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        let rows = self.selection.rows();
        let indices = Buffer::from_iter(rows.start + self.row_offset..rows.end + self.row_offset)
            .into_array();
        let array = if self.selection.mask().all_true() {
            indices
        } else {
            indices.filter(self.selection.mask().clone())?
        };
        cx.emit(array);
        Ok(NodeState::Done)
    }

    fn compute(&mut self, _cx: &mut StepCx<'_>) -> VortexResult<NodeState> {
        Ok(NodeState::Done)
    }
}
