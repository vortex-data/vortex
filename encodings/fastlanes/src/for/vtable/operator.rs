// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::OperatorVTable;

use super::FoRVTable;
use crate::FoRArray;

impl OperatorVTable<FoRVTable> for FoRVTable {
    fn pipeline_node(_array: &FoRArray) -> Option<&dyn vortex_array::pipeline::PipelinedNode> {
        None
    }
}
