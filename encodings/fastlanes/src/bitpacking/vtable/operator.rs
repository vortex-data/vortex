// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::pipeline::PipelinedNode;
use vortex_array::vtable::OperatorVTable;

use crate::{BitPackedArray, BitPackedVTable};

impl OperatorVTable<BitPackedVTable> for BitPackedVTable {
    fn pipeline_node(array: &BitPackedArray) -> Option<&dyn PipelinedNode> {
        Some(array)
    }
}
