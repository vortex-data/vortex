// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::BinaryView;
use crate::pipeline::types::{Element, VType};

impl Element for BinaryView {
    fn vtype() -> VType {
        VType::Binary
    }
}
