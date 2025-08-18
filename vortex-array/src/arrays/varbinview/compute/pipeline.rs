// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_vector::types::{Element, VType};

use crate::arrays::BinaryView;

impl Element for BinaryView {
    fn vtype() -> VType {
        VType::Binary
    }
}
