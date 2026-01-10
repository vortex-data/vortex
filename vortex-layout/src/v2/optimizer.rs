// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::v2::reader::LayoutReader2;
use crate::v2::reader::LayoutReader2Ref;

impl dyn LayoutReader2 + '_ {
    pub fn optimize(&self) -> VortexResult<LayoutReader2Ref> {
        todo!()
    }
}
