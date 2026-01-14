// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;

impl dyn Reader + '_ {
    pub fn optimize(&self) -> VortexResult<ReaderRef> {
        todo!()
    }
}
