// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;

impl dyn Reader + '_ {
    pub fn optimize(self: Arc<Self>) -> VortexResult<ReaderRef> {
        // TODO(ngates): run the reduce rules
        Ok(self)
    }
}
