// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::BatchExecution;
use crate::Canonical;
use vortex_error::VortexResult;

pub(super) struct CanonicalExecution(pub(super) Canonical);

impl BatchExecution for CanonicalExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        Ok(self.0)
    }
}
