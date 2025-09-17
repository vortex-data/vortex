// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::BatchExecution;
use crate::Canonical;
use async_trait::async_trait;
use vortex_error::VortexResult;

pub struct CanonicalExecution(pub Canonical);

#[async_trait]
impl BatchExecution for CanonicalExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        Ok(self.0)
    }
}
