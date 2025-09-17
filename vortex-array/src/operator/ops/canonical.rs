// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::operator::BatchExecution;

pub struct CanonicalExecution(pub Canonical);

#[async_trait]
impl BatchExecution for CanonicalExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        Ok(self.0)
    }
}
