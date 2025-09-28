// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_error::VortexResult;

use crate::compute::filter;
use crate::operator::{BatchExecution, MaskExecution};
use crate::{Array, Canonical};

pub struct CanonicalExecution {
    canonical: Canonical,
    mask: MaskExecution,
}

impl CanonicalExecution {
    pub fn new(canonical: Canonical, mask: MaskExecution) -> Self {
        Self { canonical, mask }
    }
}

#[async_trait]
impl BatchExecution for CanonicalExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let mask = self.mask.await?;
        Ok(if !mask.all_true() {
            filter(self.canonical.as_ref(), &mask)?.to_canonical()
        } else {
            self.canonical
        })
    }
}
