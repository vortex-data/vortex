// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::compat::Compat;

/// Compatibility adapter for `VortexReadAt` implementations that are based on Tokio.
#[deny(clippy::missing_trait_methods)]
impl<R: VortexReadAt> VortexReadAt for Compat<R> {
    fn uri(&self) -> Option<&Arc<str>> {
        self.inner().uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.inner().coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.inner().concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        Compat::new(self.inner().size()).boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        Compat::new(self.inner().read_at(offset, length, alignment)).boxed()
    }
}
