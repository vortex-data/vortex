// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{PerformanceHint, ReadAt, VortexIO};
use async_trait::async_trait;
use std::sync::Arc;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexResult, vortex_err};

/// For in-memory bytes, we are able to immediately return a response without dispatching work
/// to another thread.
///
/// Users should be careful when using this implementation with memory-mapped files, as the
/// performance ultimately depends on the underlying memory-mapped file implementation. For
/// example, filesystems backed by network storage may be better suited to dispatched I/O.
#[async_trait]
impl ReadAt for ByteBuffer {
    async fn read_range(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let offset =
            usize::try_from(offset).map_err(|e| vortex_err!("Offset too large for usize: {e}"))?;
        Ok(self
            .slice_unaligned(offset..offset + len)
            .aligned(alignment))
    }

    async fn size(&self) -> VortexResult<u64> {
        Ok(self.len() as u64)
    }

    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
    }
}

impl VortexIO for ByteBuffer {
    fn into_vortex_read_at(self) -> Arc<dyn ReadAt> {
        Arc::new(self)
    }
}
