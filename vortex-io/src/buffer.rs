// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{VortexResult, vortex_err};

use crate::{PerformanceHint, ReadAt, VortexIO};

/// Wrapper for `ByteBuffer` that implements `ReadAt`.
pub(crate) struct ByteBufferReadAt(ByteBuffer);

#[async_trait]
impl ReadAt for ByteBufferReadAt {
    async fn read_range(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let offset =
            usize::try_from(offset).map_err(|e| vortex_err!("Offset too large for usize: {e}"))?;
        Ok(self
            .0
            .slice_unaligned(offset..offset + len)
            .aligned(alignment))
    }

    async fn size(&self) -> VortexResult<u64> {
        Ok(self.0.len() as u64)
    }
}

/// For in-memory bytes, we are able to immediately return a response without dispatching work
/// to another thread.
///
/// Users should be careful when using this implementation with memory-mapped files, as the
/// performance ultimately depends on the underlying memory-mapped file implementation. For
/// example, filesystems backed by network storage may be better suited to dispatched I/O.
impl VortexIO for ByteBuffer {
    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::in_memory()
    }

    fn into_read_at(self) -> VortexResult<Arc<dyn ReadAt>> {
        Ok(Arc::new(ByteBufferReadAt(self)))
    }
}

/// See the [`ByteBuffer`] implementation for details.
impl VortexIO for ByteBufferMut {
    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::in_memory()
    }

    fn into_read_at(self) -> VortexResult<Arc<dyn ReadAt>> {
        self.freeze().into_read_at()
    }
}
