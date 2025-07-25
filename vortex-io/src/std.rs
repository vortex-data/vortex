// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::tokio::{TokioDispatchedIo, TokioReadAt};
use crate::{PerformanceHint, ReadAt, VortexIO};
use std::os::unix::prelude::FileExt;
use std::sync::Arc;
use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{ResultExt, VortexResult, vortex_err};

/// Opens a `std::fs::File` as a Vortex I/O object.
///
/// Currently, std I/O internally dispatches blocking I/O tasks onto a Tokio runtime.
impl VortexIO for std::fs::File {
    fn into_vortex_read_at(self) -> Arc<dyn ReadAt> {
        Arc::new(TokioDispatchedIo::new(
            Arc::new(self),
            PerformanceHint::local(),
        ))
    }
}

impl TokioReadAt for Arc<std::fs::File> {
    async fn read_at(
        &self,
        offset: u64,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<ByteBuffer> {
        let this = self.clone();

        tokio::task::spawn_blocking(move || {
            let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);
            unsafe { buffer.set_len(len) };
            this.read_exact_at(&mut buffer, offset)?;
            Ok(buffer.freeze())
        })
        .await
        .map_err(|e| vortex_err!("Failed to spawn blocking task: {e}"))
        .unnest()
    }
}
