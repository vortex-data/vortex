// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::os::unix::prelude::FileExt;
use std::sync::Arc;

use vortex_buffer::{Alignment, ByteBuffer, ByteBufferMut};
use vortex_error::{ResultExt, VortexExpect, VortexResult, vortex_err};

use crate::compio::CompioDispatchedIo;
use crate::tokio::TokioReadAt;
use crate::{PerformanceHint, ReadAt, VortexIO};

/// Opens a `std::fs::File` as a Vortex I/O object.
///
/// Currently, std I/O internally dispatches blocking I/O tasks onto a Tokio runtime.
// impl VortexIO for std::fs::File {
//     fn performance_hint(&self) -> PerformanceHint {
//         PerformanceHint::local()
//     }
//
//     fn into_read_at(self) -> VortexResult<Arc<dyn ReadAt>> {
//         Ok(Arc::new(TokioDispatchedIo::new(Arc::new(self))))
//     }
// }

impl VortexIO for &std::path::Path {
    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::local()
    }

    fn into_read_at(self) -> VortexResult<Arc<dyn ReadAt>> {
        let size = self
            .metadata()
            .vortex_expect("Failed to get file metadata")
            .len();
        let path = self.to_owned();
        Ok(Arc::new(CompioDispatchedIo::new(
            || async move {
                Ok(compio::fs::File::open(path)
                    .await
                    .vortex_expect("Failed to open compio file"))
            },
            size,
        )))
        // std::fs::File::open(self)
        //     .map_err(|e| vortex_err!("Failed to open file {e}"))?
        //     .into_read_at()
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

    async fn size(&self) -> VortexResult<u64> {
        let this = self.clone();

        tokio::task::spawn_blocking(move || {
            Ok(this
                .metadata()
                .map_err(|e| vortex_err!("Failed to get file metadata: {e}"))?
                .len())
        })
        .await
        .map_err(|e| vortex_err!("Failed to spawn blocking task: {e}"))
        .unnest()
    }
}
