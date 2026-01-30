// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;

use crate::stream::VortexCudaStream;

/// A wrapper that uses an allocator to produce the returned buffer handle.
#[derive(Clone)]
pub struct CopyDeviceReadAt<T: VortexReadAt + Clone> {
    read: T,
    stream: VortexCudaStream,
}

impl<T: VortexReadAt + Clone> CopyDeviceReadAt<T> {
    pub fn new(read: T, stream: VortexCudaStream) -> Self {
        Self { read, stream }
    }
}

impl<T: VortexReadAt + Clone> VortexReadAt for CopyDeviceReadAt<T> {
    fn uri(&self) -> Option<&Arc<str>> {
        self.read.uri()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.read.coalesce_config()
    }

    fn concurrency(&self) -> usize {
        self.read.concurrency()
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        self.read.size()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let read = self.read.clone();
        let stream = self.stream.clone();
        async move {
            let handle = read.read_at(offset, length, alignment).await?;
            if handle.is_on_device() {
                return Ok(handle);
            }

            let host_buffer = handle.as_host().clone();

            stream.copy_to_device(host_buffer)?.await
        }
        .boxed()
    }
}
