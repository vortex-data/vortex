// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::HostAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use crate::std_file::read_exact_at;

/// Default number of concurrent requests to allow.
pub const DEFAULT_CONCURRENCY: usize = 192;

/// An object store backed I/O source.
pub struct ObjectStoreReadAt {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    allocator: HostAllocatorRef,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
}

impl ObjectStoreReadAt {
    /// Create a new object store source with a custom writable buffer allocator.
    pub fn new(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        handle: Handle,
        allocator: HostAllocatorRef,
    ) -> Self {
        let uri = Arc::from(path.to_string());
        Self {
            store,
            path,
            uri,
            handle,
            allocator,
            concurrency: DEFAULT_CONCURRENCY,
            coalesce_config: Some(CoalesceConfig::object_storage()),
        }
    }

    /// Set the concurrency for this source.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Set the coalesce config for this source.
    pub fn with_coalesce_config(mut self, config: CoalesceConfig) -> Self {
        self.coalesce_config = Some(config);
        self
    }
}

impl VortexReadAt for ObjectStoreReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        async move {
            store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let allocator = Arc::clone(&self.allocator);
        let range = offset..(offset + length as u64);

        async move {
            let mut buffer = allocator.allocate(length, alignment)?;

            let response = store
                .get_opts(
                    &path,
                    GetOptions {
                        range: Some(GetRange::Bounded(range.clone())),
                        ..Default::default()
                    },
                )
                .await?;

            let buffer = match response.payload {
                #[cfg(not(target_arch = "wasm32"))]
                GetResultPayload::File(file, _) => {
                    handle
                        .spawn_blocking(move || {
                            read_exact_at(&file, buffer.as_mut_slice(), range.start)?;
                            Ok::<_, io::Error>(buffer)
                        })
                        .await
                        .map_err(io::Error::other)?
                }
                #[cfg(target_arch = "wasm32")]
                GetResultPayload::File(..) => {
                    unreachable!("File payload not supported on wasm32")
                }
                GetResultPayload::Stream(mut byte_stream) => {
                    let mut written = 0usize;
                    while let Some(bytes) = byte_stream.next().await {
                        let bytes = bytes?;
                        let end = written + bytes.len();
                        vortex_ensure!(
                            end <= length,
                            "Object store stream returned too many bytes: {} > expected {} (range: {:?})",
                            end,
                            length,
                            range
                        );
                        buffer.as_mut_slice()[written..end].copy_from_slice(&bytes);
                        written = end;
                    }

                    vortex_ensure!(
                        written == length,
                        "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                        written,
                        length,
                        range
                    );

                    buffer
                }
            };

            Ok(BufferHandle::new_host(buffer.freeze()))
        }
        .boxed()
    }
}
