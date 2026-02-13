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
use object_store::path::Path as ObjectPath;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::CoalesceConfig;
use crate::ReadInto;
use crate::WriteTarget;
use crate::read_at::AllocatingReader;
use crate::runtime::Handle;
#[cfg(not(target_arch = "wasm32"))]
use crate::std_file::read_exact_at;

/// Default number of concurrent requests to allow.
pub const DEFAULT_CONCURRENCY: usize = 192;

/// Low-level object store reader that implements [`ReadInto`].
pub struct ObjectStoreReader {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    handle: Handle,
}

impl ObjectStoreReader {
    /// Create a new object store reader.
    pub fn new(store: Arc<dyn ObjectStore>, path: ObjectPath, handle: Handle) -> Self {
        Self {
            store,
            path,
            handle,
        }
    }
}

impl ReadInto for ObjectStoreReader {
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = self.store.clone();
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

    fn read_into(
        &self,
        mut target: Box<dyn WriteTarget>,
        offset: u64,
    ) -> BoxFuture<'static, VortexResult<Box<dyn WriteTarget>>> {
        let store = self.store.clone();
        let path = self.path.clone();
        let handle = self.handle.clone();
        let length = target.len();
        let range = offset..(offset + length as u64);

        async move {
            let response = store
                .get_opts(
                    &path,
                    GetOptions {
                        range: Some(GetRange::Bounded(range.clone())),
                        ..Default::default()
                    },
                )
                .await?;

            match response.payload {
                #[cfg(not(target_arch = "wasm32"))]
                GetResultPayload::File(file, _) => {
                    target = handle
                        .spawn_blocking(move || {
                            let mut target = target;
                            read_exact_at(&file, target.as_mut_slice(), range.start)?;
                            Ok::<_, io::Error>(target)
                        })
                        .await
                        .map_err(io::Error::other)?;
                }
                #[cfg(target_arch = "wasm32")]
                GetResultPayload::File(..) => {
                    unreachable!("File payload not supported on wasm32")
                }
                GetResultPayload::Stream(mut byte_stream) => {
                    let mut filled = 0usize;
                    while let Some(bytes) = byte_stream.next().await {
                        let bytes = bytes?;
                        let end = filled + bytes.len();
                        vortex_ensure!(
                            end <= length,
                            "Object store stream returned more bytes than expected (expected {} bytes, got at least {} bytes, range: {:?})",
                            length,
                            end,
                            range
                        );
                        target.as_mut_slice()[filled..end].copy_from_slice(&bytes);
                        filled = end;
                    }

                    vortex_ensure!(
                        filled == length,
                        "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                        filled,
                        length,
                        range
                    );
                }
            }

            Ok(target)
        }
        .boxed()
    }
}

/// An object store backed I/O source.
///
/// This is a convenience alias for [`AllocatingReader<ObjectStoreReader>`] using the default
/// allocator.
pub type ObjectStoreReadAt = AllocatingReader<ObjectStoreReader>;

impl ObjectStoreReadAt {
    /// Create a new object store source with the default allocator.
    pub fn new(store: Arc<dyn ObjectStore>, path: ObjectPath, handle: Handle) -> Self {
        let uri: Arc<str> = Arc::from(path.to_string());
        let reader = ObjectStoreReader::new(store, path, handle);
        AllocatingReader::with_default_allocator(reader, DEFAULT_CONCURRENCY)
            .with_uri(uri)
            .with_coalesce_config(CoalesceConfig::object_storage())
    }
}
