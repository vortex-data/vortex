// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;
use std::sync::OnceLock;

use async_compat::Compat;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::path::Path as ObjectPath;
use tokio::io::AsyncReadExt;
use tokio_util::io::StreamReader;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::WriteTarget;
#[cfg(not(target_arch = "wasm32"))]
use crate::file::std_file::read_exact_at;
use crate::read::record_copy;
use crate::runtime::Handle;

const DEFAULT_COALESCING_CONFIG: CoalesceConfig = CoalesceConfig {
    distance: 1024 * 1024,      // 1 MB
    max_size: 16 * 1024 * 1024, // 16 MB
};

/// Default number of concurrent requests to allow.
const DEFAULT_CONCURRENCY: usize = 192;
const STREAM_READ_ENV: &str = "VORTEX_S3_STREAM_READ_EXACT";
const READ_CONCURRENCY_ENV: &str = "VORTEX_S3_READ_CONCURRENCY";
const COALESCE_DISTANCE_ENV: &str = "VORTEX_S3_COALESCE_DISTANCE";
const COALESCE_MAX_SIZE_ENV: &str = "VORTEX_S3_COALESCE_MAX_SIZE";
const COALESCE_DISABLE_ENV: &str = "VORTEX_S3_COALESCE_DISABLE";
const COPY_STATS_ENV: &str = "VORTEX_S3_COPY_STATS";

/// An object store backed I/O source.
pub struct ObjectStoreSource {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
}

impl ObjectStoreSource {
    /// Create a new object store source.
    pub fn new(store: Arc<dyn ObjectStore>, path: ObjectPath, handle: Handle) -> Self {
        let uri = Arc::from(path.to_string());
        let mut coalesce_config = Some(DEFAULT_COALESCING_CONFIG);
        if read_env_bool(COALESCE_DISABLE_ENV, false) {
            coalesce_config = None;
        } else if let Some(defaults) = coalesce_config.as_mut() {
            if let Some(distance) = read_env_u64(COALESCE_DISTANCE_ENV) {
                defaults.distance = distance;
            }
            if let Some(max_size) = read_env_u64(COALESCE_MAX_SIZE_ENV) {
                defaults.max_size = max_size;
            }
        }

        Self {
            store,
            path,
            uri,
            handle,
            concurrency: read_env_usize(READ_CONCURRENCY_ENV).unwrap_or(DEFAULT_CONCURRENCY),
            coalesce_config,
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

    /// Set an optional coalesce config for this source.
    pub fn with_some_coalesce_config(mut self, config: Option<CoalesceConfig>) -> Self {
        self.coalesce_config = config;
        self
    }
}

fn use_stream_reader() -> bool {
    read_env_bool(STREAM_READ_ENV, false)
}

fn copy_stats_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| read_env_bool(COPY_STATS_ENV, true))
}

fn read_env_u64(key: &str) -> Option<u64> {
    std::env::var(key).ok()?.parse::<u64>().ok()
}

fn read_env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse::<usize>().ok()
}

fn read_env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u8>().ok())
        .map(|value| value != 0)
        .unwrap_or(default)
}

impl VortexReadAt for ObjectStoreSource {
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
        let store = self.store.clone();
        let path = self.path.clone();
        Compat::new(async move {
            store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)
        })
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let mut buffer = ByteBufferMut::with_capacity_aligned(length, alignment);
        unsafe { buffer.set_len(length) };
        let target: Box<dyn WriteTarget> = Box::new(buffer);
        self.read_at_into(offset, target)
    }

    fn read_at_into(
        &self,
        offset: u64,
        mut target: Box<dyn WriteTarget>,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = self.store.clone();
        let path = self.path.clone();
        let handle = self.handle.clone();
        let length = target.len();
        let range = offset..(offset + length as u64);
        let collect_copy_stats = copy_stats_enabled();

        Compat::new(async move {
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
                    if use_stream_reader() {
                        let mut reader = StreamReader::new(byte_stream.map_err(io::Error::other));
                        let copy_start = std::time::Instant::now();
                        reader.read_exact(target.as_mut_slice()).await?;
                        if collect_copy_stats {
                            record_copy(length, copy_start.elapsed());
                        }
                    } else {
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
                            let copy_start = std::time::Instant::now();
                            target.as_mut_slice()[filled..end].copy_from_slice(&bytes);
                            if collect_copy_stats {
                                record_copy(bytes.len(), copy_start.elapsed());
                            }
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
            }

            target.into_handle()
        })
        .boxed()
    }
}
