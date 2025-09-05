// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use async_compat::Compat;
use blocking::unblock;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use vortex_buffer::ByteBufferMut;
use vortex_error::{VortexError, VortexResult};

use crate::file::{IoRequest, IoSource};

const COALESCING_WINDOW: u64 = 4 * 1024 * 1024; // 4 MB
const CONCURRENCY: usize = 192; // Number of concurrent requests to allow.

#[cfg(feature = "object_store")]
pub struct ObjectStoreIo {
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
    uri: Arc<str>,
    concurrency: usize,
    coalesce_window: Option<u64>,
}

#[cfg(feature = "object_store")]
impl ObjectStoreIo {
    pub fn new(store: Arc<dyn object_store::ObjectStore>, path: object_store::path::Path) -> Self {
        let uri = Arc::from(path.to_string());
        Self {
            store,
            path,
            uri,
            concurrency: CONCURRENCY,
            coalesce_window: Some(COALESCING_WINDOW),
        }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    pub fn with_coalesce_window(mut self, window: u64) -> Self {
        self.coalesce_window = Some(window);
        self
    }

    pub fn with_some_coalesce_window(mut self, window: Option<u64>) -> Self {
        self.coalesce_window = window;
        self
    }
}

#[cfg(feature = "object_store")]
impl IoSource for ObjectStoreIo {
    fn uri(&self) -> &Arc<str> {
        &self.uri
    }

    fn coalescing_window(&self) -> Option<u64> {
        self.coalesce_window
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = self.store.clone();
        let path = self.path.clone();
        Compat::new(async move {
            Ok(store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)?)
        })
        .boxed()
    }

    fn drive_send(&self, requests: BoxStream<'static, IoRequest>) -> BoxFuture<'static, ()> {
        let store = self.store.clone();
        let path = self.path.clone();

        requests
            .map(move |req| {
                let store = store.clone();
                let path = path.clone();

                let len = req.len();
                let offset = req.offset();
                let alignment = req.alignment();

                let read = async move {
                    // Instead of calling `ObjectStore::get_range`, we expand the implementation and run it
                    // ourselves to avoid a second copy to align the buffer. Instead, we can write directly
                    // into the aligned buffer.
                    let mut buffer = ByteBufferMut::with_capacity_aligned(len, alignment);

                    let response = store
                        .get_opts(
                            &path,
                            object_store::GetOptions {
                                range: Some(object_store::GetRange::Bounded(
                                    offset..offset + len as u64,
                                )),
                                ..Default::default()
                            },
                        )
                        .await?;

                    let buffer = match response.payload {
                        object_store::GetResultPayload::File(file, _) => {
                            // SAFETY: We're setting the length to the exact size we're about to read.
                            // The read_exact_at call will either fill the entire buffer or return an error,
                            // ensuring no uninitialized memory is exposed.
                            unsafe { buffer.set_len(len) };
                            unblock(move || {
                                file.read_exact_at(&mut buffer, offset)?;
                                Ok::<_, io::Error>(buffer)
                            })
                            .await
                            .map_err(io::Error::other)?
                        }
                        object_store::GetResultPayload::Stream(mut byte_stream) => {
                            while let Some(bytes) = byte_stream.next().await {
                                buffer.extend_from_slice(&bytes?);
                            }
                            buffer
                        }
                    };

                    Ok(buffer.freeze())
                };

                async move {
                    let result = Compat::new(read).await;
                    req.resolve(result);
                }
            })
            .buffer_unordered(self.concurrency)
            .collect::<()>()
            .boxed()
    }
}
