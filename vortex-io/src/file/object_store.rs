// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use async_compat::Compat;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use tracing::Instrument;
use vortex_buffer::ByteBufferMut;
use vortex_error::{VortexError, VortexResult, vortex_ensure};

use crate::file::IoRequest;
use crate::file::read::{CoalesceWindow, IntoReadSource, ReadSource, ReadSourceRef};
use crate::runtime::Handle;

const COALESCING_WINDOW: CoalesceWindow = CoalesceWindow {
    distance: 1024 * 1024,      // 1 MB
    max_size: 16 * 1024 * 1024, // 16 MB
};
const CONCURRENCY: usize = 192; // Number of concurrent requests to allow.

pub struct ObjectStoreReadSource {
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
    uri: Arc<str>,
    concurrency: usize,
    coalesce_window: Option<CoalesceWindow>,
}

impl ObjectStoreReadSource {
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

    pub fn with_coalesce_window(mut self, window: CoalesceWindow) -> Self {
        self.coalesce_window = Some(window);
        self
    }

    pub fn with_some_coalesce_window(mut self, window: Option<CoalesceWindow>) -> Self {
        self.coalesce_window = window;
        self
    }
}

impl IntoReadSource for ObjectStoreReadSource {
    fn into_read_source(self, handle: Handle) -> VortexResult<ReadSourceRef> {
        Ok(Arc::new(ObjectStoreIoSource { io: self, handle }))
    }
}

struct ObjectStoreIoSource {
    io: ObjectStoreReadSource,
    handle: Handle,
}

impl ReadSource for ObjectStoreIoSource {
    fn uri(&self) -> &Arc<str> {
        &self.io.uri
    }

    fn coalesce_window(&self) -> Option<CoalesceWindow> {
        self.io.coalesce_window
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = self.io.store.clone();
        let path = self.io.path.clone();
        Compat::new(async move {
            store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)
        })
        .boxed()
    }

    fn drive_send(
        self: Arc<Self>,
        requests: BoxStream<'static, IoRequest>,
    ) -> BoxFuture<'static, ()> {
        let self2 = self.clone();
        requests
            .map(move |req| {
                let handle = self.handle.clone();
                let store = self.io.store.clone();
                let path = self.io.path.clone();

                let len = req.len();
                let range = req.range();
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
                                range: Some(object_store::GetRange::Bounded(range.clone())),
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
                            handle
                                .spawn_blocking(move || {
                                    file.read_exact_at(&mut buffer, range.start)?;
                                    Ok::<_, io::Error>(buffer)
                                })
                                .await
                                .map_err(io::Error::other)?
                        }
                        object_store::GetResultPayload::Stream(mut byte_stream) => {
                            while let Some(bytes) = byte_stream.next().await {
                                buffer.extend_from_slice(&bytes?);
                            }

                            vortex_ensure!(
                                buffer.len() == len,
                                "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                                buffer.len(),
                                len,
                                range
                            );

                            buffer
                        }
                    };

                    Ok(buffer.freeze())
                }
                .in_current_span();

                async move { req.resolve(Compat::new(read).await) }
            })
            .map(move |f| self2.handle.spawn(f))
            .buffer_unordered(CONCURRENCY)
            .collect::<()>()
            .boxed()
    }
}
