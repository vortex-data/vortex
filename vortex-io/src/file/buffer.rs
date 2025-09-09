// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::{Arc, LazyLock};

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt};
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::file::{CoalesceWindow, IntoIoSource, IoRequest, IoSource, IoSourceRef};
use crate::runtime::Handle;

impl IntoIoSource for ByteBuffer {
    fn into_io_source(self, _handle: Handle) -> VortexResult<IoSourceRef> {
        Ok(Arc::new(self))
    }
}

impl IoSource for ByteBuffer {
    fn uri(&self) -> &Arc<str> {
        static URI: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from(":in-memory:"));
        &URI
    }

    fn coalesce_window(&self) -> Option<CoalesceWindow> {
        None
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.len() as u64;
        async move { Ok(len) }.boxed()
    }

    fn drive_send(
        self: Arc<Self>,
        mut requests: BoxStream<'static, IoRequest>,
    ) -> BoxFuture<'static, ()> {
        let buffer = self;
        async move {
            while let Some(req) = requests.next().await {
                let offset = usize::try_from(req.offset())
                    .vortex_expect("In-memory buffer offset exceeds usize");
                let len = req.len();

                let result = if offset + len > buffer.len() {
                    Err(vortex_err!("Read out of bounds"))
                } else {
                    let mut slice = ByteBufferMut::with_capacity_aligned(len, req.alignment());
                    unsafe { slice.set_len(len) };
                    slice
                        .as_mut_slice()
                        .copy_from_slice(&buffer.as_slice()[offset..offset + len]);
                    Ok(slice.freeze())
                };
                req.resolve(result);
            }
        }
        .boxed()
    }
}
