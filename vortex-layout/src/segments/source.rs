// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::segments::SegmentId;
use futures::future::BoxFuture;
use futures::FutureExt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_panic, VortexResult};

/// Future resolving to a segment byte buffer that depends only on the runtime.
pub struct SegmentFuture<'rt> {
    inner: BoxFuture<'rt, VortexResult<ByteBuffer>>,
    size: usize,
}

impl<'rt> SegmentFuture<'rt> {
    /// Creates a new `SegmentFuture` with the given size and future.
    pub fn new<F: Future<Output = VortexResult<ByteBuffer>> + Send + 'rt>(
        size: usize,
        fut: F,
    ) -> Self {
        Self {
            inner: fut
                .inspect(move |r| {
                    if let Ok(buffer) = r {
                        if buffer.len() != size {
                            vortex_panic!(
                                "SegmentFuture expected size {}, got {}",
                                size,
                                buffer.len()
                            )
                        }
                    }
                })
                .boxed(),
            size,
        }
    }

    /// Returns the size of the segment in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<'rt> Future for SegmentFuture<'rt> {
    type Output = VortexResult<ByteBuffer>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.poll_unpin(cx)
    }
}

pub type SegmentSourceRef<'rt> = Arc<dyn SegmentSource<'rt> + 'rt>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource<'rt>: 'rt + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> VortexResult<SegmentFuture<'rt>>;
}
