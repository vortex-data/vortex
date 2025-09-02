// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::future::{BoxFuture, Shared};
use futures::{FutureExt, TryFutureExt};
use vortex_error::{SharedVortexResult, VortexError, VortexResult, vortex_panic};
use vortex_mask::Mask;

#[derive(Clone)]
pub struct MaskFuture {
    inner: Shared<BoxFuture<'static, SharedVortexResult<Mask>>>,
    len: usize,
}

impl MaskFuture {
    pub fn new<F>(len: usize, fut: F) -> Self
    where
        F: Future<Output = VortexResult<Mask>> + Send + 'static,
    {
        Self {
            inner: fut
                .inspect(move |r| {
                    if let Ok(mask) = r
                        && mask.len() != len {
                            vortex_panic!("MaskFuture created with future that returned mask of incorrect length (expected {}, got {})", len, mask.len());
                        }
                })
                .map_err(Arc::new)
                .boxed()
                .shared(),
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn ready(mask: Mask) -> Self {
        Self::new(mask.len(), async move { Ok(mask) })
    }

    pub fn new_true(row_count: usize) -> Self {
        Self::ready(Mask::new_true(row_count))
    }

    pub fn slice(&self, range: Range<usize>) -> Self {
        let inner = self.inner.clone();
        Self::new(range.len(), async move { Ok(inner.await?.slice(range)) })
    }
}

impl Future for MaskFuture {
    type Output = VortexResult<Mask>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.inner.poll_unpin(cx).map_err(VortexError::from)
    }
}
