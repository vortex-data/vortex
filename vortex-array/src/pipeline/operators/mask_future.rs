// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::ops::Range;
use std::sync::Arc;

use futures_util::future::{BoxFuture, Shared};
use futures_util::{FutureExt, TryFutureExt};
use vortex_error::{SharedVortexResult, VortexError, VortexResult, vortex_panic};
use vortex_mask::Mask;

/// A future that resolves to a mask.
#[derive(Clone)]
pub struct MaskFuture {
    inner: Shared<BoxFuture<'static, SharedVortexResult<Mask>>>,
    len: usize,
}

impl MaskFuture {
    /// Create a new MaskFuture from a future that returns a mask.
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

    /// Returns the length of the mask.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the mask is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Create a MaskFuture from a ready mask.
    pub fn ready(mask: Mask) -> Self {
        Self::new(mask.len(), async move { Ok(mask) })
    }

    /// Create a MaskFuture that resolves to a mask with all values set to true.
    pub fn new_true(row_count: usize) -> Self {
        Self::ready(Mask::new_true(row_count))
    }

    /// Create a MaskFuture that resolves to a slice of the original mask.
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
