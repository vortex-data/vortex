// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::future::{BoxFuture, FusedFuture, Shared};
use futures::{FutureExt, TryFutureExt, pin_mut, select};
use vortex_error::{SharedVortexResult, VortexResult, vortex_panic};
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
            inner: fut.map_err(Arc::new)
                .inspect(move |r| {
                    if let Ok(mask) = r
                        && mask.len() != len {
                            vortex_panic!("MaskFuture created with future that returned mask of incorrect length (expected {}, got {})", len, mask.len());
                        }
                }).boxed().shared(),
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

    /// Race this mask future against another future, returning the result of the
    /// other future along with the resolved mask.
    ///
    /// If the mask resolves to all false, returns None immediately without waiting
    /// for the other future
    pub async fn race<T, F>(&self, other: F) -> VortexResult<Option<(T, Mask)>>
    where
        F: Future<Output = VortexResult<T>> + FusedFuture,
    {
        let mut mask = self.clone().fuse();
        pin_mut!(other);

        let (result, mask) = select! {
            mask_result = mask => {
                let mask = mask_result?;
                if mask.all_false() {
                    // Early return - no need to wait for the other side
                    return Ok(None);
                }
                // Need to wait for array since mask isn't all false
                let other = other.await?;
                (other, mask)
            }
            other_result = other => {
                let other = other_result?;
                let mask = mask.await?;
                if mask.all_false() {
                    // Still early return, even if the other side resolved first.
                    return Ok(None);
                }
                (other, mask)
            }
        };

        Ok(Some((result, mask)))
    }
}

impl Future for MaskFuture {
    type Output = VortexResult<Mask>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.poll_unpin(cx)
    }
}
