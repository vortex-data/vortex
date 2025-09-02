// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::{BoxFuture, FusedFuture, Shared};
use futures::{pin_mut, select, FutureExt, TryFutureExt};
use std::ops::Range;
use std::sync::Arc;
use vortex_error::{vortex_panic, SharedVortexResult, VortexResult};
use vortex_mask::Mask;

#[derive(Clone)]
pub struct MaskFuture<'rt> {
    inner: Shared<BoxFuture<'rt, SharedVortexResult<Mask>>>,
    len: usize,
}

impl<'rt> MaskFuture<'rt> {
    pub fn new<F>(len: usize, fut: F) -> Self
    where
        F: Future<Output = VortexResult<Mask>> + Send + 'rt,
    {
        Self {
            inner: fut.map_err(Arc::new)
                .inspect(move |r| {
                    if let Ok(mask) = r {
                        if mask.len() != len {
                            vortex_panic!("MaskFuture created with future that returned mask of incorrect length (expected {}, got {})", len, mask.len());
                        }
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

    pub fn slice(self, range: Range<usize>) -> Self {
        Self::new(
            range.len(),
            async move { Ok(self.inner.await?.slice(range)) },
        )
    }

    /// Race this mask future against another future, returning the result of the
    /// other future along with the resolved mask.
    ///
    /// If the mask resolves to all false, returns None immediately without waiting
    /// for the other future
    pub async fn race_all_false<T, F>(self, other: F) -> VortexResult<Raced<T>>
    where
        F: Future<Output = VortexResult<T>> + FusedFuture,
    {
        let mut mask = self.fuse();
        pin_mut!(other);

        let (result, mask) = select! {
            mask_result = mask => {
                let mask = mask_result?;
                if mask.all_false() {
                    // Early return - no need to wait for the other side
                    return Ok(Raced::AllFalse(mask));
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
                    return Ok(Raced::AllFalse(mask));
                }
                (other, mask)
            }
        };

        Ok(Raced::Result((result, mask)))
    }
}

pub enum Raced<T> {
    AllFalse(Mask),
    Result((T, Mask)),
}

impl<'rt> Future for MaskFuture<'rt> {
    type Output = VortexResult<Mask>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.poll_unpin(cx)
    }
}
