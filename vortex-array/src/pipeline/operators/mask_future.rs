// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::ops::{BitAnd, Range};
use std::sync::Arc;

use futures_util::future::{BoxFuture, Either, SelectAll, Shared, select};
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

    /// Race this mask with another future, returning `None` early if the result of the mask is
    /// all false, or else returns a pair of the mask and the result of the other future.
    pub fn race<T>(
        self,
        other: impl Future<Output = VortexResult<T>> + Unpin,
    ) -> impl Future<Output = VortexResult<Option<(Mask, T)>>> {
        let len = self.len();
        async move {
            match select(self, other).await {
                Either::Left((mask, other_fut)) => {
                    let mask = mask?;
                    if mask.all_false() {
                        return Ok(None);
                    }
                    let other = other_fut.await?;
                    Ok(Some((mask, other)))
                }
                Either::Right((other, mask_fut)) => {
                    let other = other?;
                    let mask = mask_fut.await?;
                    if mask.all_false() {
                        return Ok(None);
                    }
                    Ok(Some((mask, other)))
                }
            }
        }
    }

    /// Create a MaskFuture that resolves to the intersection of multiple MaskFutures.
    ///
    /// If the accumulated intersection is all false at any point, the future will resolve early,
    /// dropping any remaining futures.
    pub fn intersect(init: MaskFuture, mut futures: Vec<MaskFuture>) -> Self {
        if futures.is_empty() {
            return init;
        }
        if futures.iter().any(|m| m.len() != init.len()) {
            vortex_panic!("MaskFuture::intersect called with futures of different lengths");
        }
        let len = init.len();
        // Include the initial future in the list also
        futures.push(init);

        MaskFuture::new(len, async move {
            // Now we race all futures, intersect their results as they come in, and return early
            // if the intersection is all false.
            let mut futures = SelectAll::from_iter(futures);
            let mut acc = Mask::new_true(len);

            loop {
                let (mask, _index, remaining) = futures.await;
                acc = acc.bitand(&mask?);

                if acc.all_false() {
                    return Ok(acc);
                }
                if remaining.is_empty() {
                    return Ok(acc);
                }

                futures = SelectAll::from_iter(remaining);
            }
        })
    }

    /// Runs a set of futures concurrently, but passes the result mask into each future from left
    /// to right.
    ///
    /// In other words, the individual futures will each make progress concurrently until they
    /// await their input mask, at which point they will block waiting for the previous future to
    /// complete. The overall result is the intersection of all the futures.
    pub fn fold_intersect<F>(init: MaskFuture, fns: impl Iterator<Item = F>) -> Self
    where
        F: FnOnce(MaskFuture) -> MaskFuture,
    {
        let mut current = init.clone();
        let mut futures = Vec::new();
        for f in fns {
            let fut = f(current.clone());
            assert_eq!(
                fut.len(),
                init.len(),
                "MaskFuture::fold called with futures of different lengths"
            );
            futures.push(fut.clone());
            current = fut;
        }
        MaskFuture::intersect(init, futures)
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
