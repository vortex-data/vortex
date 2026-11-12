// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
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
        let fut = Self::new(mask.len(), async move { Ok(mask) });
        // Complete the shared future now: it is ready on its first poll, and once complete,
        // `peek` sees it.
        let _ready = fut.inner.clone().now_or_never();
        fut
    }

    /// Returns the mask if the future has completed, without polling it.
    ///
    /// [`MaskFuture::ready`] and slices of a completed mask have completed from the start. A
    /// reader can check their mask before it requests any data, for example to skip unselected
    /// rows. A failed future returns `None`; awaiting it reports the error.
    pub fn peek(&self) -> Option<&Mask> {
        self.inner.peek().and_then(|result| result.as_ref().ok())
    }

    /// Create a MaskFuture that resolves to a mask with all values set to true.
    pub fn new_true(row_count: usize) -> Self {
        Self::ready(Mask::new_true(row_count))
    }

    /// Create a MaskFuture that resolves to a slice of the original mask.
    pub fn slice(&self, range: Range<usize>) -> Self {
        // Slicing the whole mask is the identity. Cloning shares the existing future instead of
        // allocating another boxed, shared one that would await it only to hand the mask back.
        if range.start == 0 && range.end == self.len {
            return self.clone();
        }

        // A slice of a completed mask is complete too.
        if let Some(mask) = self.peek() {
            return Self::ready(mask.slice(range));
        }

        let inner = self.inner.clone();
        Self::new(range.len(), async move { Ok(inner.await?.slice(range)) })
    }

    pub fn inspect(
        self,
        f: impl FnOnce(&SharedVortexResult<Mask>) + 'static + Send + Sync,
    ) -> Self {
        let len = self.len;

        Self {
            inner: self.inner.inspect(f).boxed().shared(),
            len,
        }
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

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;

    use super::*;

    /// Slicing resolves to the same mask the equivalent [`Mask::slice`] would produce, for both
    /// the full range (which takes the identity fast path) and a sub-range.
    #[test]
    fn slice_resolves_to_sliced_mask() -> VortexResult<()> {
        futures::executor::block_on(async {
            let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false]));
            let fut = MaskFuture::ready(mask.clone());

            let full = fut.slice(0..mask.len());
            assert_eq!(full.len(), mask.len());
            assert_eq!(full.await?, mask);

            let partial = fut.slice(0..mask.len() - 1);
            assert_eq!(partial.len(), mask.len() - 1);
            assert_eq!(partial.await?, Mask::from_iter([true, false, true, true]));
            Ok(())
        })
    }

    #[test]
    fn ready_masks_and_their_slices_can_be_peeked() -> VortexResult<()> {
        let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, false]));
        let fut = MaskFuture::ready(mask.clone());
        assert_eq!(fut.peek(), Some(&mask));
        assert_eq!(MaskFuture::new_true(3).peek(), Some(&Mask::new_true(3)));

        let slice = fut.slice(1..3);
        assert_eq!(slice.peek(), Some(&mask.slice(1..3)));
        // A ready future returns its mask on the first poll.
        assert_eq!(slice.now_or_never().transpose()?, Some(mask.slice(1..3)));
        Ok(())
    }

    #[test]
    fn pending_masks_can_be_peeked_once_complete() -> VortexResult<()> {
        let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true]));
        let pending = MaskFuture::new(3, async move { Ok(mask) });
        assert!(pending.peek().is_none());
        assert!(pending.slice(0..2).peek().is_none());

        let mask = futures::executor::block_on(pending.clone())?;
        assert_eq!(pending.peek(), Some(&mask));
        assert_eq!(pending.slice(0..2).peek(), Some(&mask.slice(0..2)));
        Ok(())
    }
}
