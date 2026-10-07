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
    upper_bound: Option<Mask>,
    upper_bound_is_exact: bool,
    partial_reads_allowed: bool,
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
            upper_bound: None,
            upper_bound_is_exact: false,
            partial_reads_allowed: false,
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
        let upper_bound = mask.clone();
        let mut future = Self::new(mask.len(), async move { Ok(mask) });
        future.upper_bound = Some(upper_bound);
        future.upper_bound_is_exact = true;
        future.partial_reads_allowed = true;
        future
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

        let inner = self.inner.clone();
        let upper_bound = self
            .upper_bound
            .as_ref()
            .map(|upper_bound| upper_bound.slice(range.clone()));
        let mut sliced = Self::new(range.len(), async move { Ok(inner.await?.slice(range)) });
        sliced.upper_bound = upper_bound;
        sliced.upper_bound_is_exact = self.upper_bound_is_exact;
        sliced.partial_reads_allowed = self.partial_reads_allowed;
        sliced
    }

    /// Attach a conservative upper bound for the mask resolved by this future.
    ///
    /// Readers can use this to register I/O eagerly without waiting for filter evaluation. The
    /// resolved mask must not contain a true row that is false in `upper_bound`.
    pub fn with_upper_bound(mut self, upper_bound: Mask) -> Self {
        assert_eq!(
            upper_bound.len(),
            self.len,
            "MaskFuture upper bound length mismatch"
        );
        self.upper_bound = Some(upper_bound);
        self.upper_bound_is_exact = false;
        self
    }

    /// Return the conservative upper bound for this future, when one is known.
    pub fn upper_bound(&self) -> Option<&Mask> {
        self.upper_bound.as_ref()
    }

    /// Return whether the upper bound is the exact mask returned by this future.
    pub fn upper_bound_is_exact(&self) -> bool {
        self.upper_bound_is_exact
    }

    /// Permit readers to satisfy this selection using partial segment reads.
    pub fn with_partial_reads(mut self) -> Self {
        self.partial_reads_allowed = true;
        self
    }

    /// Prevent readers from turning this mask into partial segment reads.
    pub fn without_partial_reads(mut self) -> Self {
        self.partial_reads_allowed = false;
        self
    }

    /// Return whether readers may satisfy this selection using partial segment reads.
    pub fn partial_reads_allowed(&self) -> bool {
        self.partial_reads_allowed
    }

    pub fn inspect(
        self,
        f: impl FnOnce(&SharedVortexResult<Mask>) + 'static + Send + Sync,
    ) -> Self {
        let len = self.len;

        Self {
            inner: self.inner.inspect(f).boxed().shared(),
            len,
            upper_bound: self.upper_bound,
            upper_bound_is_exact: self.upper_bound_is_exact,
            partial_reads_allowed: self.partial_reads_allowed,
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
            assert_eq!(partial.upper_bound(), Some(&mask.slice(0..mask.len() - 1)));
            assert_eq!(partial.await?, mask.slice(0..mask.len() - 1));
            Ok(())
        })
    }

    #[test]
    fn new_future_has_no_upper_bound_until_attached() {
        let future = MaskFuture::new(3, async { Ok(Mask::new_false(3)) });
        assert!(future.upper_bound().is_none());

        let upper_bound = Mask::from_indices(3, [0, 2]);
        let future = future.with_upper_bound(upper_bound.clone());
        assert_eq!(future.upper_bound(), Some(&upper_bound));
        assert!(!future.upper_bound_is_exact());
    }

    #[test]
    fn ready_future_has_an_exact_upper_bound() {
        let future = MaskFuture::ready(Mask::from_indices(3, [0, 2]));
        assert!(future.upper_bound_is_exact());
    }
}
