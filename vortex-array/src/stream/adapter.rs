use std::pin::Pin;
use std::task::Poll;

use futures_util::Stream;
use pin_project::pin_project;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::stream::ArrayStream;

/// An adapter for a stream of array chunks to implement an ArrayReader.
#[pin_project]
pub struct ArrayStreamAdapter<S> {
    dtype: DType,
    #[pin]
    inner: S,
}

impl<S> ArrayStreamAdapter<S> {
    pub fn new(dtype: DType, inner: S) -> Self {
        Self { dtype, inner }
    }
}

impl<S> ArrayStream for ArrayStreamAdapter<S>
where
    S: Stream<Item = VortexResult<ArrayRef>>,
{
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<S> Stream for ArrayStreamAdapter<S>
where
    S: Stream<Item = VortexResult<ArrayRef>>,
{
    type Item = VortexResult<ArrayRef>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let array = futures_util::ready!(this.inner.poll_next(cx));
        if let Some(Ok(array)) = array.as_ref() {
            debug_assert_eq!(array.dtype(), this.dtype);
        }

        Poll::Ready(array)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
