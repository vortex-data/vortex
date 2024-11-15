use std::pin::Pin;
use std::task::Poll;

use futures_util::Stream;
use pin_project::pin_project;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::stream::ArrayStream;
use crate::ArrayData;

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
    S: Stream<Item = VortexResult<ArrayData>>,
{
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<S> Stream for ArrayStreamAdapter<S>
where
    S: Stream<Item = VortexResult<ArrayData>>,
{
    type Item = VortexResult<ArrayData>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.project().inner.poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
