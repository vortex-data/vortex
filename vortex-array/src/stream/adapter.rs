// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::Poll;

use futures::Stream;
use pin_project_lite::pin_project;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::stream::ArrayStream;

pin_project! {
    /// An adapter for a stream of array chunks to implement an ArrayReader.
    pub struct ArrayStreamAdapter<S> {
        dtype: DType,
        #[pin]
        inner: S,
    }
}

impl<S> ArrayStreamAdapter<S>
where
    S: Stream<Item = VortexResult<ArrayRef>>,
{
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
        let array = futures::ready!(this.inner.poll_next(cx));
        if let Some(Ok(array)) = array.as_ref() {
            debug_assert_eq!(
                array.dtype(),
                this.dtype,
                "ArrayStreamAdapter expected array with type {}, actual {}",
                this.dtype,
                array.dtype(),
            );
        }

        Poll::Ready(array)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
