// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`crate::LayoutData`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use async_trait::async_trait;
use futures::{Stream, TryStreamExt};
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};
use vortex_array::stream::ArrayStream;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::LayoutRef;
use crate::segments::SequenceWriter;
use crate::sequence::{SequenceId, SequencePointer};

pub trait ArrayStreamSequentialExt: ArrayStream {
    /// Convert an [`ArrayStream`] into a [`SequentialStream`].
    fn sequential(self, mut sequence_ptr: SequencePointer) -> SendableSequentialStream
    where
        Self: Sized + Send + 'static,
    {
        SequentialStreamAdapter::new(
            self.dtype().clone(),
            self.map_ok(move |chunk| (sequence_ptr.advance(), chunk)),
        )
        .sendable()
    }
}

impl<S: ArrayStream> ArrayStreamSequentialExt for S {}

pub trait SequentialStream: Stream<Item = VortexResult<(SequenceId, ArrayRef)>> {
    fn dtype(&self) -> &DType;
}

pub type SendableSequentialStream = Pin<Box<dyn SequentialStream + Send>>;

impl SequentialStream for SendableSequentialStream {
    fn dtype(&self) -> &DType {
        (**self).dtype()
    }
}

#[async_trait]
pub trait LayoutStrategy: 'static + Send + Sync {
    /// Write a stream of arrays to the layout.
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: &SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef>;
}

pub trait SequentialStreamExt: SequentialStream {
    // not named boxed to prevent clashing with StreamExt
    fn sendable(self) -> SendableSequentialStream
    where
        Self: Sized + Send + 'static,
    {
        Box::pin(self)
    }
}

impl<S: SequentialStream> SequentialStreamExt for S {}

pin_project! {
    pub struct SequentialStreamAdapter<S> {
        dtype: DType,
        #[pin]
        inner: S,
    }
}

impl<S> SequentialStreamAdapter<S> {
    pub fn new(dtype: DType, inner: S) -> Self {
        Self { dtype, inner }
    }
}

impl<S> SequentialStream for SequentialStreamAdapter<S>
where
    S: Stream<Item = VortexResult<(SequenceId, ArrayRef)>>,
{
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl<S> Stream for SequentialStreamAdapter<S>
where
    S: Stream<Item = VortexResult<(SequenceId, ArrayRef)>>,
{
    type Item = VortexResult<(SequenceId, ArrayRef)>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        let array = futures::ready!(this.inner.poll_next(cx));
        if let Some(Ok((_, array))) = array.as_ref() {
            assert_eq!(
                array.dtype(),
                this.dtype,
                "Sequential stream of {} got chunk of {}.",
                array.dtype(),
                this.dtype
            );
        }

        Poll::Ready(array)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
