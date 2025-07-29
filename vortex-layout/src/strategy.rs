// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`crate::LayoutData`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};
use vortex_array::stream::ArrayStream;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::LayoutRef;
use crate::segments::SegmentSink;
use crate::sequence::{SequenceId, SequencePointer};

#[async_trait(?Send)]
pub trait LayoutStrategy: 'static + Send + Sync {
    /// Write a stream of arrays to the layout.
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        segment_sink: &dyn SegmentSink,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef>;
}

pub trait SequentialStream: Stream<Item = VortexResult<(SequenceId, ArrayRef)>> {
    /// Returns the data type of the arrays in the stream.
    fn dtype(&self) -> &DType;
    /// Returns a sequence pointer that is guaranteed to be at the end of the stream.
    fn end_of_stream(&mut self) -> SequencePointer;
}

pub type SendableSequentialStream = Pin<Box<dyn SequentialStream + Send>>;

impl SequentialStream for SendableSequentialStream {
    fn dtype(&self) -> &DType {
        (**self).dtype()
    }

    fn end_of_stream(&mut self) -> SequencePointer {
        self.as_mut().end_of_stream()
    }
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

pub trait SequentialArrayStreamExt: ArrayStream {
    /// Converts the stream to a [`SendableSequentialStream`].
    fn sequenced(self, mut pointer: SequencePointer) -> SendableSequentialStream
    where
        Self: Sized + Send + 'static,
    {
        let mut start_of_stream = pointer.advance().descend();
        let end_of_stream = pointer.advance().descend();
        Box::pin(SequentialStreamAdapter::new(
            self.dtype().clone(),
            StreamExt::map(self, move |item| {
                item.map(|array| (start_of_stream.advance(), array))
            }),
            end_of_stream,
        ))
    }
}

impl<S: ArrayStream> SequentialArrayStreamExt for S {}

pin_project! {
    pub struct SequentialStreamAdapter<S> {
        dtype: DType,
        #[pin]
        inner: S,
        end_of_stream: SequencePointer,
    }
}

impl<S> SequentialStreamAdapter<S> {
    pub fn new(dtype: DType, inner: S, end_of_stream: SequencePointer) -> Self {
        Self {
            dtype,
            inner,
            end_of_stream,
        }
    }
}

impl<S> SequentialStream for SequentialStreamAdapter<S>
where
    S: Stream<Item = VortexResult<(SequenceId, ArrayRef)>>,
{
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn end_of_stream(&mut self) -> SequencePointer {
        self.end_of_stream.advance().descend()
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
