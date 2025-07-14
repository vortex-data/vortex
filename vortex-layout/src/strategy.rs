// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`crate::LayoutData`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::SendableLayoutFuture;
use crate::segments::SequenceWriter;
use crate::sequence::SequenceId;

pub trait SequentialStream: Stream<Item = VortexResult<(SequenceId, ArrayRef)>> {
    fn dtype(&self) -> &DType;
}

pub type SendableSequentialStream = Pin<Box<dyn SequentialStream + Send>>;

impl SequentialStream for SendableSequentialStream {
    fn dtype(&self) -> &DType {
        (**self).dtype()
    }
}

pub trait LayoutStrategy: 'static + Send + Sync {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> SendableLayoutFuture;
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
