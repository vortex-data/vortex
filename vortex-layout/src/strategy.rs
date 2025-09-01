// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::Stream;
use pin_project_lite::pin_project;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::LayoutRef;
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

// [layout writer]
#[async_trait]
pub trait LayoutStrategy: 'static + Send + Sync {
    /// Asynchronously process an ordered stream of array chunks, emitting them into a sink and
    /// returning the [`Layout`][crate::Layout] instance that can be parsed to retrieve the data
    /// from rest.
    ///
    /// This trait uses the `#[async_trait]` attribute to denote that trait objects of this type
    /// can be `Box`ed or `Arc`ed and shared around. Commonly, these strategies are composed to
    /// form a pipeline of operations, each of which modifies the chunk stream in some way before
    /// passing the data on to a downstream writer.
    ///
    /// # Blocking operations
    ///
    /// This is an async trait method, which will return a `BoxFuture` that you can await from
    /// any runtime. Implementations should avoid directly performing blocking work within the
    /// `write_stream`, and should instead spawn it onto an appropriate runtime or threadpool
    /// dedicated to such work.
    ///
    /// Such operations are common, and include things like compression and parsing large blobs
    /// of data, or serializing very large messages to flatbuffers.
    ///
    /// Consider accepting a [`TaskExecutor`][crate::TaskExecutor] as an input to your strategy
    /// to support spawning this work in the background.
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef>;
}
// [layout writer]

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
