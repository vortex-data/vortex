// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`crate::LayoutData`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arcref::ArcRef;
use async_trait::async_trait;
use futures::Stream;
use pin_project_lite::pin_project;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::layouts::buffered::BufferedStrategy;
use crate::segments::SequenceWriter;
use crate::sequence::SequenceId;
use crate::LayoutRef;

pub trait SequentialStream: Stream<Item = VortexResult<(SequenceId, ArrayRef)>> {
    fn dtype(&self) -> &DType;
}

pub type SendableSequentialStream = Pin<Box<dyn SequentialStream + Send>>;

impl SequentialStream for SendableSequentialStream {
    fn dtype(&self) -> &DType {
        (**self).dtype()
    }
}

/// A LayoutStrategy is how a stream of data chunks becomes
/// a Layout.
///
/// The strategy accepts a stream of chunks, and yields a new
/// layout
// Tag for Python docs:
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

// Helper implementation for async functions that accept the parameters of the write_stream method.
#[async_trait]
impl<F, Fut> LayoutStrategy for F
where
    F: (Fn(&ArrayContext, SequenceWriter, SendableSequentialStream) -> Fut) + Send + Sync + 'static,
    Fut: Future<Output = VortexResult<LayoutRef>> + Send + Sync + 'static,
{
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        self(ctx, sequence_writer, stream).await
    }
}

pub trait LayoutStrategyExt: LayoutStrategy {
    /// Wrap a layout with a buffer. The input chunk stream will be reorganized into chunks of
    /// size `bytes`.
    fn buffered(self, bytes: u64) -> impl LayoutStrategy
    where
        Self: Sized,
    {
        BufferedStrategy::new(ArcRef::new_arc(Arc::new(self)), bytes)
    }
}

impl<T: LayoutStrategy> LayoutStrategyExt for T {}

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
