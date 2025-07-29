// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`crate::LayoutData`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::task::{Context, Poll};
use vortex_array::stream::{ArrayStream, ArrayStreamExt, SendableArrayStream};
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
        stream: SequentialArrayStream,
    ) -> VortexResult<LayoutRef>;
}

/// A wrapper around an [`ArrayStream`] that emits arrays tagged with a sequence ID.
pub struct SequentialArrayStream {
    inner: SendableArrayStream,
    pointer: SequencePointer,
}

impl SequentialArrayStream {
    pub fn new(inner: SendableArrayStream, pointer: SequencePointer) -> Self {
        Self { inner, pointer }
    }

    pub fn sequence_id(&mut self) -> SequenceId {
        self.pointer.advance()
    }

    /// Returns a new stream and a [`SequencePointer`], where the sequence pointer is guaranteed
    /// to be ordered after all elements of the returned stream.
    pub fn split_off(mut self) -> (Self, SequencePointer) {
        // We take our current sequence ID and descend.
        let mut ptr = self.sequence_id().descend();
        // We advance twice to get two sibling pointers.
        let first = ptr.advance();
        let second = ptr.advance();

        // Now we re-wrap our stream with the descendents of the first pointer, ensuring all
        // elements of `second` are after all elements of `first`.
        self.pointer = first.descend();

        (self, second.descend())
    }

    /// Map the stream including a [`SequenceId`] with each element.
    pub fn map(mut self) -> impl Stream<Item = VortexResult<(SequenceId, ArrayRef)>> {
        self.inner.map(move |array| {
            let sequence_id = self.sequence_id();
            Ok((sequence_id, array))
        })
    }
}

impl ArrayStream for SequentialArrayStream {
    fn dtype(&self) -> &DType {
        self.inner.dtype()
    }
}

impl Stream for SequentialArrayStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.poll_next_unpin(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Extension trait for [`ArrayStream`] to convert it into a sequential stream.
pub trait ArrayStreamSequentialExt: ArrayStream {
    /// Convert an [`ArrayStream`] into a [`SequentialStream`].
    fn sequenced(self, sequence_ptr: SequencePointer) -> SequentialArrayStream
    where
        Self: Sized + Send + 'static,
    {
        SequentialArrayStream::new(ArrayStreamExt::boxed(self), sequence_ptr)
    }
}

impl<S: ArrayStream> ArrayStreamSequentialExt for S {}
