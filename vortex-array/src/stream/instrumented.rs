use std::{
    pin::Pin,
    task::{Context, Poll},
};

use crate::dtype::DType;
use crate::stream::ArrayStream;
use futures::Stream;
use pin_project_lite::pin_project;
use tracing::Span;
use tracing_futures::{Instrument, Instrumented};

pin_project! {
    pub struct InstrumentedArrayStream<S> {
        #[pin]
        stream: Instrumented<S>,
    }
}

impl<S: ArrayStream> ArrayStream for InstrumentedArrayStream<S> {
    fn dtype(&self) -> &DType {
        self.stream.inner().dtype()
    }
}

impl<S: Stream> Stream for InstrumentedArrayStream<S> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().stream.poll_next(cx)
    }
}

pub fn instrument_array_stream<S: ArrayStream>(
    stream: S,
    span: Span,
) -> InstrumentedArrayStream<S> {
    let stream = stream.instrument(span);
    InstrumentedArrayStream { stream }
}
