// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::StreamExt;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

pub(crate) fn limit_array_stream<S>(
    stream: S,
    limit: Option<u64>,
) -> BoxStream<'static, VortexResult<ArrayRef>>
where
    S: Stream<Item = VortexResult<ArrayRef>> + Send + 'static,
{
    match limit {
        Some(limit) => RowLimitedStream::new(stream.boxed(), limit).boxed(),
        None => stream.boxed(),
    }
}

struct RowLimitedStream {
    inner: BoxStream<'static, VortexResult<ArrayRef>>,
    remaining: u64,
}

impl RowLimitedStream {
    fn new(inner: BoxStream<'static, VortexResult<ArrayRef>>, remaining: u64) -> Self {
        Self { inner, remaining }
    }

    fn abort_pending(&mut self) {
        let inner = std::mem::replace(&mut self.inner, stream::empty().boxed());
        drop(inner);
    }
}

impl Stream for RowLimitedStream {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.remaining == 0 {
            return Poll::Ready(None);
        }

        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let chunk_len = chunk.len() as u64;
                if chunk_len <= self.remaining {
                    self.remaining -= chunk_len;
                    if self.remaining == 0 {
                        self.abort_pending();
                    }
                    Poll::Ready(Some(Ok(chunk)))
                } else {
                    let limit = match usize::try_from(self.remaining) {
                        Ok(limit) => limit,
                        Err(_) => unreachable!("remaining rows cannot exceed the current chunk"),
                    };
                    self.remaining = 0;
                    self.abort_pending();
                    Poll::Ready(Some(chunk.slice(0..limit)))
                }
            }
            other => other,
        }
    }
}
