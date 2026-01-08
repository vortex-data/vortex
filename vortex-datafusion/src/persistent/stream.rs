// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use datafusion_common::Result as DFResult;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_pruning::FilePruner;
use futures::Stream;
use futures::StreamExt;
use futures::ready;
use futures::stream::BoxStream;

/// Utility to end a stream early if we can stop processing it by a limit or a dynamic expression in the [`FilePruner`].
pub(crate) struct EarlyTerminatingStream {
    file_pruner: Option<FilePruner>,
    limit: Option<usize>,
    stream: BoxStream<'static, DFResult<RecordBatch>>,
}

impl EarlyTerminatingStream {
    pub fn new(
        file_pruner: Option<FilePruner>,
        limit: Option<usize>,
        stream: BoxStream<'static, DFResult<RecordBatch>>,
    ) -> Self {
        Self {
            file_pruner,
            limit,
            stream,
        }
    }
}

impl Stream for EarlyTerminatingStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.limit.is_some_and(|l| l == 0)
            || self
                .file_pruner
                .as_mut()
                .map(|fp| fp.should_prune())
                .transpose()?
                .unwrap_or_default()
        {
            Poll::Ready(None)
        } else {
            match ready!(self.stream.poll_next_unpin(cx)) {
                Some(rb) => {
                    let rb = rb?;

                    if let Some(limit) = self.limit.as_mut() {
                        *limit = limit.saturating_sub(rb.num_rows());
                    }

                    Poll::Ready(Some(Ok(rb)))
                }
                None => Poll::Ready(None),
            }
        }
    }
}
