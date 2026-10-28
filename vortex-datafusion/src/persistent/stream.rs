// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use datafusion_common::Result as DFResult;
use datafusion_common::arrow::array::RecordBatch;
use futures::Stream;
use futures::StreamExt;
use futures::stream::BoxStream;

use super::pruning::Pruner;

/// Utility to end a stream early if its backing [`PartitionedFile`] can be pruned away by an updated dynamic expression.
///
/// [`PartitionedFile`]: datafusion_datasource::PartitionedFile
pub(crate) struct PrunableStream {
    file_pruner: Pruner,
    stream: Option<BoxStream<'static, DFResult<RecordBatch>>>,
}

impl PrunableStream {
    pub fn new(file_pruner: Pruner, stream: BoxStream<'static, DFResult<RecordBatch>>) -> Self {
        Self {
            file_pruner,
            stream: Some(stream),
        }
    }
}

impl Stream for PrunableStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.as_mut().file_pruner.should_prune()? {
            self.stream.take();
            Poll::Ready(None)
        } else {
            match self.stream.as_mut() {
                Some(stream) => stream.poll_next_unpin(cx),
                None => Poll::Ready(None),
            }
        }
    }
}
