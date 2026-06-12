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
use futures::stream::BoxStream;

/// Utility to end a stream early if its backing [`PartitionedFile`] can be pruned away by an updated dynamic expression.
///
/// [`PartitionedFile`]: datafusion_datasource::PartitionedFile
pub(crate) struct PrunableStream {
    file_pruner: FilePruner,
    stream: BoxStream<'static, DFResult<RecordBatch>>,
}

impl PrunableStream {
    pub fn new(file_pruner: FilePruner, stream: BoxStream<'static, DFResult<RecordBatch>>) -> Self {
        Self {
            file_pruner,
            stream,
        }
    }
}

impl Stream for PrunableStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.as_mut().file_pruner.should_prune()? {
            Poll::Ready(None)
        } else {
            self.stream.poll_next_unpin(cx)
        }
    }
}
