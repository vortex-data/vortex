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
    stream: Option<BoxStream<'static, DFResult<RecordBatch>>>,
    path: String,
    polls: u64,
}

impl PrunableStream {
    pub fn new(
        file_pruner: FilePruner,
        stream: BoxStream<'static, DFResult<RecordBatch>>,
        path: String,
    ) -> Self {
        Self {
            file_pruner,
            stream: Some(stream),
            path,
            polls: 0,
        }
    }
}

/// Temporary audit wrapper: counts what a file stream emitted and reports it when dropped,
/// which also covers streams cancelled early by a limit or a pruned file.
pub(crate) struct AuditStream {
    stream: BoxStream<'static, DFResult<RecordBatch>>,
    path: String,
    partition: usize,
    batches: u64,
    rows: u64,
    bytes: u64,
    finished: bool,
    started: std::time::Instant,
    first_batch_us: Option<u64>,
}

impl AuditStream {
    pub fn new(
        stream: BoxStream<'static, DFResult<RecordBatch>>,
        path: String,
        partition: usize,
    ) -> Self {
        Self {
            stream,
            path,
            partition,
            batches: 0,
            rows: 0,
            bytes: 0,
            finished: false,
            started: std::time::Instant::now(),
            first_batch_us: None,
        }
    }
}

impl Stream for AuditStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.stream.poll_next_unpin(cx);
        match &poll {
            Poll::Ready(Some(Ok(batch))) => {
                self.batches += 1;
                self.rows += batch.num_rows() as u64;
                self.bytes += batch.get_array_memory_size() as u64;
                if self.first_batch_us.is_none() {
                    self.first_batch_us = Some(self.started.elapsed().as_micros() as u64);
                }
            }
            Poll::Ready(None) => self.finished = true,
            _ => {}
        }
        poll
    }
}

impl Drop for AuditStream {
    fn drop(&mut self) {
        tracing::debug!(
            target: "vortex_audit::output",
            path = %self.path,
            partition = self.partition,
            batches = self.batches,
            rows = self.rows,
            arrow_bytes = self.bytes,
            finished = self.finished,
            first_batch_us = self.first_batch_us,
            elapsed_us = self.started.elapsed().as_micros() as u64,
            "file stream dropped"
        );
    }
}

impl Stream for PrunableStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.polls += 1;
        if self.stream.is_some() && self.as_mut().file_pruner.should_prune()? {
            tracing::debug!(
                target: "vortex_audit::output",
                path = %self.path,
                polls = self.polls,
                "file stream pruned mid-scan by dynamic filter"
            );
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
