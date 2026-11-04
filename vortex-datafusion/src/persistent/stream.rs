// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::time::Instant;

use datafusion_common::Result as DFResult;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
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
    observer: Option<FilterObserver>,
}

impl PrunableStream {
    pub fn new(file_pruner: FilePruner, stream: BoxStream<'static, DFResult<RecordBatch>>) -> Self {
        Self {
            file_pruner,
            stream: Some(stream),
            observer: None,
        }
    }

    pub(crate) fn observe_filter(
        mut self,
        predicate: PhysicalExprRef,
        path: String,
        started: Instant,
    ) -> DFResult<Self> {
        if tracing::enabled!(tracing::Level::DEBUG) {
            let mut filters = Vec::new();
            predicate.apply(|expr| {
                if expr.is::<DynamicFilterPhysicalExpr>() {
                    filters.push((Arc::clone(expr), 0));
                }
                Ok(TreeNodeRecursion::Continue)
            })?;
            self.observer = Some(FilterObserver {
                filters,
                path,
                started,
            });
        }
        Ok(self)
    }
}

struct FilterObserver {
    filters: Vec<(PhysicalExprRef, u64)>,
    path: String,
    started: Instant,
}

impl FilterObserver {
    fn observe(&mut self) -> bool {
        let mut changed = false;
        for (filter, previous) in &mut self.filters {
            let generation = filter.snapshot_generation();
            if generation != *previous {
                changed = true;
                tracing::debug!(path = %self.path, elapsed_us = self.started.elapsed().as_micros(), filter_id = ?filter.expression_id(), generation, previous_generation = *previous, predicate = %filter, "dynamic filter observed");
                *previous = generation;
            }
        }
        changed
    }
}

impl Stream for PrunableStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let changed = self.observer.as_mut().is_some_and(FilterObserver::observe);
        let started = self.observer.as_ref().map(|_| Instant::now());
        let pruned = self.as_mut().file_pruner.should_prune()?;
        if let Some(started) = started {
            tracing::debug!(
                elapsed_us = started.elapsed().as_micros(),
                changed,
                pruned,
                "dynamic file pruning checked"
            );
        }
        if pruned {
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
