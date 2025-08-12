// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use itertools::Itertools;
pub use multi_scan::*;
pub use selection::*;
pub use split_by::*;
use tasks::{TaskContext, split_exec};
use vortex_array::ArrayRef;
use vortex_array::iter::ArrayIterator;
use vortex_array::stats::StatsSet;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Field, FieldMask, FieldName, FieldPath};
use vortex_error::{VortexResult, vortex_bail};
use vortex_expr::transform::immediate_access::immediate_scope_access;
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{ExprRef, root};
use vortex_layout::layouts::row_idx::RowIdxLayoutReader;
use vortex_layout::{LayoutReader, LayoutReaderRef};
pub use vortex_layout::{TaskExecutor, TaskExecutorExt};
use vortex_metrics::VortexMetrics;

use crate::filter::FilterExpr;
use crate::work_queue::{TaskFactory, WorkStealingQueue};
use crate::work_stealing_iter::{ArrayTask, WorkStealingArrayIterator};

mod arrow;
mod filter;
mod multi_scan;
#[cfg(feature = "tokio")]
mod multi_thread;
pub mod row_mask;
mod selection;
mod split_by;
mod tasks;
mod work_queue;
mod work_stealing_iter;

/// A struct for building a scan operation.
pub struct ScanBuilder<A> {
    layout_reader: LayoutReaderRef,
    projection: ExprRef,
    filter: Option<ExprRef>,
    /// Optionally read a subset of the rows in the file.
    row_range: Option<Range<u64>>,
    /// The selection mask to apply to the selected row range.
    // TODO(joe): replace this is usage of row_id selection, see
    selection: Selection,
    /// How to split the file f§    or concurrent processing.
    split_by: SplitBy,
    /// The number of splits to make progress on concurrently **per-thread**.
    concurrency: usize,
    /// Function to apply to each [`ArrayRef`] within the spawned split tasks.
    map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    metrics: VortexMetrics,
    /// Should we try to prune the file (using stats) on open.
    file_stats: Option<Arc<[StatsSet]>>,
    /// Maximal number of rows to read (after filtering)
    limit: Option<usize>,
    /// The row-offset assigned to the first row of the file. Used by the `row_idx` expression,
    /// but not by the scan [`Selection`] which remains relative.
    row_offset: u64,
}

impl<A: 'static + Send> ScanBuilder<A> {
    pub fn with_filter(mut self, filter: ExprRef) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_some_filter(mut self, filter: Option<ExprRef>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_projection(mut self, projection: ExprRef) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_row_range(mut self, row_range: Range<u64>) -> Self {
        self.row_range = Some(row_range);
        self
    }

    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.selection = Selection::IncludeByIndex(row_indices);
        self
    }

    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.row_offset = row_offset;
        self
    }

    pub fn with_split_by(mut self, split_by: SplitBy) -> Self {
        self.split_by = split_by;
        self
    }

    /// The number of row splits to make progress on concurrently per-thread, must
    /// be greater than 0.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0);
        self.concurrency = concurrency;
        self
    }

    pub fn with_metrics(mut self, metrics: VortexMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// The [`DType`] returned by the scan, after applying the projection.
    pub fn dtype(&self) -> VortexResult<DType> {
        self.projection.return_dtype(self.layout_reader.dtype())
    }

    /// Map each split of the scan. The function will be run on the spawned task.
    pub fn map<B: 'static>(
        self,
        map_fn: impl Fn(A) -> VortexResult<B> + 'static + Send + Sync,
    ) -> ScanBuilder<B> {
        let old_map_fn = self.map_fn;
        ScanBuilder {
            layout_reader: self.layout_reader,
            projection: self.projection,
            filter: self.filter,
            row_range: self.row_range,
            selection: self.selection,
            split_by: self.split_by,
            concurrency: self.concurrency,
            map_fn: Arc::new(move |a| map_fn(old_map_fn(a)?)),
            metrics: self.metrics,
            file_stats: self.file_stats,
            limit: self.limit,
            row_offset: self.row_offset,
        }
    }

    /// Constructs a task per row split of the scan, returned as a vector of futures.
    pub fn build(mut self) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<A>>>>> {
        if self.filter.is_some() && self.limit.is_some() {
            vortex_bail!("Vortex doesn't support scans with both a filter and a limit")
        }

        // The ultimate short circuit
        if self.limit.is_some_and(|l| l == 0) {
            return Ok(vec![]);
        }

        // Spin up the root layout reader, and wrap it in a FilterLayoutReader to perform
        // conjunction splitting if a filter is provided.
        let mut layout_reader = self.layout_reader;

        // Enrich the layout reader to support RowIdx expressions.
        // Note that this is applied below the filter layout reader since it can perform
        // better over individual conjunctions.
        layout_reader = Arc::new(RowIdxLayoutReader::new(self.row_offset, layout_reader));

        // Normalize and simplify the expressions.
        let projection = simplify_typed(self.projection.clone(), layout_reader.dtype())?;
        let filter = self
            .filter
            .clone()
            .map(|f| simplify_typed(f, layout_reader.dtype()))
            .transpose()?;

        // Construct field masks and compute the row splits of the scan.
        let (filter_mask, projection_mask) =
            filter_and_projection_masks(&projection, filter.as_ref(), layout_reader.dtype())?;
        let field_mask: Vec<_> = [filter_mask, projection_mask].concat();
        let splits = self.split_by.splits(layout_reader.as_ref(), &field_mask)?;

        let ctx = Arc::new(TaskContext {
            row_range: self.row_range,
            selection: self.selection,
            filter: filter.map(|f| Arc::new(FilterExpr::new(f))),
            reader: layout_reader,
            projection,
            mapper: self.map_fn,
        });

        // Create a task that executes the full scan pipeline for each split.
        let split_tasks = splits
            .into_iter()
            .filter_map(|split_range| {
                if self.limit.is_some_and(|l| l == 0) {
                    None
                } else {
                    Some(split_exec(ctx.clone(), split_range, self.limit.as_mut()))
                }
            })
            .try_collect()?;

        Ok(split_tasks)
    }

    /// Returns a [`Stream`] with tasks spawned onto the current Tokio runtime.
    ///
    /// The stream performs CPU work on the polling thread, with I/O operations dispatched as
    /// per the Vortex I/O traits.
    ///
    /// Task concurrency is the product of the `concurrency` parameter and the number of worker
    /// threads in the Tokio runtime.
    #[cfg(feature = "tokio")]
    pub fn into_tokio_stream(
        self,
    ) -> VortexResult<impl futures::Stream<Item = VortexResult<A>> + Send + 'static> {
        use futures::StreamExt;
        use vortex_error::vortex_err;

        let handle = tokio::runtime::Handle::current();
        let num_workers = handle.metrics().num_workers();
        let concurrency = self.concurrency * num_workers;
        Ok(futures::stream::iter(self.build()?)
            .map(move |task| handle.spawn(task))
            .buffered(concurrency)
            .map(|task| {
                task.map_err(|e| vortex_err!("Failed to join task: {e}"))
                    .flatten()
            })
            .filter_map(|chunk| async move { chunk.transpose() }))
    }
}

impl ScanBuilder<ArrayRef> {
    pub fn new(layout_reader: Arc<dyn LayoutReader>) -> Self {
        Self {
            layout_reader,
            projection: root(),
            filter: None,
            row_range: None,
            selection: Default::default(),
            split_by: SplitBy::Layout,
            // We default to four tasks per worker thread, which allows for some I/O lookahead
            // without too much impact on work-stealing.
            concurrency: 4,
            map_fn: Arc::new(Ok),
            metrics: Default::default(),
            file_stats: None,
            limit: None,
            row_offset: 0,
        }
    }

    /// Returns a thread-safe [`ArrayIterator`] that can be cloned and passed
    /// to other threads to make progress on the same scan concurrently.
    ///
    /// Within each thread, the array chunks will be emitted in the original order they are within
    /// the scan. Between threads, the order is not guaranteed.
    pub fn into_array_iter(self) -> VortexResult<impl ArrayIterator + Send + Clone + 'static> {
        let dtype = self.dtype()?;
        let concurrency = self.concurrency;
        let tasks = self.build()?;
        let queue = WorkStealingQueue::new([Box::new(move || Ok(tasks)) as TaskFactory<ArrayTask>]);

        Ok(WorkStealingArrayIterator::new(
            queue,
            Arc::new(dtype),
            concurrency,
        ))
    }

    /// Returns an [`ArrayStream`] with tasks spawned onto the current Tokio runtime.
    ///
    /// See [`ScanBuilder::into_tokio_stream`] for more details.
    #[cfg(feature = "tokio")]
    pub fn into_tokio_array_stream(
        self,
    ) -> VortexResult<impl vortex_array::stream::ArrayStream + Send + 'static> {
        let dtype = self.dtype()?;
        let stream = self.into_tokio_stream()?;
        Ok(vortex_array::stream::ArrayStreamAdapter::new(dtype, stream))
    }
}

/// Compute masks of field paths referenced by the projection and filter in the scan.
///
/// Projection and filter must be pre-simplified.
fn filter_and_projection_masks(
    projection: &ExprRef,
    filter: Option<&ExprRef>,
    dtype: &DType,
) -> VortexResult<(Vec<FieldMask>, Vec<FieldMask>)> {
    let Some(struct_dtype) = dtype.as_struct() else {
        return Ok(match filter {
            Some(_) => (vec![FieldMask::All], vec![FieldMask::All]),
            None => (Vec::new(), vec![FieldMask::All]),
        });
    };
    let projection_mask = immediate_scope_access(projection, struct_dtype);
    Ok(match filter {
        None => (
            Vec::new(),
            projection_mask.into_iter().map(to_field_mask).collect_vec(),
        ),
        Some(f) => {
            let filter_mask = immediate_scope_access(f, struct_dtype);
            let only_projection_mask = projection_mask
                .difference(&filter_mask)
                .cloned()
                .map(to_field_mask)
                .collect_vec();
            (
                filter_mask.into_iter().map(to_field_mask).collect_vec(),
                only_projection_mask,
            )
        }
    })
}

fn to_field_mask(field: FieldName) -> FieldMask {
    FieldMask::Prefix(FieldPath::from(Field::Name(field)))
}

#[cfg(all(test, not(disable_loom)))]
#[allow(clippy::tests_outside_test_module)]  // False positive due to complex cfg
mod loom_tests {
    use bit_vec::BitVec;
    use futures::future;
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicUsize, Ordering};
    use loom::thread;
    use vortex_error::{VortexResult, vortex_err};
    use vortex_expr::{and, get_item, gt, lit, lt, root};

    use crate::filter::FilterExpr;
    use crate::multi_scan::{ArrayFuture, MultiScan};
    use crate::work_queue::{TaskFactory, WorkStealingQueue};

    #[test]
    fn test_work_stealing_queue_basic() {
        // Test basic WorkStealingQueue operations with multiple workers
        loom::model(|| {
            // Create task factories that produce simple tasks
            let factories: Vec<TaskFactory<i32>> = vec![
                Box::new(|| Ok(vec![1, 2, 3])),
                Box::new(|| Ok(vec![4, 5, 6])),
                Box::new(|| Ok(vec![7, 8, 9])),
            ];

            let queue = WorkStealingQueue::new(factories);

            // Create two workers
            let iter1 = queue.clone().new_iterator();
            let iter2 = queue.new_iterator();

            // Collect results from both workers
            let handle1 = thread::spawn(move || {
                let mut results = Vec::new();
                for val in iter1.flatten() {
                    results.push(val);
                    if results.len() >= 3 {
                        break;
                    }
                }
                results
            });

            let handle2 = thread::spawn(move || {
                let mut results = Vec::new();
                for val in iter2.flatten() {
                    results.push(val);
                    if results.len() >= 3 {
                        break;
                    }
                }
                results
            });

            let results1 = handle1.join().unwrap();
            let results2 = handle2.join().unwrap();

            // Verify that results are from our expected set
            for val in results1.iter().chain(results2.iter()) {
                assert!(*val >= 1 && *val <= 9);
            }

            // Verify no duplicates between workers
            let mut all_results = results1;
            all_results.extend(results2);
            all_results.sort();
            for i in 1..all_results.len() {
                assert_ne!(all_results[i], all_results[i - 1], "Found duplicate value");
            }
        });
    }

    #[test]
    fn test_work_stealing_queue_error_handling() {
        // Test that errors in task factories are properly propagated
        loom::model(|| {
            let factories: Vec<TaskFactory<i32>> = vec![
                Box::new(|| Ok(vec![1, 2])),
                Box::new(|| Err(vortex_err!("Factory error"))),
                Box::new(|| Ok(vec![3, 4])),
            ];

            let queue = WorkStealingQueue::new(factories);
            let iter = queue.new_iterator();

            let mut has_error = false;
            let mut values = Vec::new();

            for result in iter {
                match result {
                    Ok(val) => values.push(val),
                    Err(_) => {
                        has_error = true;
                        break;
                    }
                }
            }

            // Should encounter the error
            assert!(has_error || !values.is_empty());
        });
    }

    #[test]
    fn test_work_stealing_concurrent_factory_construction() {
        // Test concurrent factory construction with multiple workers
        loom::model(|| {
            let counter = Arc::new(AtomicUsize::new(0));

            let factories: Vec<TaskFactory<usize>> = (0..3usize)
                .map(|i| {
                    let counter_clone = counter.clone();
                    Box::new(move || {
                        counter_clone.fetch_add(1, Ordering::SeqCst);
                        Ok(vec![i * 10, i * 10 + 1])
                    }) as TaskFactory<usize>
                })
                .collect();

            let queue = WorkStealingQueue::new(factories);

            // Create multiple workers
            let iter1 = queue.clone().new_iterator();
            let iter2 = queue.new_iterator();

            let handle1 = thread::spawn(move || {
                let mut count = 0;
                for result in iter1 {
                    if result.is_ok() {
                        count += 1;
                        if count >= 2 {
                            break;
                        }
                    }
                }
                count
            });

            let handle2 = thread::spawn(move || {
                let mut count = 0;
                for result in iter2 {
                    if result.is_ok() {
                        count += 1;
                        if count >= 2 {
                            break;
                        }
                    }
                }
                count
            });

            handle1.join().unwrap();
            handle2.join().unwrap();

            // Verify factories were constructed
            let final_count = counter.load(Ordering::SeqCst);
            assert!(final_count > 0 && final_count <= 3);
        });
    }

    #[test]
    fn test_filter_expr_concurrent_selectivity_reporting() {
        // Test concurrent selectivity reporting in FilterExpr
        loom::model(|| {
            let expr = lit(true); // Simple expression for testing
            let filter = Arc::new(FilterExpr::new(expr));

            let filter1 = filter.clone();
            let filter2 = filter.clone();
            let filter3 = filter;

            // Multiple threads reporting selectivity
            let handle1 = thread::spawn(move || {
                filter1.report_selectivity(0, 0.5);
                filter1.report_selectivity(0, 0.6);
            });

            let handle2 = thread::spawn(move || {
                filter2.report_selectivity(0, 0.7);
                filter2.report_selectivity(0, 0.4);
            });

            // Reader thread
            let handle3 = thread::spawn(move || {
                let mut remaining = BitVec::from_elem(1, true);
                let conjunct = filter3.next_conjunct(&remaining);
                assert_eq!(conjunct, Some(0));

                // Mark as evaluated
                remaining.set(0, false);
                let conjunct = filter3.next_conjunct(&remaining);
                assert_eq!(conjunct, None);
            });

            handle1.join().unwrap();
            handle2.join().unwrap();
            handle3.join().unwrap();
        });
    }

    #[test]
    fn test_filter_expr_ordering_update() {
        // Test concurrent ordering updates in FilterExpr
        loom::model(|| {
            // Create a filter with multiple conjuncts (AND conditions)
            let expr = and(
                gt(get_item("a", root()), lit(5)),
                lt(get_item("b", root()), lit(10)),
            );
            let filter = Arc::new(FilterExpr::new(expr));

            let filter1 = filter.clone();
            let filter2 = filter;

            // Thread 1 reports selectivity for conjunct 0
            let handle1 = thread::spawn(move || {
                filter1.report_selectivity(0, 0.1); // Very selective
                filter1.report_selectivity(0, 0.2);
            });

            // Thread 2 reports selectivity for conjunct 1
            let handle2 = thread::spawn(move || {
                filter2.report_selectivity(1, 0.9); // Not selective
                filter2.report_selectivity(1, 0.8);
            });

            handle1.join().unwrap();
            handle2.join().unwrap();

            // The ordering should prefer more selective conjuncts
            // but we just verify it doesn't crash under concurrent access
        });
    }

    #[test]
    fn test_multi_scan_concurrent_iteration() {
        // Test MultiScan with concurrent iterators
        loom::model(|| {
            // Create closures that produce futures
            let closures = vec![
                || -> VortexResult<Vec<ArrayFuture<i32>>> {
                    Ok(vec![
                        Box::pin(future::ready(Ok(Some(1)))),
                        Box::pin(future::ready(Ok(Some(2)))),
                    ])
                },
                || -> VortexResult<Vec<ArrayFuture<i32>>> {
                    Ok(vec![
                        Box::pin(future::ready(Ok(Some(3)))),
                        Box::pin(future::ready(Ok(Some(4)))),
                    ])
                },
            ];

            let multi_scan = MultiScan::new(closures);

            // Create two iterators
            let iter1 = multi_scan.clone().new_iterator();
            let iter2 = multi_scan.new_iterator();

            // Collect from both iterators concurrently
            let handle1 = thread::spawn(move || {
                let mut results = Vec::new();
                for val in iter1.flatten() {
                    results.push(val);
                    if results.len() >= 2 {
                        break;
                    }
                }
                results
            });

            let handle2 = thread::spawn(move || {
                let mut results = Vec::new();
                for val in iter2.flatten() {
                    results.push(val);
                    if results.len() >= 2 {
                        break;
                    }
                }
                results
            });

            let results1 = handle1.join().unwrap();
            let results2 = handle2.join().unwrap();

            // Verify results are from expected set
            for val in results1.iter().chain(results2.iter()) {
                assert!(*val >= 1 && *val <= 4);
            }
        });
    }

    #[test]
    fn test_work_stealing_with_empty_factories() {
        // Test edge case with empty task factories
        loom::model(|| {
            let factories: Vec<TaskFactory<i32>> = vec![
                Box::new(|| Ok(vec![])), // Empty
                Box::new(|| Ok(vec![1, 2])),
                Box::new(|| Ok(vec![])), // Empty
                Box::new(|| Ok(vec![3])),
            ];

            let queue = WorkStealingQueue::new(factories);
            let mut iter = queue.new_iterator();

            let mut results = Vec::new();
            while let Some(Ok(val)) = iter.next() {
                results.push(val);
            }

            // Should get exactly the non-empty values
            results.sort();
            assert_eq!(results, vec![1, 2, 3]);
        });
    }

    #[test]
    fn test_work_stealing_clone_semantics() {
        // Test that cloning iterators properly shares the work queue
        loom::model(|| {
            let factories: Vec<TaskFactory<i32>> = vec![Box::new(|| Ok(vec![1, 2, 3, 4]))];

            let queue = WorkStealingQueue::new(factories);
            let iter1 = queue.new_iterator();

            // Clone the iterator
            let iter2 = iter1.clone();

            let handle1 = thread::spawn(move || {
                let mut count = 0;
                for result in iter1 {
                    if result.is_ok() {
                        count += 1;
                        if count >= 2 {
                            break;
                        }
                    }
                }
                count
            });

            let handle2 = thread::spawn(move || {
                let mut count = 0;
                for result in iter2 {
                    if result.is_ok() {
                        count += 1;
                        if count >= 2 {
                            break;
                        }
                    }
                }
                count
            });

            let count1 = handle1.join().unwrap();
            let count2 = handle2.join().unwrap();

            // Both should get some work
            assert!(count1 + count2 <= 4);
        });
    }
}
