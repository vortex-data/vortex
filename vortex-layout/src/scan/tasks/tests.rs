// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::executor::block_on;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::dynamic;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scan::row_mask::RowMask;

use super::TaskContext;
use super::conditional_selectivity;
use super::split_exec;
use crate::ArrayFuture;
use crate::LayoutReader;
use crate::RowSplits;
use crate::SplitRange;
use crate::scan::filter::FilterExpr;
use crate::scan::metrics::ScanCounters;

struct DeferredReader {
    name: Arc<str>,
    dtype: DType,
    bound: Arc<AtomicI32>,
    pruning_calls: AtomicUsize,
    file_pruning_calls: AtomicUsize,
    file_pruning_update: Option<i32>,
    pending: bool,
}

impl LayoutReader for DeferredReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        1
    }

    fn can_prune_file(&self, _expr: &BoundExpression) -> VortexResult<bool> {
        self.file_pruning_calls.fetch_add(1, Ordering::Relaxed);
        if let Some(value) = self.file_pruning_update {
            self.bound.store(value, Ordering::Relaxed);
            return Ok(true);
        }
        Ok(false)
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        _split_range: &SplitRange,
        _splits: &mut RowSplits,
    ) -> VortexResult<()> {
        unreachable!("these tests construct split tasks directly")
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        self.pruning_calls.fetch_add(1, Ordering::Relaxed);
        if self.pending {
            return Ok(MaskFuture::new(mask.len(), futures::future::pending()));
        }
        let reject = self.bound.load(Ordering::Relaxed) > 100;
        let bound = self.bound.clone();
        Ok(MaskFuture::new(mask.len(), async move {
            bound.store(200, Ordering::Relaxed);
            Ok(if reject {
                Mask::new_false(mask.len())
            } else {
                mask
            })
        }))
    }

    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &BoundExpression,
        _mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        unreachable!("statistics reject the updated predicate")
    }

    fn projection_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &BoundExpression,
        _mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        Ok(Box::pin(futures::future::pending()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[rstest::rstest]
#[case::dynamic_update(false, 1, 2)]
#[case::cancellation(true, 0, 1)]
fn scan_counters_after_deferred_pruning(
    #[case] pending: bool,
    #[case] pruned: u64,
    #[case] pruning_calls: usize,
) -> VortexResult<()> {
    let bound = Arc::new(AtomicI32::new(50));
    let value = bound.clone();
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let predicate = dynamic(
        CompareOperator::Gt,
        move || Some(value.load(Ordering::Relaxed).into()),
        dtype.clone(),
        true,
        root(),
    )
    .bind(&dtype)?;
    let reader = Arc::new(DeferredReader {
        name: Arc::from("deferred"),
        dtype: dtype.clone(),
        bound,
        pruning_calls: AtomicUsize::new(0),
        file_pruning_calls: AtomicUsize::new(0),
        file_pruning_update: None,
        pending,
    });
    let counters = Arc::new(ScanCounters::default());
    let ctx = Arc::new(TaskContext {
        filter: Some(Arc::new(FilterExpr::new(predicate))),
        reader: reader.clone(),
        projection: BoundExpression::new_root(dtype),
        mapper: Arc::new(Ok::<ArrayRef, _>),
        counters: Some(counters.clone()),
        metrics: None,
    });
    let task = split_exec(ctx, RowMask::new(0, Mask::new_true(1)), None)?;
    assert_eq!(counters.snapshot().splits_considered, 0);
    if pending {
        assert!(task.now_or_never().is_none());
    } else {
        assert!(block_on(task)?.is_none());
    }
    assert_eq!(counters.snapshot().splits_considered, 1);
    assert_eq!(counters.snapshot().splits_pruned, pruned);
    assert_eq!(counters.snapshot().splits_filtered, 0);
    assert_eq!(reader.pruning_calls.load(Ordering::Relaxed), pruning_calls);
    assert_eq!(reader.file_pruning_calls.load(Ordering::Relaxed), pruning_calls);
    Ok(())
}

#[test]
fn unchanged_dynamic_predicates_reuse_file_statistics() -> VortexResult<()> {
    let bound = Arc::new(AtomicI32::new(50));
    let value = bound.clone();
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let predicate = dynamic(
        CompareOperator::Gt,
        move || Some(value.load(Ordering::Relaxed).into()),
        dtype.clone(),
        true,
        root(),
    )
    .bind(&dtype)?;
    let filter = FilterExpr::new(predicate.clone());
    let reader = DeferredReader {
        name: Arc::from("cached"),
        dtype,
        bound: bound.clone(),
        pruning_calls: AtomicUsize::new(0),
        file_pruning_calls: AtomicUsize::new(0),
        file_pruning_update: None,
        pending: false,
    };
    for expected_calls in 1..=2 {
        std::thread::scope(|scope| {
            let mut threads = Vec::new();
            for _ in 0..8 {
                let filter = &filter;
                let reader = &reader;
                threads.push(scope.spawn(move || -> VortexResult<()> {
                    for _ in 0..100 {
                        let version = filter.dynamic_updates(0).unwrap().version();
                        assert!(!filter.can_prune_file(0, Some(version), reader)?);
                    }
                    Ok(())
                }));
            }
            for thread in threads {
                thread.join().unwrap()?;
            }
            Ok::<_, vortex_error::VortexError>(())
        })?;
        assert_eq!(reader.file_pruning_calls.load(Ordering::Relaxed), expected_calls);
        bound.store(200, Ordering::Relaxed);
    }

    let next_execution = FilterExpr::new(predicate);
    let version = next_execution.dynamic_updates(0).unwrap().version();
    assert!(!next_execution.can_prune_file(0, Some(version), &reader)?);
    assert_eq!(reader.file_pruning_calls.load(Ordering::Relaxed), 3);
    Ok(())
}

#[test]
fn updates_during_file_statistics_evaluation_discard_the_result() -> VortexResult<()> {
    let bound = Arc::new(AtomicI32::new(50));
    let value = bound.clone();
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let predicate = dynamic(
        CompareOperator::Gt,
        move || Some(value.load(Ordering::Relaxed).into()),
        dtype.clone(),
        true,
        root(),
    )
    .bind(&dtype)?;
    let filter = FilterExpr::new(predicate);
    let reader = DeferredReader {
        name: Arc::from("update"),
        dtype,
        bound,
        pruning_calls: AtomicUsize::new(0),
        file_pruning_calls: AtomicUsize::new(0),
        file_pruning_update: Some(200),
        pending: false,
    };
    let version = filter.dynamic_updates(0).unwrap().version();
    assert!(!filter.can_prune_file(0, Some(version), &reader)?);
    let version = filter.dynamic_updates(0).unwrap().version();
    assert!(filter.can_prune_file(0, Some(version), &reader)?);
    assert!(filter.can_prune_file(0, Some(version), &reader)?);
    assert_eq!(reader.file_pruning_calls.load(Ordering::Relaxed), 2);
    Ok(())
}

#[test]
fn selectivity_is_relative_to_the_input_mask() {
    assert_eq!(conditional_selectivity(20, 5), 0.25);
}
