// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::ok;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::dtype::FieldMask;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_layout::LayoutReader;
use vortex_mask::Mask;

use crate::filter::FilterExpr;
use crate::selection::Selection;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// A split whose selection, pruning, and filter stages have already completed.
#[derive(Debug)]
pub(super) struct FilteredSplit {
    pub(super) row_range: Range<u64>,
    pub(super) mask: Mask,
    pub(super) projection_ranges: Vec<Range<u64>>,
}

/// Execute the selection, pruning, and filter stages for a single split.
pub(super) fn filter_split<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    split: Range<u64>,
    limit: Option<&mut u64>,
) -> VortexResult<TaskFuture<Option<FilteredSplit>>> {
    // Apply the selection to calculate a read mask
    let read_mask = ctx.selection.row_mask(&split);
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    if row_mask.all_false() {
        return Ok(ok(None).boxed());
    }

    let filter_mask = match ctx.filter.as_ref() {
        // No filter == immediate mask
        None => {
            let row_mask = match limit {
                Some(l) if *l == 0 => Mask::new_false(row_mask.len()),
                Some(l) => {
                    let true_count = row_mask.true_count();
                    let mask_limit = usize::try_from(*l)
                        .map(|l| l.min(true_count))
                        .unwrap_or(true_count);
                    let row_mask = row_mask.limit(mask_limit);
                    *l -= mask_limit as u64;
                    row_mask
                }
                None => row_mask,
            };

            MaskFuture::ready(row_mask)
        }
        Some(filter) => {
            // NOTE: it's very important that the pruning and filter evaluations are built OUTSIDE
            // the future. Registering these row ranges eagerly is a hint to the IO system that
            // we want to start prefetching the IO for this split.
            let reader = ctx.reader.clone();
            let filter = filter.clone();
            let row_range = row_range.clone();

            MaskFuture::new(row_mask.len(), async move {
                let mut mask = row_mask;
                let mut dynamic_versions = vec![None; filter.conjuncts().len()];

                // TODO(ngates): we could use FuturedUnordered to intersect the masks in parallel.
                for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    // Store the latest version of the dynamic expression prior to pruning.
                    // We will re-run the pruning later if the version has changed in the meantime.
                    dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());

                    let conjunct_mask = reader
                        .pruning_evaluation(&row_range, conjunct, mask.clone())?
                        .await?;
                    mask = mask.bitand(&conjunct_mask);
                }

                // Now we loop through the conjuncts in the preferred order and evaluate them.
                let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
                while let Some(idx) = filter.next_conjunct(&remaining) {
                    remaining.set(idx, false);
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let conjunct = &filter.conjuncts()[idx];

                    // If the dynamic expression has changed since pruning, re-run the pruning.
                    // Store the dynamic update once to avoid TOCTOU race condition
                    let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                    if let Some(dv) = current_version
                        && dynamic_versions[idx].is_none_or(|v| v < dv)
                    {
                        // The dynamic expression has been updated, re-run the pruning.
                        dynamic_versions[idx] = Some(dv);
                        let conjunct_mask = reader
                            .pruning_evaluation(&row_range, conjunct, mask.clone())?
                            .await?;
                        mask = mask.bitand(&conjunct_mask);
                    }
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let conjunct_mask = reader
                        .filter_evaluation(&row_range, conjunct, MaskFuture::ready(mask))?
                        .await?;
                    filter.report_selectivity(idx, conjunct_mask.density());

                    // Filter evaluations return a mask already intersected with the input mask.
                    mask = conjunct_mask;
                }

                Ok(mask)
            })
        }
    };

    let filtered = async move {
        let mask = filter_mask.await?;
        if mask.all_false() {
            return Ok(None);
        }

        let projection_ranges =
            projection_split_ranges(ctx.reader.as_ref(), &ctx.projection_field_mask, &row_range)?;
        Ok(Some(FilteredSplit {
            row_range,
            mask,
            projection_ranges,
        }))
    };

    Ok(filtered.boxed())
}

/// Project and map a split after its filter mask has already been resolved.
pub(super) fn project_filtered_split<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    filtered: FilteredSplit,
) -> VortexResult<TaskFuture<A>> {
    let reader = ctx.reader.clone();
    let projection = ctx.projection.clone();
    let mapper = ctx.mapper.clone();
    let FilteredSplit {
        row_range, mask, ..
    } = filtered;

    let array_fut = async move {
        // Only schedule payload reads once the filter has resolved for this split.
        let projection_future =
            reader.projection_evaluation(&row_range, &projection, MaskFuture::ready(mask))?;
        let array = projection_future.await?;
        mapper(array)
    };

    Ok(array_fut.boxed())
}

/// Logic for executing a single split reading task.
///
/// # Task execution flow
///
/// First, the task's row range (split) is intersected with the global file row-range requested,
/// if any.
///
/// The intersected row range is then further reduced via expression-based pruning. After pruning
/// has eliminated more blocks, the full filter is executed over the remainder of the split.
///
/// This mask is then provided to the reader to perform a filtered projection over the split data,
/// finally mapping the Vortex columnar record batches into some result type `A`.
pub(super) fn split_exec<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    split: Range<u64>,
    limit: Option<&mut u64>,
) -> VortexResult<TaskFuture<Option<A>>> {
    let filtered = filter_split(ctx.clone(), split, limit)?;
    let array_fut = async move {
        let Some(filtered) = filtered.await? else {
            return Ok(None);
        };
        let array = project_filtered_split(ctx, filtered)?.await?;
        Ok(Some(array))
    };

    Ok(array_fut.boxed())
}

/// Information needed to execute a single split task.
pub(super) struct TaskContext<A> {
    /// A row selection to apply.
    pub(super) selection: Selection,
    /// The shared filter expression.
    pub(super) filter: Option<Arc<FilterExpr>>,
    /// The layout reader.
    pub(super) reader: Arc<dyn LayoutReader>,
    /// The projection expression to apply to gather the scanned rows.
    pub(super) projection: Expression,
    /// Field mask for the projected columns, used to discover projection boundaries.
    pub(super) projection_field_mask: Vec<FieldMask>,
    /// Function that maps into an A.
    pub(super) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

fn projection_split_ranges(
    reader: &dyn LayoutReader,
    projection_field_mask: &[FieldMask],
    row_range: &Range<u64>,
) -> VortexResult<Vec<Range<u64>>> {
    if row_range.is_empty() {
        return Ok(Vec::new());
    }

    let mut start = row_range.start;
    let mut ranges = Vec::new();
    let split_points = reader.split_points(projection_field_mask.to_vec(), row_range.clone())?;
    for end in split_points {
        if end > start {
            ranges.push(start..end);
            start = end;
        }
    }

    if ranges.is_empty() {
        ranges.push(row_range.clone());
    }

    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::ops::BitAnd;
    use std::ops::Range;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::executor::block_on;
    use parking_lot::Mutex;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_layout::ArrayFuture;
    use vortex_layout::LayoutReader;
    use vortex_mask::Mask;

    use super::TaskContext;
    use crate::filter::FilterExpr;
    use crate::selection::Selection;
    use crate::tasks::filter_split;
    use crate::tasks::project_filtered_split;
    use crate::tasks::split_exec;

    struct ProjectionCountingReader {
        name: Arc<str>,
        dtype: DType,
        projection_calls: Arc<AtomicUsize>,
    }

    impl ProjectionCountingReader {
        fn new(projection_calls: Arc<AtomicUsize>) -> Self {
            Self {
                name: Arc::from("projection-counting"),
                dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
                projection_calls,
            }
        }
    }

    impl LayoutReader for ProjectionCountingReader {
        fn name(&self) -> &Arc<str> {
            &self.name
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            4
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            row_range: &Range<u64>,
            splits: &mut BTreeSet<u64>,
        ) -> VortexResult<()> {
            splits.insert(row_range.end);
            Ok(())
        }

        fn pruning_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            mask: Mask,
        ) -> VortexResult<MaskFuture> {
            Ok(MaskFuture::ready(mask))
        }

        fn filter_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            mask: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            let len = mask.len();
            Ok(MaskFuture::new(len, async move {
                drop(mask.await?);
                Ok(Mask::new_false(len))
            }))
        }

        fn projection_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            _mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            self.projection_calls.fetch_add(1, Ordering::Relaxed);
            let array = PrimitiveArray::from_iter(buffer![1i32, 2, 3, 4]).into_array();
            Ok(Box::pin(async move { Ok(array) }))
        }
    }

    struct MaskForwardingReader {
        name: Arc<str>,
        dtype: DType,
        filter_mask: Mask,
        projected_mask: Arc<Mutex<Option<Mask>>>,
    }

    impl MaskForwardingReader {
        fn new(filter_mask: Mask, projected_mask: Arc<Mutex<Option<Mask>>>) -> Self {
            Self {
                name: Arc::from("mask-forwarding"),
                dtype: DType::Primitive(PType::I32, Nullability::NonNullable),
                filter_mask,
                projected_mask,
            }
        }
    }

    impl LayoutReader for MaskForwardingReader {
        fn name(&self) -> &Arc<str> {
            &self.name
        }

        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn row_count(&self) -> u64 {
            4
        }

        fn register_splits(
            &self,
            _field_mask: &[FieldMask],
            row_range: &Range<u64>,
            splits: &mut BTreeSet<u64>,
        ) -> VortexResult<()> {
            splits.insert(row_range.end);
            Ok(())
        }

        fn pruning_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            mask: Mask,
        ) -> VortexResult<MaskFuture> {
            Ok(MaskFuture::ready(mask))
        }

        fn filter_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            mask: MaskFuture,
        ) -> VortexResult<MaskFuture> {
            let expected = self.filter_mask.clone();
            Ok(MaskFuture::new(mask.len(), async move {
                let mask = mask.await?;
                Ok(mask.bitand(&expected))
            }))
        }

        fn projection_evaluation(
            &self,
            _row_range: &Range<u64>,
            _expr: &Expression,
            mask: MaskFuture,
        ) -> VortexResult<ArrayFuture> {
            let projected_mask = self.projected_mask.clone();
            let array = PrimitiveArray::from_iter(buffer![20i32, 40]).into_array();
            Ok(Box::pin(async move {
                let mask = mask.await?;
                *projected_mask.lock() = Some(mask);
                Ok(array)
            }))
        }
    }

    #[test]
    fn split_exec_skips_projection_for_all_false_filter() {
        let projection_calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(ProjectionCountingReader::new(projection_calls.clone()));
        let ctx = Arc::new(TaskContext {
            selection: Selection::default(),
            filter: Some(Arc::new(FilterExpr::new(eq(root(), lit(1i32))))),
            reader,
            projection: root(),
            projection_field_mask: vec![FieldMask::All],
            mapper: Arc::new(|array: ArrayRef| Ok(array)),
        });

        let result = block_on(split_exec(ctx, 0..4, None).unwrap()).unwrap();
        assert!(result.is_none());
        assert_eq!(projection_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn filtered_split_mask_is_forwarded_unchanged_to_projection() {
        let filter_mask = Mask::from_indices(4, vec![1, 3]);
        let projected_mask = Arc::new(Mutex::new(None));
        let reader = Arc::new(MaskForwardingReader::new(
            filter_mask,
            projected_mask.clone(),
        ));
        let ctx = Arc::new(TaskContext {
            selection: Selection::default(),
            filter: Some(Arc::new(FilterExpr::new(eq(root(), lit(1i32))))),
            reader,
            projection: root(),
            projection_field_mask: vec![FieldMask::All],
            mapper: Arc::new(|array: ArrayRef| Ok(array)),
        });

        let filtered = block_on(filter_split(ctx.clone(), 0..4, None).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(filtered.row_range, 0..4);
        assert_eq!(filtered.mask.values().unwrap().indices(), &[1, 3]);

        let result = block_on(project_filtered_split(ctx, filtered).unwrap()).unwrap();
        assert_eq!(result.len(), 2);

        let projected_mask = projected_mask.lock();
        let projected_mask = projected_mask.as_ref().unwrap();
        assert_eq!(projected_mask.values().unwrap().indices(), &[1, 3]);
    }
}
