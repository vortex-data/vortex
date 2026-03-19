// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Split scanning task implementation.

use std::collections::BTreeMap;
use std::ops::BitAnd;
use std::ops::Range;
use std::sync::Arc;

use bit_vec::BitVec;
use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::ok;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::expr::Expression;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::LayoutReader;
use vortex_layout::ProjectionFetchHint;
use vortex_layout::segments::SegmentSource;
use vortex_layout::with_request_count_scope;
use vortex_mask::Mask;

use crate::fetch_plan::DeferredMaterializationPlan;
use crate::fetch_plan::MaterializationPlan;
use crate::filter::FilterExpr;
use crate::scan_metrics::ScanMetrics;
use crate::selection::Selection;

pub type TaskFuture<A> = BoxFuture<'static, VortexResult<A>>;

/// A split whose selection, pruning, and filter stages have already completed.
#[derive(Debug)]
pub(super) struct FilteredSplit {
    pub(super) row_range: Range<u64>,
    pub(super) mask: Mask,
    pub(super) projection_fetch_hints: Vec<ProjectionFetchHint>,
    pub(super) estimated_projection_bytes: usize,
    pub(super) selection_bytes_estimate: usize,
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
        let projection_fetch_hints = ctx.materialization_plan.fetch_hints(
            ctx.reader.as_ref(),
            &ctx.projection_field_mask,
            &row_range,
        )?;
        let estimated_projection_bytes = projection_fetch_hints.iter().fold(0usize, |sum, hint| {
            sum.saturating_add(hint.estimated_fetch_bytes)
        });
        Ok(Some(FilteredSplit {
            row_range,
            selection_bytes_estimate: mask.estimated_selection_bytes(),
            estimated_projection_bytes,
            mask,
            projection_fetch_hints,
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
        row_range,
        mask,
        projection_fetch_hints,
        ..
    } = filtered;
    let projection_field_count = projection_field_count(&ctx.materialization_plan, &ctx);
    let (projection_future, segment_request_count) =
        with_request_count_scope(|| -> VortexResult<_> {
            match &ctx.materialization_plan {
                MaterializationPlan::Monolithic { .. } => {
                    reader.projection_evaluation(&row_range, &projection, MaskFuture::ready(mask))
                }
                MaterializationPlan::Deferred(plan) => {
                    prepare_deferred_projection(reader, row_range.clone(), mask, plan.clone())
                }
            }
        });
    let projection_future = projection_future?;
    if let Some(metrics) = &ctx.scan_metrics {
        match &ctx.materialization_plan {
            MaterializationPlan::Monolithic { .. } => metrics.projection_tasks_monolithic.add(1),
            MaterializationPlan::Deferred(_) => metrics.projection_tasks_deferred.add(1),
        }
        metrics
            .projection_segment_requests
            .update(segment_request_count as f64);
        metrics
            .projection_fetch_hints
            .update(projection_fetch_hints.len() as f64);
        metrics
            .projection_fields
            .update(projection_field_count as f64);
    }

    let array_fut = async move {
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
    if matches!(
        ctx.materialization_plan,
        MaterializationPlan::Monolithic { .. }
    ) {
        return split_exec_monolithic(ctx, split, limit);
    }

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

fn split_exec_monolithic<A: 'static + Send>(
    ctx: Arc<TaskContext<A>>,
    split: Range<u64>,
    limit: Option<&mut u64>,
) -> VortexResult<TaskFuture<Option<A>>> {
    let read_mask = ctx.selection.row_mask(&split);
    let row_range = read_mask.row_range();
    let row_mask = read_mask.mask().clone();
    if row_mask.all_false() {
        return Ok(ok(None).boxed());
    }

    let filter_mask = match ctx.filter.as_ref() {
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
            let reader = ctx.reader.clone();
            let filter = filter.clone();
            let row_range = row_range.clone();

            MaskFuture::new(row_mask.len(), async move {
                let mut mask = row_mask;
                let mut dynamic_versions = vec![None; filter.conjuncts().len()];

                for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    dynamic_versions[idx] = filter.dynamic_updates(idx).map(|du| du.version());

                    let conjunct_mask = reader
                        .pruning_evaluation(&row_range, conjunct, mask.clone())?
                        .await?;
                    mask = mask.bitand(&conjunct_mask);
                }

                let mut remaining = BitVec::from_elem(filter.conjuncts().len(), true);
                while let Some(idx) = filter.next_conjunct(&remaining) {
                    remaining.set(idx, false);
                    if mask.all_false() {
                        return Ok(mask);
                    }

                    let conjunct = &filter.conjuncts()[idx];
                    let current_version = filter.dynamic_updates(idx).map(|du| du.version());
                    if let Some(dv) = current_version
                        && dynamic_versions[idx].is_none_or(|v| v < dv)
                    {
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
                    mask = conjunct_mask;
                }

                Ok(mask)
            })
        }
    };

    let projection_fetch_hints = ctx.materialization_plan.fetch_hints(
        ctx.reader.as_ref(),
        &ctx.projection_field_mask,
        &row_range,
    )?;
    let projection_field_count = projection_field_count(&ctx.materialization_plan, &ctx);
    let (projection_future, segment_request_count) = with_request_count_scope(|| {
        ctx.reader
            .projection_evaluation(&row_range, &ctx.projection, filter_mask.clone())
    });
    let projection_future = projection_future?;

    let mapper = ctx.mapper.clone();
    let scan_metrics = ctx.scan_metrics.clone();
    let array_fut = async move {
        let mask = filter_mask.await?;
        if mask.all_false() {
            return Ok(None);
        }

        if let Some(metrics) = &scan_metrics {
            metrics.projection_tasks_monolithic.add(1);
            metrics
                .projection_segment_requests
                .update(segment_request_count as f64);
            metrics
                .projection_fetch_hints
                .update(projection_fetch_hints.len() as f64);
            metrics
                .projection_fields
                .update(projection_field_count as f64);
        }

        let array = projection_future.await?;
        mapper(array).map(Some)
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
    /// The per-field materialization plan for projected output.
    pub(super) materialization_plan: MaterializationPlan,
    /// Optional metrics for scan scheduling and projection shaping.
    pub(super) scan_metrics: Option<Arc<ScanMetrics>>,
    /// Function that maps into an A.
    pub(super) mapper: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
    /// Optional segment source for signaling batch boundaries to the IO driver.
    pub(super) segment_source: Option<Arc<dyn SegmentSource>>,
}

fn projection_field_count<A>(plan: &MaterializationPlan, ctx: &TaskContext<A>) -> usize {
    match plan {
        MaterializationPlan::Monolithic { .. } => ctx.projection_field_mask.len(),
        MaterializationPlan::Deferred(plan) => {
            let immediate = usize::from(plan.immediate_expr().is_some());
            let deferred = plan.deferred_groups().len();
            immediate.saturating_add(deferred)
        }
    }
}

fn prepare_deferred_projection(
    reader: Arc<dyn LayoutReader>,
    row_range: Range<u64>,
    mask: Mask,
    plan: DeferredMaterializationPlan,
) -> VortexResult<TaskFuture<ArrayRef>> {
    let immediate_future = plan
        .immediate_expr()
        .map(|expr| {
            reader.projection_evaluation(&row_range, &expr, MaskFuture::ready(mask.clone()))
        })
        .transpose()?;
    let deferred_futures = plan
        .deferred_groups()
        .iter()
        .map(|group| {
            reader.projection_evaluation(
                &row_range,
                &group.projection_expr(),
                MaskFuture::ready(mask.clone()),
            )
        })
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(async move {
        let mut projected_fields = BTreeMap::<FieldName, ArrayRef>::new();
        let mut validity = None;

        if let Some(immediate) = immediate_future {
            let immediate = immediate.await?;
            collect_struct_fields(&immediate, &mut projected_fields, &mut validity)?;
        }

        for deferred in deferred_futures {
            let projected = deferred.await?;
            collect_struct_fields(&projected, &mut projected_fields, &mut validity)?;
        }

        let fields = plan
            .final_fields()
            .iter()
            .map(|field_name| {
                projected_fields
                    .remove(field_name)
                    .ok_or_else(|| vortex_err!("missing projected field {}", field_name))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(StructArray::try_new(
            plan.final_fields().clone(),
            fields,
            mask.true_count(),
            validity.unwrap_or(Validity::NonNullable),
        )?
        .into_array())
    }
    .boxed())
}

fn collect_struct_fields(
    array: &ArrayRef,
    projected_fields: &mut BTreeMap<FieldName, ArrayRef>,
    validity: &mut Option<Validity>,
) -> VortexResult<()> {
    let Some(struct_array) = array.as_opt::<Struct>() else {
        vortex_bail!("deferred materialization expects struct projection results");
    };

    if validity.is_none() {
        *validity = Some(struct_array.validity().clone());
    }

    for (field_name, field) in struct_array
        .names()
        .iter()
        .cloned()
        .zip(struct_array.unmasked_fields().iter().cloned())
    {
        projected_fields.insert(field_name, field);
    }

    Ok(())
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
    use crate::fetch_plan::MaterializationPlan;
    use crate::filter::FilterExpr;
    use crate::selection::Selection;
    use crate::tasks::FilteredSplit;
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
    fn split_exec_monolithic_registers_projection_before_filter_resolves() {
        let projection_calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(ProjectionCountingReader::new(projection_calls.clone()));
        let ctx = Arc::new(TaskContext {
            selection: Selection::default(),
            filter: Some(Arc::new(FilterExpr::new(eq(root(), lit(1i32))))),
            reader,
            projection: root(),
            projection_field_mask: vec![FieldMask::All],
            materialization_plan: MaterializationPlan::Monolithic {
                projected_row_bytes: 0,
                projection_aligned_splits: false,
            },
            scan_metrics: None,
            mapper: Arc::new(|array: ArrayRef| Ok(array)),
            segment_source: None,
        });

        let future = split_exec(ctx, 0..4, None).unwrap();
        assert_eq!(projection_calls.load(Ordering::Relaxed), 1);

        let result = block_on(future).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn split_exec_monolithic_registers_projection_before_poll() {
        let projection_calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(ProjectionCountingReader::new(projection_calls.clone()));
        let ctx = Arc::new(TaskContext {
            selection: Selection::default(),
            filter: None,
            reader,
            projection: root(),
            projection_field_mask: vec![FieldMask::All],
            materialization_plan: MaterializationPlan::Monolithic {
                projected_row_bytes: 0,
                projection_aligned_splits: false,
            },
            scan_metrics: None,
            mapper: Arc::new(|array: ArrayRef| Ok(array)),
            segment_source: None,
        });

        let _future = split_exec(ctx, 0..4, None).unwrap();
        assert_eq!(projection_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn filtered_split_still_defers_projection_registration() {
        let projection_calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(ProjectionCountingReader::new(projection_calls.clone()));
        let ctx = Arc::new(TaskContext {
            selection: Selection::default(),
            filter: None,
            reader,
            projection: root(),
            projection_field_mask: vec![FieldMask::All],
            materialization_plan: MaterializationPlan::Monolithic {
                projected_row_bytes: 0,
                projection_aligned_splits: false,
            },
            scan_metrics: None,
            mapper: Arc::new(|array: ArrayRef| Ok(array)),
            segment_source: None,
        });

        let _future = filter_split(ctx.clone(), 0..4, None).unwrap();
        assert_eq!(projection_calls.load(Ordering::Relaxed), 0);

        let result = block_on(split_exec(ctx, 0..4, None).unwrap()).unwrap();
        assert!(result.is_some());
        assert_eq!(projection_calls.load(Ordering::Relaxed), 1);
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
            materialization_plan: MaterializationPlan::Monolithic {
                projected_row_bytes: 0,
                projection_aligned_splits: false,
            },
            scan_metrics: None,
            mapper: Arc::new(|array: ArrayRef| Ok(array)),
            segment_source: None,
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

    #[test]
    fn project_filtered_split_registers_projection_before_poll() {
        let projection_calls = Arc::new(AtomicUsize::new(0));
        let reader = Arc::new(ProjectionCountingReader::new(projection_calls.clone()));
        let ctx = Arc::new(TaskContext {
            selection: Selection::default(),
            filter: None,
            reader,
            projection: root(),
            projection_field_mask: vec![FieldMask::All],
            materialization_plan: MaterializationPlan::Monolithic {
                projected_row_bytes: 0,
                projection_aligned_splits: false,
            },
            scan_metrics: None,
            mapper: Arc::new(|array: ArrayRef| Ok(array)),
            segment_source: None,
        });
        let filtered = FilteredSplit {
            row_range: 0..4,
            mask: Mask::new_true(4),
            projection_fetch_hints: Vec::new(),
            estimated_projection_bytes: 0,
            selection_bytes_estimate: 0,
        };

        let _future = project_filtered_split(ctx, filtered).unwrap();

        assert_eq!(projection_calls.load(Ordering::Relaxed), 1);
    }
}
