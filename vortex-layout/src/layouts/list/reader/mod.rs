// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests;

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::list::ListLayout;
use crate::layouts::list::expr::ElementProjection;
use crate::layouts::list::expr::ListChildrenNeeded;
use crate::layouts::list::expr::get_necessary_bound_list_children;
use crate::layouts::list::expr::rewrite_offsets_expr;
use crate::layouts::list::expr::rewrite_validity_expr;
use crate::segments::SegmentSource;

type OptionalArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// The threshold of mask density below which we push the input mask into projection evaluation,
/// and above which we evaluate the expression over all rows and intersect afterward.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Minimum size of the outer-row split grid, so lists in small files still fan out.
const MIN_LIST_SPLIT_COUNT: u64 = 64;

/// Outer rows per grid cell that the grid grows to stay under for large files.
const TARGET_LIST_SPLIT_ROWS: u64 = 8192;

/// Number of cells in the outer-row split grid: at least [`MIN_LIST_SPLIT_COUNT`], and enough
/// that a cell covers at most about [`TARGET_LIST_SPLIT_ROWS`] outer rows.
fn list_split_grid_size(outer_row_count: u64) -> u64 {
    outer_row_count
        .div_ceil(TARGET_LIST_SPLIT_ROWS)
        .next_power_of_two()
        .max(MIN_LIST_SPLIT_COUNT)
}

/// Reader for [`ListLayout`].
///
/// An expression that reads the list elements only through `list_transform(root, lambda)` runs
/// the lambda bodies over the elements child, so a lambda that reads one field of struct elements
/// reads only the column of that field. See [`ElementProjection`].
#[derive(Clone)]
pub struct ListReader {
    layout: ListLayout,
    name: Arc<str>,
    session: VortexSession,
    elements: LayoutReaderRef,
    offsets: LayoutReaderRef,
    validity: Option<LayoutReaderRef>,
    /// Element projections keyed by the identity of the expression they were derived from.
    /// Cached so the elements reader sees the same tree on every split, keeping its own
    /// identity-keyed caches warm.
    projections: Arc<ProjectionCache>,
}

type ProjectionCache = DashMap<ExactBoundExpr, Arc<OnceLock<Arc<ElementProjection>>>>;

impl ListReader {
    pub(super) fn try_new(
        layout: ListLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<Self> {
        let elements = layout.elements()?.new_reader(
            format!("{name}.elements").into(),
            Arc::clone(&segment_source),
            &session,
            ctx,
        )?;
        let offsets = layout.offsets()?.new_reader(
            format!("{name}.offsets").into(),
            Arc::clone(&segment_source),
            &session,
            ctx,
        )?;
        let validity = layout
            .validity()?
            .map(|v| {
                v.new_reader(
                    format!("{name}.validity").into(),
                    Arc::clone(&segment_source),
                    &session,
                    ctx,
                )
            })
            .transpose()?;

        Ok(Self {
            layout,
            name,
            session,
            elements,
            offsets,
            validity,
            projections: Arc::new(DashMap::with_hasher_and_shard_amount(Default::default(), 8)),
        })
    }

    /// Returns how the [`ListChildrenNeeded::All`] expression `expr` reads the elements child: the
    /// lambdas of its `list_transform(root, lambda)` nodes when it reads the elements only through
    /// them, or the elements as stored.
    fn element_projection(&self, expr: &BoundExpression) -> VortexResult<Arc<ElementProjection>> {
        let key = ExactBoundExpr(expr.clone());
        let cell = match self.projections.get(&key) {
            Some(entry) => Arc::clone(entry.value()),
            None => Arc::clone(
                self.projections
                    .entry(key)
                    .or_insert_with(|| Arc::new(OnceLock::new()))
                    .value(),
            ),
        };
        if let Some(projection) = cell.get() {
            return Ok(Arc::clone(projection));
        }
        let projection = match ElementProjection::push_down(expr)? {
            Some(projection) => projection,
            None => ElementProjection::identity(expr, self.layout.elements_dtype()),
        };
        Ok(Arc::clone(cell.get_or_init(|| Arc::new(projection))))
    }

    /// Projection for [`ListChildrenNeeded::Validity`] expressions. Reads only the validity child,
    /// synthesizing all-valid for a non-nullable list, and never touches the offsets or elements.
    fn project_validity(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let validity_reader = self.validity.clone();
        let nullability = self.layout.dtype().nullability();
        let row_range = row_range.clone();
        // Evaluate the rewritten expression against the validity bool array (true == valid row).
        let rewritten = rewrite_validity_expr(expr)?;

        Ok(async move {
            let mask = mask.await?;
            let row_count = usize::try_from(row_range.end - row_range.start)?;
            let out_len = if mask.all_true() {
                row_count
            } else {
                mask.true_count()
            };

            let validity_array = match validity_reader.as_ref() {
                Some(v) => {
                    let root = root().bind(v.dtype())?;
                    Some(
                        v.projection_evaluation(&row_range, &root, MaskFuture::ready(mask))?
                            .await?,
                    )
                }
                None => None,
            };

            let validity = create_validity(validity_array, nullability).to_array(out_len);

            validity.apply_bound(&rewritten)
        }
        .boxed())
    }

    /// Projection for [`ListChildrenNeeded::All`] expressions.
    ///
    /// The elements child is read through the [`ElementProjection`] of `expr`. A projection that
    /// can fail reads only the elements of the selected valid lists. Otherwise, an all-true mask
    /// over the full local range reads every child concurrently, and any other read is bounded to
    /// the first and last selected list.
    fn project_all(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let projection = self.element_projection(expr)?;
        let is_full_range = row_range.start == 0 && row_range.end == self.layout.row_count();
        let reader = self.clone();
        let row_range = row_range.clone();
        Ok(async move {
            let mask = mask.await?;
            if projection.is_fallible() {
                reader
                    .project_all_compacted(&row_range, projection, mask)?
                    .await
            } else if is_full_range && mask.all_true() {
                reader.project_all_full(projection)?.await
            } else {
                reader
                    .project_all_bounded(&row_range, projection, mask)?
                    .await
            }
        }
        .boxed())
    }

    /// Fetch the complete `elements`, `offsets`, and `validity` children concurrently.
    fn project_all_full(&self, projection: Arc<ElementProjection>) -> VortexResult<ArrayFuture> {
        let row_count = self.layout.row_count();
        let elements_row_count = self.elements.row_count();
        let nullability = self.layout.dtype().nullability();

        let offsets_fut = self.fetch_raw_offsets(&(0..row_count))?;
        let elements_fut = self.fetch_elements(
            &(0..elements_row_count),
            projection.elements(),
            MaskFuture::new_true(usize::try_from(elements_row_count)?),
        )?;
        let validity_fut = fetch_validity(
            self.validity.as_ref(),
            &(0..row_count),
            MaskFuture::new_true(usize::try_from(row_count)?),
        )?;

        Ok(async move {
            let (offsets, elements, validity) = try_join!(offsets_fut, elements_fut, validity_fut)?;
            // SAFETY: ListLayout is constructed from a valid ListArray, and the element-wise
            // projection keeps one value per element.
            let lists = unsafe {
                projection.rebuild(elements, offsets, create_validity(validity, nullability))
            }?;
            lists.apply_bound(projection.lists())
        }
        .boxed())
    }

    /// Bounded read for a sub-range or selective mask.
    ///
    /// Crops leading and trailing unselected lists, reads their offsets, and translates the first
    /// and last offset into the element-row range to fetch. Any holes in the selection are filtered
    /// after reconstructing the lists.
    fn project_all_bounded(
        &self,
        row_range: &Range<u64>,
        projection: Arc<ElementProjection>,
        mask: Mask,
    ) -> VortexResult<ArrayFuture> {
        // Crop to the smallest contiguous row range containing every selected list.
        let Some(selected_rows) = selected_row_range(&mask) else {
            let empty = Canonical::empty(projection.lists().dtype()).into_array();
            return Ok(async move { Ok(empty) }.boxed());
        };

        let selected_mask = mask.slice(selected_rows.clone());
        let selected_row_range = (row_range.start + u64::try_from(selected_rows.start)?)
            ..(row_range.start + u64::try_from(selected_rows.end)?);

        let nullability = self.layout.dtype().nullability();
        let reader = self.clone();
        let offsets_fut = self.fetch_raw_offsets(&selected_row_range)?;

        Ok(async move {
            let offsets = offsets_fut.await?;

            let elements_range = elements_range_from_offsets(&offsets, &reader.session)?;
            let elements_fut = reader.fetch_elements(
                &elements_range,
                projection.elements(),
                MaskFuture::new_true(usize::try_from(elements_range.end - elements_range.start)?),
            )?;
            let validity_fut = fetch_validity(
                reader.validity.as_ref(),
                &selected_row_range,
                MaskFuture::new_true(selected_mask.len()),
            )?;
            let (elements, validity) = try_join!(elements_fut, validity_fut)?;

            let offsets = rebase_offsets(offsets, elements_range.start)?;
            // SAFETY: the selected offsets remain monotonically increasing, rebasing them against
            // the selected element range preserves their lengths, the element-wise projection
            // keeps one value per element, and validity covers the same cropped list rows.
            let lists = unsafe {
                projection.rebuild(elements, offsets, create_validity(validity, nullability))
            }?;
            let lists = if selected_mask.all_true() {
                lists
            } else {
                lists.filter(selected_mask)?
            };
            lists.apply_bound(projection.lists())
        }
        .boxed())
    }

    /// Read for an element projection that can fail.
    ///
    /// `list_transform` runs a lambda that can fail only on the elements of valid lists, so this
    /// read evaluates the projection only on the elements of the selected valid lists. It reads
    /// the offsets and validity from the first to the last selected list, then the projected
    /// elements under a mask of the elements of the selected valid lists, and rebuilds the
    /// selected lists over just those elements.
    fn project_all_compacted(
        &self,
        row_range: &Range<u64>,
        projection: Arc<ElementProjection>,
        mask: Mask,
    ) -> VortexResult<ArrayFuture> {
        let Some(selected_rows) = selected_row_range(&mask) else {
            let empty = Canonical::empty(projection.lists().dtype()).into_array();
            return Ok(async move { Ok(empty) }.boxed());
        };

        let selected_mask = mask.slice(selected_rows.clone());
        let selected_row_range = (row_range.start + u64::try_from(selected_rows.start)?)
            ..(row_range.start + u64::try_from(selected_rows.end)?);

        let nullability = self.layout.dtype().nullability();
        let reader = self.clone();
        let offsets_fut = self.fetch_raw_offsets(&selected_row_range)?;
        let validity_fut = fetch_validity(
            self.validity.as_ref(),
            &selected_row_range,
            MaskFuture::new_true(selected_mask.len()),
        )?;

        Ok(async move {
            let (offsets, validity) = try_join!(offsets_fut, validity_fut)?;
            let mut ctx = reader.session.create_execution_ctx();
            let offsets = offsets
                .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
                .execute::<PrimitiveArray>(&mut ctx)?;
            let valid = match &validity {
                Some(validity) => validity.clone().execute::<Mask>(&mut ctx)?,
                None => Mask::new_true(selected_mask.len()),
            };
            let compacted = compact_lists(offsets.as_slice::<u64>(), &selected_mask, &valid)?;

            let elements = reader
                .fetch_elements(
                    &compacted.elements_range,
                    projection.elements(),
                    MaskFuture::ready(compacted.elements_mask),
                )?
                .await?;
            let validity = validity
                .map(|validity| validity.filter(selected_mask))
                .transpose()?;
            // SAFETY: the compacted offsets index exactly the masked elements, the element-wise
            // projection keeps one value per element, and validity is filtered to the same
            // selected lists.
            let lists = unsafe {
                projection.rebuild(
                    elements,
                    compacted.offsets.into_array(),
                    create_validity(validity, nullability),
                )
            }?;
            lists.apply_bound(projection.lists())
        }
        .boxed())
    }

    /// Projection for [`ListChildrenNeeded::OffsetsAndValidity`] expressions. Only reads offsets and validity children.
    fn project_offsets_validity(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let offsets = self.fetch_raw_offsets(row_range)?;
        let reader = self.clone();
        let row_range = row_range.clone();
        let lengths_dtype = DType::Primitive(PType::U64, self.layout.dtype().nullability());
        let rewritten = rewrite_offsets_expr(expr, &lengths_dtype)?;

        Ok(async move {
            let mask = mask.await?;
            let row_count = usize::try_from(row_range.end - row_range.start)?;
            let nullability = reader.layout.dtype().nullability();

            let validity_mask = if mask.all_true() {
                MaskFuture::new_true(row_count)
            } else {
                MaskFuture::ready(mask.clone())
            };
            let validity_fut = fetch_validity(reader.validity.as_ref(), &row_range, validity_mask)?;

            let offsets = offsets.await?;
            let lengths = list_lengths_from_offsets(offsets)?;
            let lengths = if mask.all_true() {
                lengths
            } else {
                lengths.filter(mask)?
            };
            let validity = validity_fut.await?;
            let lengths = apply_lengths_validity(lengths, validity, nullability)?;

            lengths.apply_bound(&rewritten)
        }
        .boxed())
    }

    /// Fire the offsets read for `row_range` in list row space. The offsets child has an extra entry, so reading
    /// `row_range` maps to offsets in `[row_range.start..row_range.end + 1)`.
    ///
    /// No mask or expression is applied.
    fn fetch_raw_offsets(&self, row_range: &Range<u64>) -> VortexResult<ArrayFuture> {
        let offsets_range = row_range.start..(row_range.end + 1);
        let offsets_count = usize::try_from(offsets_range.end - offsets_range.start)?;
        let root = root().bind(self.offsets.dtype())?;
        self.offsets.projection_evaluation(
            &offsets_range,
            &root,
            MaskFuture::new_true(offsets_count),
        )
    }

    /// Fire the read of `expr` over the elements in `row_range`, in element space, under `mask`.
    fn fetch_elements(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.elements.projection_evaluation(row_range, expr, mask)
    }
}

fn selected_row_range(mask: &Mask) -> Option<Range<usize>> {
    Some(mask.first()?..mask.last()? + 1)
}

fn create_validity(validity_array: Option<ArrayRef>, nullability: Nullability) -> Validity {
    match validity_array {
        Some(arr) => Validity::Array(arr),
        None => match nullability {
            Nullability::Nullable => Validity::AllValid,
            Nullability::NonNullable => Validity::NonNullable,
        },
    }
}

impl LayoutReader for ListReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        split_range.check_bounds(self.layout.row_count())?;

        // Splits are difficult to calculate because all children live in different row coordinate spaces.
        // List elements typically comprise the majority of the data in a list, and validity/offsets can be treated
        // as metadata. We therefore want to parallelize the scan based on element work.
        //
        // Scan splits must be expressed in the list layout's outer-row space, but the elements child
        // reports its natural boundaries in element-row space. So we translate the element splits using a
        // heuristic to outer-row space.

        let element_row_count = self.elements.row_count();
        if element_row_count != 0 {
            let mut element_splits = RowSplits::new_capacity(128);
            self.elements.register_splits(
                &[FieldMask::All],
                &SplitRange::root(0..element_row_count)?,
                &mut element_splits,
            )?;

            let row_range = split_range.row_range();
            let grid_size = list_split_grid_size(self.layout.row_count());
            let mut last_split = None;
            for element_split in element_splits.into_sorted_deduped() {
                let Some(split) = map_element_split_to_outer_grid(
                    element_split,
                    element_row_count,
                    self.layout.row_count(),
                    grid_size,
                ) else {
                    continue;
                };
                if split <= row_range.start {
                    continue;
                }
                if split >= row_range.end {
                    break;
                }
                if last_split == Some(split) {
                    continue;
                }
                splits.push(
                    split_range
                        .row_offset()
                        .checked_add(split)
                        .vortex_expect("List layout split offset overflow"),
                );
                last_split = Some(split);
            }
        }
        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    // TODO(mk): handle zones for list elements
    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let len = mask.len();
        let reader = self.clone();
        let row_range = row_range.clone();
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(MaskFuture::new(len, async move {
            let mask = mask.await?;

            if mask.all_false() {
                return Ok(mask);
            }

            let mut ctx = session.create_execution_ctx();
            if mask.density() < EXPR_EVAL_THRESHOLD {
                let predicate = reader
                    .projection_evaluation(&row_range, &expr, MaskFuture::ready(mask.clone()))?
                    .await?;
                let predicate_mask = predicate.fill_null(false)?.execute::<Mask>(&mut ctx)?;
                Ok(mask.intersect_by_rank(&predicate_mask))
            } else {
                let predicate = reader
                    .projection_evaluation(&row_range, &expr, MaskFuture::new_true(len))?
                    .await?;
                let predicate_mask = predicate.fill_null(false)?.execute::<Mask>(&mut ctx)?;
                Ok(mask & &predicate_mask)
            }
        }))
    }

    /// Reads only the list children needed to evaluate `expr`.
    ///
    /// Validity-only expressions avoid offsets and elements, length expressions read offsets and
    /// validity, and general expressions reconstruct a list from all applicable children.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        // Read as little as possible based on which list children the expression needs.
        match get_necessary_bound_list_children(expr) {
            ListChildrenNeeded::Validity => self.project_validity(row_range, expr, mask),
            ListChildrenNeeded::OffsetsAndValidity => {
                self.project_offsets_validity(row_range, expr, mask)
            }
            ListChildrenNeeded::All => self.project_all(row_range, expr, mask),
        }
    }
}

/// Converts a natural boundary from element-row space into an approximate outer-row scan split.
///
/// Scan splits must be expressed in the list layout's outer-row space, but the elements child
/// reports its natural boundaries in element-row space. Translating a boundary exactly would
/// require consulting the list offsets, so this function instead preserves its relative position:
///
/// ```text
/// element_split / element_row_count ≈ outer_split / outer_row_count
/// ```
///
/// The relative position is first rounded onto a grid containing at most `max_split_count` scan
/// ranges, then mapped into outer-row space. Multiple element boundaries may therefore map to the
/// same outer split and are deduplicated by the caller. With a grid size of 64, at most 63 interior
/// splits, and therefore 64 scan ranges, can be produced.
///
/// The result is only a task-sizing hint. It is always between outer rows, but it is not guaranteed
/// to correspond exactly to the original physical element boundary. Endpoint boundaries are
/// omitted because they do not subdivide the scan
fn map_element_split_to_outer_grid(
    element_split: u64,
    element_row_count: u64,
    outer_row_count: u64,
    max_split_count: u64,
) -> Option<u64> {
    if element_split == 0
        || element_split >= element_row_count
        || outer_row_count == 0
        || max_split_count < 2
    {
        return None;
    }
    debug_assert!(max_split_count.is_power_of_two());

    let grid_index = (u128::from(element_split) * u128::from(max_split_count)
        + u128::from(element_row_count / 2))
        / u128::from(element_row_count);
    if grid_index == 0 || grid_index >= u128::from(max_split_count) {
        return None;
    }

    let outer_split = grid_index * u128::from(outer_row_count) / u128::from(max_split_count);
    let outer_split = u64::try_from(outer_split)
        .vortex_expect("Outer split is bounded by the list layout row count");
    (outer_split != 0 && outer_split < outer_row_count).then_some(outer_split)
}

/// Fetch the validity child for `row_range` under `mask`, yielding `None` for a non-nullable list
/// (which has no validity child).
fn fetch_validity(
    validity: Option<&LayoutReaderRef>,
    row_range: &Range<u64>,
    mask: MaskFuture,
) -> VortexResult<OptionalArrayFuture> {
    let fut = validity
        .map(|v| {
            let root = root().bind(v.dtype())?;
            v.projection_evaluation(row_range, &root, mask)
        })
        .transpose()?;
    Ok(async move {
        match fut {
            Some(f) => f.await.map(Some),
            None => Ok(None),
        }
    }
    .boxed())
}

/// Read `offsets[0]` and `offsets[-1]` and return the elements range they bound.
fn elements_range_from_offsets(
    offsets: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<Range<u64>> {
    if offsets.is_empty() {
        return Ok(0..0);
    }
    let mut exec_ctx = session.create_execution_ctx();
    let start = offsets
        .execute_scalar(0, &mut exec_ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("offset value fits in u64");
    let end = offsets
        .execute_scalar(offsets.len() - 1, &mut exec_ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("offset value fits in u64");
    Ok(start..end)
}

/// The elements of the selected valid lists, and offsets that index them once the other elements
/// are dropped.
struct CompactedLists {
    /// The element rows from the first to the last of the lists.
    elements_range: Range<u64>,
    /// The elements of the selected valid lists, relative to `elements_range`.
    elements_mask: Mask,
    /// The offsets of the selected lists into the masked elements. Null lists are empty.
    offsets: Buffer<u64>,
}

/// Compacts the lists that `offsets` describe to the `selected` ones, keeping only the elements
/// of the `valid` ones, as `list_transform` does before it runs a lambda that can fail.
fn compact_lists(offsets: &[u64], selected: &Mask, valid: &Mask) -> VortexResult<CompactedLists> {
    let (Some(&first), Some(&last)) = (offsets.first(), offsets.last()) else {
        vortex_bail!("List offsets must hold at least one entry");
    };
    let mut slices: Vec<(usize, usize)> = Vec::new();
    let mut compacted = BufferMut::<u64>::with_capacity(selected.true_count() + 1);
    let mut kept = 0;
    compacted.push(kept);
    for (row, (selected, valid)) in selected.iter().zip(valid.iter()).enumerate() {
        if !selected {
            continue;
        }
        let (start, end) = (offsets[row], offsets[row + 1]);
        if valid && start < end {
            kept += end - start;
            let slice = (
                usize::try_from(start - first)?,
                usize::try_from(end - first)?,
            );
            match slices.last_mut() {
                Some((_, previous_end)) if *previous_end == slice.0 => *previous_end = slice.1,
                _ => slices.push(slice),
            }
        }
        compacted.push(kept);
    }
    Ok(CompactedLists {
        elements_range: first..last,
        elements_mask: Mask::from_slices(usize::try_from(last - first)?, slices),
        offsets: compacted.freeze(),
    })
}

/// Subtract `first` from every offset so they index into a sliced `elements[first..]` buffer that
/// starts at zero.
fn rebase_offsets(offsets: ArrayRef, first: u64) -> VortexResult<ArrayRef> {
    if first == 0 {
        return Ok(offsets);
    }
    let constant = ConstantArray::new(first, offsets.len())
        .into_array()
        .cast(offsets.dtype().clone())?;
    offsets.binary(constant, Operator::Sub)
}

/// Compute `offsets[i + 1] - offsets[i]` as the unmasked list length values.
fn list_lengths_from_offsets(offsets: ArrayRef) -> VortexResult<ArrayRef> {
    let len = offsets.len().saturating_sub(1);
    offsets
        .slice(1..offsets.len())?
        .binary(offsets.slice(0..len)?, Operator::Sub)
}

fn apply_lengths_validity(
    lengths: ArrayRef,
    validity: Option<ArrayRef>,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let len = lengths.len();
    let lengths = lengths.cast(DType::Primitive(PType::U64, nullability))?;

    if matches!(nullability, Nullability::Nullable) {
        lengths.mask(create_validity(validity, nullability).to_array(len))
    } else {
        Ok(lengths)
    }
}
