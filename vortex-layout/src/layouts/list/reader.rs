// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::list::ListLayout;
use crate::layouts::list::expr::ElementsSplit;
use crate::layouts::list::expr::ListChildrenNeeded;
use crate::layouts::list::expr::get_necessary_bound_list_children;
use crate::layouts::list::expr::rewrite_offsets_expr;
use crate::layouts::list::expr::rewrite_validity_expr;
use crate::layouts::list::expr::split_list_transform;
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
#[derive(Clone)]
pub struct ListReader {
    layout: ListLayout,
    name: Arc<str>,
    session: VortexSession,
    elements: LayoutReaderRef,
    offsets: LayoutReaderRef,
    validity: Option<LayoutReaderRef>,
}

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
        })
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
    /// An all-true mask over the full local range reads every child concurrently. Otherwise, the
    /// read is bounded to the first and last selected list.
    ///
    /// When the list is read through `list_transform`, the elements reader evaluates the lambdas'
    /// bodies instead of reading whole elements. See [`split_list_transform`].
    fn project_all(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let is_full_range = row_range.start == 0 && row_range.end == self.layout.row_count();
        let split = match split_list_transform(expr)? {
            Some(split) => split,
            None => ElementsSplit::whole_elements(self.elements.dtype(), expr),
        };
        let reader = self.clone();
        let row_range = row_range.clone();
        Ok(async move {
            let mask = mask.await?;
            if is_full_range && mask.all_true() && split.infallible {
                reader.project_all_full(split)?.await
            } else {
                reader.project_all_bounded(&row_range, split, mask)?.await
            }
        }
        .boxed())
    }

    /// Fetch the complete `elements`, `offsets`, and `validity` children concurrently.
    fn project_all_full(&self, split: ElementsSplit) -> VortexResult<ArrayFuture> {
        let row_count = self.layout.row_count();
        let elements_row_count = self.elements.row_count();
        let nullability = self.layout.dtype().nullability();
        let session = self.session.clone();

        let offsets_fut = self.fetch_raw_offsets(&(0..row_count))?;
        let elements_fut = self.fetch_elements(
            &(0..elements_row_count),
            &split.elements,
            MaskFuture::new_true(usize::try_from(elements_row_count)?),
        )?;
        let validity_fut = fetch_validity(
            self.validity.as_ref(),
            &(0..row_count),
            MaskFuture::new_true(usize::try_from(row_count)?),
        )?;

        Ok(async move {
            let (offsets, elements, validity) = try_join!(offsets_fut, elements_fut, validity_fut)?;
            let lists = lists_over(
                elements,
                offsets,
                create_validity(validity, nullability),
                split.fields.as_ref(),
                &mut session.create_execution_ctx(),
            )?;
            lists.apply_bound(&split.outer)
        }
        .boxed())
    }

    /// Bounded read for a sub-range, a selective mask, or a fallible elements expression.
    ///
    /// Crops leading and trailing unselected lists, reads their offsets, and translates the first
    /// and last offset into the element-row range to fetch. When the selection has holes, or the
    /// elements expression can fail and lists can be null, the elements read is masked to the
    /// elements the selected (valid) lists reference, so it skips the rest, and the lists are
    /// rebuilt over only those.
    fn project_all_bounded(
        &self,
        row_range: &Range<u64>,
        split: ElementsSplit,
        mask: Mask,
    ) -> VortexResult<ArrayFuture> {
        // Crop to the smallest contiguous row range containing every selected list.
        let Some(selected_rows) = selected_row_range(&mask) else {
            let empty = Canonical::empty(split.outer.dtype()).into_array();
            return Ok(async move { Ok(empty) }.boxed());
        };

        let selection = mask.slice(selected_rows.clone());
        let selected_row_range = (row_range.start + u64::try_from(selected_rows.start)?)
            ..(row_range.start + u64::try_from(selected_rows.end)?);

        let nullability = self.layout.dtype().nullability();
        let reader = self.clone();
        let offsets_fut = self.fetch_raw_offsets(&selected_row_range)?;
        let validity_fut = fetch_validity(
            self.validity.as_ref(),
            &selected_row_range,
            MaskFuture::new_true(selection.len()),
        )?;

        Ok(async move {
            let offsets = offsets_fut.await?;
            let elements_range = elements_range_from_offsets(&offsets, &reader.session)?;
            let mut ctx = reader.session.create_execution_ctx();

            let reads_every_element = selection.all_true()
                && (split.infallible || nullability == Nullability::NonNullable);
            let (elements, offsets, validity) = if reads_every_element {
                let elements_fut = reader.fetch_elements(
                    &elements_range,
                    &split.elements,
                    MaskFuture::new_true(usize::try_from(
                        elements_range.end - elements_range.start,
                    )?),
                )?;
                let (elements, validity) = try_join!(elements_fut, validity_fut)?;
                (
                    elements,
                    rebase_offsets(offsets, elements_range.start)?,
                    validity,
                )
            } else {
                let validity = validity_fut.await?;
                let selected =
                    select_lists(&offsets, validity, &selection, split.infallible, &mut ctx)?;
                let elements = reader
                    .fetch_elements(
                        &elements_range,
                        &split.elements,
                        MaskFuture::ready(selected.elements),
                    )?
                    .await?;
                (elements, selected.offsets, selected.validity)
            };

            let lists = lists_over(
                elements,
                offsets,
                create_validity(validity, nullability),
                split.fields.as_ref(),
                &mut ctx,
            )?;
            lists.apply_bound(&split.outer)
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

    /// Fire the read of `expr` over the elements of `row_range` that `mask` selects, in element
    /// space.
    fn fetch_elements(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        self.elements.projection_evaluation(row_range, expr, mask)
    }
}

/// The lists a bounded read keeps, over only the elements it reads.
struct SelectedLists {
    /// The elements to read, out of the range the first and last list span.
    elements: Mask,
    /// The kept lists' offsets into the elements read.
    offsets: ArrayRef,
    /// The kept lists' validity.
    validity: Option<ArrayRef>,
}

/// Keep the `selection` lists of a cropped read, given their offsets (one more than the lists)
/// and validity.
///
/// The elements read are those of the kept lists. Unless `infallible`, null lists come out empty
/// and their elements are not read: whatever they hold, the elements expression never sees it.
fn select_lists(
    offsets: &ArrayRef,
    validity: Option<ArrayRef>,
    selection: &Mask,
    infallible: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<SelectedLists> {
    let keep = match &validity {
        Some(validity) if !infallible => {
            selection & &Validity::Array(validity.clone()).execute_mask(selection.len(), ctx)?
        }
        _ => selection.clone(),
    };
    let offsets = offsets
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.as_slice::<u64>();
    let lists = selection.len();

    let (elements, kept_offsets) = kept_elements(offsets, &keep)?;
    let offsets = match selection.slices() {
        AllOr::All => kept_offsets.freeze(),
        AllOr::None => Buffer::from_iter([0]),
        AllOr::Some(runs) => {
            let mut picked = BufferMut::with_capacity(selection.true_count() + 1);
            for &(start, end) in runs {
                picked.extend_from_slice(&kept_offsets[start..end]);
            }
            picked.push(kept_offsets[lists]);
            picked.freeze()
        }
    };
    let validity = match validity {
        Some(validity) if !selection.all_true() => Some(validity.filter(selection.clone())?),
        validity => validity,
    };
    Ok(SelectedLists {
        elements,
        offsets: offsets.into_array(),
        validity,
    })
}

/// Which elements the `keep` lists reference, counted from where the first list starts, and the
/// offsets of every list over only those elements. Lists not kept come out empty.
fn kept_elements(offsets: &[u64], keep: &Mask) -> VortexResult<(Mask, BufferMut<u64>)> {
    let lists = keep.len();
    let first = offsets[0];
    let len = usize::try_from(offsets[lists] - first)?;
    let all = [(0, lists)];
    let runs = keep.slices().unwrap_or_else(|| &all, || &[]);

    let mut kept_offsets = BufferMut::with_capacity(lists + 1);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    let mut kept = 0u64;
    let mut from = 0;
    for &(start, end) in runs {
        // Lists between runs are dropped, so they all start where the next kept list does.
        kept_offsets.push_n(kept, start - from);
        let base = offsets[start];
        kept_offsets.extend(
            offsets[start..end]
                .iter()
                .map(|&offset| kept + (offset - base)),
        );
        let range = (
            usize::try_from(base - first)?,
            usize::try_from(offsets[end] - first)?,
        );
        if range.0 < range.1 {
            match ranges.last_mut() {
                Some(last) if last.1 == range.0 => last.1 = range.1,
                _ => ranges.push(range),
            }
        }
        kept += offsets[end] - base;
        from = end;
    }
    kept_offsets.push_n(kept, lists + 1 - from);

    let elements = match ranges[..] {
        [] => Mask::new_false(len),
        [(0, end)] if end == len => Mask::new_true(len),
        [_] => Mask::from_slices(len, ranges),
        _ => {
            let mut bits = BitBufferMut::new_unset(len);
            for &(start, end) in &ranges {
                bits.fill_range(start, end, true);
            }
            Mask::from_buffer_with_slices(bits.freeze(), ranges)
        }
    };
    Ok((elements, kept_offsets))
}

/// The lists over `elements`, or when the elements are a pack of `fields`, a struct of one list
/// per field, all sharing `offsets` and `validity`.
fn lists_over(
    elements: ArrayRef,
    offsets: ArrayRef,
    validity: Validity,
    fields: Option<&FieldNames>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // SAFETY: the offsets are monotonically increasing and index into the elements read, which
    // give one value per element, and the validity covers the same lists.
    let Some(fields) = fields else {
        return Ok(unsafe { ListArray::new_unchecked(elements, offsets, validity) }.into_array());
    };
    let len = offsets.len() - 1;
    let elements = elements.execute::<StructArray>(ctx)?;
    let lists = fields
        .iter()
        .map(|field| {
            let values = elements.unmasked_field_by_name(field)?.clone();
            // SAFETY: as above, for each field of the elements.
            Ok(
                unsafe { ListArray::new_unchecked(values, offsets.clone(), validity.clone()) }
                    .into_array(),
            )
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(StructArray::try_new(fields.clone(), lists, len, Validity::NonNullable)?.into_array())
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
/// splits—and therefore 64 scan ranges—can be produced.
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::ops::Range;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use parking_lot::Mutex;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::RecursiveCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::Expression;
    use vortex_array::expr::Lambda;
    use vortex_array::expr::cast;
    use vortex_array::expr::checked_add;
    use vortex_array::expr::col;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::list_length;
    use vortex_array::expr::list_sum;
    use vortex_array::expr::list_transform;
    use vortex_array::expr::lit;
    use vortex_array::expr::pack;
    use vortex_array::expr::var;
    use vortex_buffer::buffer;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;
    use vortex_utils::aliases::hash_map::HashMap;

    use super::*;
    use crate::LayoutChildType;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::writer::ListLayoutStrategy;
    use crate::layouts::repartition::RepartitionStrategy;
    use crate::layouts::repartition::RepartitionWriterOptions;
    use crate::layouts::table::TableStrategy;
    use crate::scan::split_by::SplitBy;
    use crate::segments::SegmentFuture;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;
    use crate::test::SESSION;

    /// Validity-class projections (`is_null` / `is_not_null` of the list) round-trip through the
    /// validity-only read path, for both nullable and non-nullable lists.
    #[rstest]
    // `create_basic_list_array(true)` has validity `[true, false, true]`.
    #[case::nullable(true, vec![true, false, true])]
    #[case::non_nullable(false, vec![true, true, true])]
    #[tokio::test]
    async fn projection_validity_class(
        #[case] nullable: bool,
        #[case] valid: Vec<bool>,
    ) -> VortexResult<()> {
        let list = create_basic_list_array(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let not_null_expr = is_not_null(root()).bind(reader.dtype())?;
        let not_null = reader
            .projection_evaluation(&(0..3), &not_null_expr, MaskFuture::new_true(3))?
            .await?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(not_null, BoolArray::from_iter(valid.clone()), &mut exec_ctx);

        let is_null_expr = is_null(root()).bind(reader.dtype())?;
        let is_null_res = reader
            .projection_evaluation(&(0..3), &is_null_expr, MaskFuture::new_true(3))?
            .await?;
        assert_arrays_eq!(
            is_null_res,
            BoolArray::from_iter(valid.iter().map(|v| !v).collect::<Vec<_>>()),
            &mut exec_ctx
        );

        Ok(())
    }

    #[tokio::test]
    async fn projection_list_length_reads_offsets() -> VortexResult<()> {
        let list = create_basic_list_array(false);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = list_length(root()).bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
            .await?;

        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, buffer![2u64, 2, 1].into_array(), &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_list_length_preserves_validity() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = list_length(root()).bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
            .await?;

        let expected =
            PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(1)]).into_array();
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_list_length_applies_sparse_mask() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let mask = Mask::from_iter([false, true, true]);
        let expr = list_length(root()).bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(0..3), &expr, MaskFuture::ready(mask))?
            .await?;

        let expected = PrimitiveArray::from_option_iter::<u64, _>([None, Some(1)]).into_array();
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_cast_list_length() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = cast(
            list_length(root()),
            DType::Primitive(PType::I64, Nullability::Nullable),
        )
        .bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
            .await?;

        let expected =
            PrimitiveArray::from_option_iter::<i64, _>([Some(2), None, Some(1)]).into_array();
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn filter_evaluation_list_length() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = gt(list_length(root()), lit(1u64)).bind(reader.dtype())?;
        let result = reader
            .filter_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
            .await?;

        assert_eq!(result, Mask::from_iter([true, false, false]));
        Ok(())
    }

    #[rstest]
    #[case::is_not_null_nullable(true, is_not_null(root()), Mask::from_iter([true, false, true]))]
    #[case::is_not_null_non_nullable(false, is_not_null(root()), Mask::new_true(3))]
    #[case::is_null_nullable(true, is_null(root()), Mask::from_iter([false, true, false]))]
    #[case::is_null_non_nullable(false, is_null(root()), Mask::new_false(3))]
    #[tokio::test]
    async fn filter_evaluation_validity_class(
        #[case] nullable: bool,
        #[case] expr: Expression,
        #[case] expected: Mask,
    ) -> VortexResult<()> {
        let list = create_basic_list_array(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = expr.bind(reader.dtype())?;
        let result = reader
            .filter_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
            .await?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[tokio::test]
    async fn filter_evaluation_intersects_with_input_mask() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let input_mask = Mask::from_iter([true, true, false]);
        let expr = is_not_null(root()).bind(reader.dtype())?;
        let result = reader
            .filter_evaluation(&(0..3), &expr, MaskFuture::ready(input_mask))?
            .await?;

        assert_eq!(result, Mask::from_iter([true, false, false]));
        Ok(())
    }

    #[tokio::test]
    async fn filter_evaluation_sparse_mask_maps_by_rank() -> VortexResult<()> {
        let list = create_six_list_array();
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let input_mask = Mask::from_iter([false, false, false, false, true, false]);
        let expr = is_not_null(root()).bind(reader.dtype())?;
        let result = reader
            .filter_evaluation(&(0..6), &expr, MaskFuture::ready(input_mask))?
            .await?;

        assert_eq!(
            result,
            Mask::from_iter([false, false, false, false, true, false])
        );
        Ok(())
    }

    fn flat_list_strategy() -> ListLayoutStrategy {
        ListLayoutStrategy::default()
    }

    fn layout_test_session() -> VortexSession {
        VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
    }

    async fn write_layout<S: LayoutStrategy>(
        strategy: &S,
        array: ArrayRef,
    ) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef, VortexSession)> {
        let session = layout_test_session().with_tokio();
        let segments = Arc::new(TestSegments::default());
        let segments_ref: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let layout = strategy
            .write_stream(
                ArrayContext::empty().into(),
                segments,
                stream,
                eof,
                &session,
            )
            .await?;
        Ok((segments_ref, layout, session))
    }

    fn materialize_u64_array(array: ArrayRef) -> Vec<u64> {
        let mut ctx = SESSION.create_execution_ctx();
        array
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
            .as_slice::<u64>()
            .to_vec()
    }

    fn create_basic_list_array(nullable: bool) -> ArrayRef {
        let mut ctx = SESSION.create_execution_ctx();
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array())
        } else {
            Validity::NonNullable
        };

        ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 4, 5].into_array(),
            validity,
            &mut ctx,
        )
        .expect("array is valid")
        .into_array()
    }

    fn create_six_list_array() -> ArrayRef {
        let mut ctx = SESSION.create_execution_ctx();
        let validity = Validity::Array(
            BoolArray::from_iter([true, false, true, false, true, true]).into_array(),
        );

        ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 1, 2, 3, 4, 5, 6].into_array(),
            validity,
            &mut ctx,
        )
        .expect("array is valid")
        .into_array()
    }

    #[tokio::test]
    async fn fetch_offsets_includes_extra_endpoint() -> VortexResult<()> {
        let list = create_basic_list_array(false);

        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let ctx = LayoutReaderContext::new();
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;
        let reader = reader
            .as_any()
            .downcast_ref::<ListReader>()
            .expect("ListReader");

        let offsets = reader.fetch_raw_offsets(&(1..3))?.await?;
        assert_eq!(materialize_u64_array(offsets), vec![2u64, 4, 5]);

        Ok(())
    }

    #[rstest]
    #[case::full_range(0..3, false)]
    #[case::partial_start(0..2, false)]
    #[case::partial_end(1..3, false)]
    #[case::middle_single(1..2, false)]
    #[case::empty_range(1..1, false)]
    #[case::full_range_null(0..3, true)]
    #[tokio::test]
    async fn projection_evaluation_round_trips(
        #[case] row_range: Range<u64>,
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let list = create_basic_list_array(nullable);
        let ctx = LayoutReaderContext::new();

        let len = usize::try_from(row_range.end - row_range.start)?;
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = root().bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&row_range, &expr, MaskFuture::new_true(len))?
            .await?;

        let expected =
            list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_evaluation_applies_mask() -> VortexResult<()> {
        let list = create_basic_list_array(false);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let mask = Mask::from_iter([true, false, true]);
        let expr = root().bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(0..3), &expr, MaskFuture::ready(mask.clone()))?
            .await?;

        let expected = list.filter(mask)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    /// Build a list with 5 rows and lengths [2, 3, 0, 3, 2].
    fn create_wider_list_array(nullable: bool) -> ArrayRef {
        let mut ctx = SESSION.create_execution_ctx();
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, true, false, true, true]).into_array())
        } else {
            Validity::NonNullable
        };
        ListArray::try_new(
            buffer![10i32, 11, 20, 21, 22, 30, 31, 32, 40, 41].into_array(),
            buffer![0u32, 2, 5, 5, 8, 10].into_array(),
            validity,
            &mut ctx,
        )
        .expect("array is valid")
        .into_array()
    }

    /// Sparse row masks over a whole chunk round-trip: the whole-chunk read is filtered by the
    /// mask in memory.
    #[rstest]
    #[case::single_middle(Mask::from_iter([false, false, false, true, false]), false)]
    #[case::two_far_apart(Mask::from_iter([true, false, false, true, false]), false)]
    #[case::boundaries(Mask::from_iter([true, false, false, false, true]), false)]
    #[case::kept_empty_row(Mask::from_iter([false, false, true, false, false]), false)]
    #[case::sparse_nullable(Mask::from_iter([true, false, true, false, true]), true)]
    #[case::all_false(Mask::new_false(5), false)]
    #[tokio::test]
    async fn projection_evaluation_sparse_mask_round_trips(
        #[case] mask: Mask,
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let list = create_wider_list_array(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = root().bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(0..5), &expr, MaskFuture::ready(mask.clone()))?
            .await?;

        let expected = list.filter(mask)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    /// A partial range against a single flat elements segment takes the bounded path. The flat
    /// reader may still fetch its whole segment, but the list reader reconstructs only the requested
    /// element range.
    #[tokio::test]
    async fn projection_evaluation_partial_range_bounded() -> VortexResult<()> {
        let list = create_wider_list_array(false);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = root().bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(1..4), &expr, MaskFuture::new_true(3))?
            .await?;

        let expected = list.slice(1..4)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[test]
    fn maps_element_splits_to_outer_grid() {
        assert_eq!(map_element_split_to_outer_grid(0, 100, 100, 8), None);
        assert_eq!(map_element_split_to_outer_grid(20, 100, 100, 8), Some(25));
        assert_eq!(map_element_split_to_outer_grid(25, 100, 100, 8), Some(25));
        assert_eq!(map_element_split_to_outer_grid(50, 100, 100, 8), Some(50));
        assert_eq!(map_element_split_to_outer_grid(75, 100, 100, 8), Some(75));
        assert_eq!(map_element_split_to_outer_grid(100, 100, 100, 8), None);

        let mut splits = (1..1_000)
            .filter_map(|split| {
                map_element_split_to_outer_grid(split, 1_000, 100_000, MIN_LIST_SPLIT_COUNT)
            })
            .collect::<Vec<_>>();
        splits.dedup();

        let expected = (1..MIN_LIST_SPLIT_COUNT)
            .map(|grid_index| grid_index * 100_000 / MIN_LIST_SPLIT_COUNT)
            .collect::<Vec<_>>();
        assert_eq!(splits, expected);
    }

    #[rstest]
    #[case::empty(0, 64)]
    #[case::small(100_000, 64)]
    #[case::floor_boundary(64 * 8192, 64)]
    #[case::past_floor(64 * 8192 + 1, 128)]
    #[case::million(1_000_000, 128)]
    #[case::ten_million(10_000_000, 2048)]
    fn split_grid_grows_with_row_count(#[case] outer_row_count: u64, #[case] expected: u64) {
        assert_eq!(list_split_grid_size(outer_row_count), expected);
    }

    #[tokio::test]
    async fn nested_list_propagates_element_splits() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let inner = ListArray::try_new(
            PrimitiveArray::from_iter(0..128_i32).into_array(),
            PrimitiveArray::from_iter((0..=8_u32).map(|idx| idx * 16)).into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let outer = ListArray::try_new(
            inner,
            buffer![0u32, 2, 4, 6, 8].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();

        let inner_strategy =
            ListLayoutStrategy::default().with_elements(chunked_elements_strategy());
        let strategy = ListLayoutStrategy::default().with_elements(Arc::new(inner_strategy));
        let (segments, layout, session) = write_layout(&strategy, outer).await?;
        let reader =
            layout.new_reader("".into(), segments, &session, &LayoutReaderContext::new())?;

        let splits = SplitBy::Layout.splits(reader.as_ref(), &(0..4), &[FieldMask::All])?;
        assert_eq!(splits, vec![0, 1, 2, 3, 4]);
        Ok(())
    }

    /// A list strategy whose `elements` child is repartitioned into two-element chunks, so the
    /// reader takes the bounded (chunk-skipping) path for strict sub-ranges. Offsets stay flat.
    fn chunked_elements_list_strategy() -> ListLayoutStrategy {
        ListLayoutStrategy::default().with_elements(chunked_elements_strategy())
    }

    fn chunked_elements_strategy() -> Arc<dyn LayoutStrategy> {
        Arc::new(RepartitionStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            RepartitionWriterOptions {
                block_size_minimum: 0,
                block_len_multiple: 2,
                block_size_target: None,
                canonicalize: true,
            },
        ))
    }

    struct CountingSegmentSource {
        inner: Arc<dyn SegmentSource>,
        request_count: Arc<AtomicUsize>,
    }

    impl SegmentSource for CountingSegmentSource {
        fn request(&self, id: crate::segments::SegmentId) -> SegmentFuture {
            self.request_count.fetch_add(1, Ordering::Relaxed);
            self.inner.request(id)
        }
    }

    /// The chunked-elements strategy must actually produce a chunked `elements` layout, otherwise
    /// the reader would silently take the whole-chunk path and the bounded read would be untested.
    #[tokio::test]
    async fn chunked_elements_produces_chunked_layout() -> VortexResult<()> {
        let list = create_wider_list_array(false);
        let (_segments, layout, _session) =
            write_layout(&chunked_elements_list_strategy(), list).await?;
        let tree = layout.display_tree().to_string();
        assert!(
            tree.contains("elements: vortex.chunked"),
            "elements should be chunked:\n{tree}"
        );
        Ok(())
    }

    /// A sparse mask must remain selective even when the requested row range covers the complete
    /// local list layout. The selected first list touches one element chunk plus the offsets
    /// segment; reading all five element chunks would make the count six.
    #[tokio::test]
    async fn full_range_sparse_mask_crops_element_read() -> VortexResult<()> {
        let list = create_wider_list_array(false);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) =
            write_layout(&chunked_elements_list_strategy(), list.clone()).await?;
        let request_count = Arc::new(AtomicUsize::new(0));
        let source = Arc::new(CountingSegmentSource {
            inner: segments,
            request_count: Arc::clone(&request_count),
        });
        let reader = layout.new_reader("".into(), source, &session, &ctx)?;

        let mask = Mask::from_iter([true, false, false, false, false]);
        let expr = root().bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(0..5), &expr, MaskFuture::ready(mask.clone()))?
            .await?;

        let expected = list.filter(mask)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        assert_eq!(request_count.load(Ordering::Relaxed), 2);
        Ok(())
    }

    /// With chunked elements, sub-range projections take the bounded read path. Every
    /// range/mask/nullability combination must match the same projection over the ground-truth
    /// array (`list.slice(range).filter(mask)`).
    #[rstest]
    #[case::full_all_true(0..5, Mask::new_true(5), false)]
    #[case::subrange_all_true(1..4, Mask::new_true(3), false)]
    #[case::subrange_sparse(1..4, Mask::from_iter([true, false, true]), false)]
    #[case::partial_start(0..2, Mask::new_true(2), false)]
    #[case::partial_end(2..5, Mask::new_true(3), false)]
    #[case::single_non_empty(0..1, Mask::new_true(1), false)]
    #[case::single_empty_row(2..3, Mask::from_iter([true]), false)]
    #[case::empty_range(2..2, Mask::new_true(0), false)]
    #[case::subrange_all_false(1..4, Mask::new_false(3), false)]
    #[case::subrange_single_interior(0..4, Mask::from_iter([false, true, false, false]), false)]
    #[case::subrange_sparse_nullable(1..4, Mask::from_iter([true, false, true]), true)]
    #[case::partial_end_nullable(2..5, Mask::new_true(3), true)]
    #[tokio::test]
    async fn chunked_elements_round_trips(
        #[case] row_range: Range<u64>,
        #[case] mask: Mask,
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let list = create_wider_list_array(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) =
            write_layout(&chunked_elements_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = root().bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&row_range, &expr, MaskFuture::ready(mask.clone()))?
            .await?;

        let sliced =
            list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?;
        let expected = sliced.filter(mask)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    struct RecordingSegmentSource {
        inner: Arc<dyn SegmentSource>,
        requested: Arc<Mutex<Vec<crate::segments::SegmentId>>>,
    }

    impl SegmentSource for RecordingSegmentSource {
        fn request(&self, id: crate::segments::SegmentId) -> SegmentFuture {
            self.requested.lock().push(id);
            self.inner.request(id)
        }
    }

    /// Map each segment to its column, a dotted path of field and auxiliary child names. Chunks
    /// and transparent children add nothing to the path.
    fn segment_columns(
        layout: &LayoutRef,
        path: &str,
        columns: &mut HashMap<crate::segments::SegmentId, String>,
    ) -> VortexResult<()> {
        for id in layout.segment_ids() {
            columns.insert(id, path.to_string());
        }
        for (child_type, child) in layout.child_types().zip(layout.children()?) {
            let child_path = match child_type {
                LayoutChildType::Field(name) if path.is_empty() => name.to_string(),
                LayoutChildType::Field(name) => format!("{path}.{name}"),
                LayoutChildType::Auxiliary(name) => format!("{path}.{name}"),
                LayoutChildType::Chunk(_) | LayoutChildType::Transparent(_) => path.to_string(),
            };
            segment_columns(&child, &child_path, columns)?;
        }
        Ok(())
    }

    /// `{l: list<struct<a: i32, f: i32, g: list<struct<h: i32, i: i32>>>>, o: i32}`. The five
    /// lists have lengths [2, 3, 0, 3, 2], and when `nullable` the second is null over elements
    /// that are still stored.
    fn transform_table(nullable: bool) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let inner = StructArray::from_fields(&[
            ("h", PrimitiveArray::from_iter(0..15_i32).into_array()),
            ("i", PrimitiveArray::from_iter(100..115_i32).into_array()),
        ])?
        .into_array();
        let g = ListArray::try_new(
            inner,
            buffer![0u32, 1, 3, 3, 4, 6, 6, 7, 9, 9, 15].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let elements = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter(0..10_i32).into_array()),
            ("f", PrimitiveArray::from_iter(10..20_i32).into_array()),
            ("g", g),
        ])?
        .into_array();
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, false, true, true, true]).into_array())
        } else {
            Validity::NonNullable
        };
        let l = ListArray::try_new(
            elements,
            buffer![0u32, 2, 5, 5, 8, 10].into_array(),
            validity,
            &mut ctx,
        )?
        .into_array();
        Ok(StructArray::from_fields(&[
            ("l", l),
            ("o", PrimitiveArray::from_iter(0..5_i32).into_array()),
        ])?
        .into_array())
    }

    fn lambda(param: &str, body: Expression) -> Lambda {
        Lambda::new([param], body).vortex_expect("lambda is valid")
    }

    /// `list_transform(l, x -> body)`.
    fn transform(body: Expression) -> Expression {
        list_transform(col("l"), lambda("x", body))
    }

    const EVERY_LIST_COLUMN: &[&str] = &[
        "l.offsets",
        "l.elements.a",
        "l.elements.f",
        "l.elements.g.offsets",
        "l.elements.g.elements.h",
        "l.elements.g.elements.i",
    ];

    /// A list read through `list_transform` lambdas reads only the element columns their bodies
    /// use, through nested lists too, whether or not the bodies can fail. Reading the list whole
    /// beside a lambda reads whole elements.
    #[rstest]
    #[case::field(transform(get_item("f", var("x"))), &["l.offsets", "l.elements.f"])]
    #[case::under_another_fn(
        list_sum(transform(get_item("f", var("x")))),
        &["l.offsets", "l.elements.f"],
    )]
    #[case::beside_another_column(
        pack([("t", transform(get_item("f", var("x")))), ("o", col("o"))], Nullability::NonNullable),
        &["l.offsets", "l.elements.f", "o"],
    )]
    #[case::nested(
        transform(list_transform(get_item("g", var("x")), lambda("y", get_item("h", var("y"))))),
        &["l.offsets", "l.elements.g.offsets", "l.elements.g.elements.h"],
    )]
    #[case::fallible(
        transform(checked_add(get_item("f", var("x")), lit(1_i32))),
        &["l.offsets", "l.elements.f"],
    )]
    #[case::nested_fallible(
        transform(list_transform(
            get_item("g", var("x")),
            lambda("y", checked_add(get_item("h", var("y")), lit(1_i32))),
        )),
        &["l.offsets", "l.elements.g.offsets", "l.elements.g.elements.h"],
    )]
    #[case::two_lambdas(
        pack(
            [("a", transform(get_item("a", var("x")))), ("f", transform(get_item("f", var("x"))))],
            Nullability::NonNullable,
        ),
        &["l.offsets", "l.elements.a", "l.elements.f"],
    )]
    #[case::same_lambda_twice(
        pack(
            [("a", transform(get_item("f", var("x")))), ("b", list_sum(transform(get_item("f", var("x")))))],
            Nullability::NonNullable,
        ),
        &["l.offsets", "l.elements.f"],
    )]
    #[case::beside_list_length(
        pack(
            [("t", transform(get_item("f", var("x")))), ("n", list_length(col("l")))],
            Nullability::NonNullable,
        ),
        &["l.offsets", "l.elements.f"],
    )]
    #[case::beside_the_whole_list(
        pack([("t", transform(get_item("f", var("x")))), ("l", col("l"))], Nullability::NonNullable),
        EVERY_LIST_COLUMN,
    )]
    #[tokio::test]
    async fn list_transform_reads_only_what_its_body_uses(
        #[case] expr: Expression,
        #[case] columns: &[&str],
        #[values(false, true)] nullable: bool,
        #[values(Mask::new_true(5), Mask::from_iter([false, true, false, true, true]))] mask: Mask,
    ) -> VortexResult<()> {
        let table = transform_table(nullable)?;
        let strategy = TableStrategy::new(chunked_elements_strategy(), chunked_elements_strategy())
            .with_list_layout();
        let (segments, layout, session) = write_layout(&strategy, table.clone()).await?;
        let requested = Arc::new(Mutex::new(Vec::new()));
        let source = Arc::new(RecordingSegmentSource {
            inner: segments,
            requested: Arc::clone(&requested),
        });
        let reader = layout.new_reader("".into(), source, &session, &LayoutReaderContext::new())?;

        let bound = expr.bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(0..5), &bound, MaskFuture::ready(mask.clone()))?
            .await?;
        let expected = table.filter(mask)?.apply(&expr)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);

        let mut segment_column = HashMap::new();
        segment_columns(&layout, "", &mut segment_column)?;
        let read = requested
            .lock()
            .iter()
            .map(|id| segment_column[id].as_str())
            .collect::<BTreeSet<_>>();
        let mut expected_read = columns.iter().copied().collect::<BTreeSet<_>>();
        if nullable {
            expected_read.insert("l.validity");
        }
        assert_eq!(read, expected_read);
        Ok(())
    }

    /// Read `expr` through the reader and over the array in memory, and check they agree, as
    /// values or as both failing.
    async fn assert_read_matches(
        array: &ArrayRef,
        strategy: &impl LayoutStrategy,
        expr: &Expression,
        row_range: Range<u64>,
        mask: Mask,
    ) -> VortexResult<()> {
        let (segments, layout, session) = write_layout(strategy, array.clone()).await?;
        let reader =
            layout.new_reader("".into(), segments, &session, &LayoutReaderContext::new())?;
        let mut ctx = session.create_execution_ctx();
        let bound = expr.bind(reader.dtype())?;
        let read =
            match reader.projection_evaluation(&row_range, &bound, MaskFuture::ready(mask.clone()))
            {
                Ok(read) => read
                    .await
                    .and_then(|read| read.execute::<RecursiveCanonical>(&mut ctx)),
                Err(err) => Err(err),
            };
        let expected = array
            .slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?
            .filter(mask)?
            .apply(expr)
            .and_then(|expected| expected.execute::<RecursiveCanonical>(&mut ctx));
        match (read, expected) {
            (Ok(read), Ok(expected)) => {
                assert_arrays_eq!(read.0.into_array(), expected.0.into_array(), &mut ctx)
            }
            (Err(_), Err(_)) => {}
            (read, expected) => panic!(
                "{expr} over {row_range:?}: read {:?}, expected {:?}",
                read.map(|read| read.0.into_array()),
                expected.map(|expected| expected.0.into_array()),
            ),
        }
        Ok(())
    }

    /// `[[1, 2], null over [MAX], [MAX, 3], [], null over [4, MAX], [5]]`.
    fn list_hiding_max() -> VortexResult<ArrayRef> {
        Ok(ListArray::try_new(
            buffer![1i32, 2, i32::MAX, i32::MAX, 3, 4, i32::MAX, 5].into_array(),
            buffer![0u32, 2, 3, 5, 5, 7, 8].into_array(),
            Validity::Array(
                BoolArray::from_iter([true, false, true, true, false, true]).into_array(),
            ),
            &mut SESSION.create_execution_ctx(),
        )?
        .into_array())
    }

    /// A body that can fail sees only the elements of the selected valid lists: `x + 1` overflows
    /// on the `MAX` under the null lists only if it reads them, and on the third list only when
    /// that list is selected.
    #[rstest]
    #[case::every_row(0..6, Mask::new_true(6))]
    #[case::skips_the_overflowing_list(0..6, Mask::from_iter([true, true, false, true, true, true]))]
    #[case::only_null_lists(1..5, Mask::from_iter([true, false, false, true]))]
    #[case::sub_range(3..6, Mask::new_true(3))]
    #[case::selects_the_overflowing_list(0..6, Mask::from_iter([false, false, true, false, false, true]))]
    #[tokio::test]
    async fn fallible_body_sees_only_selected_valid_lists(
        #[case] row_range: Range<u64>,
        #[case] mask: Mask,
        #[values(false, true)] chunked: bool,
    ) -> VortexResult<()> {
        let expr = list_transform(root(), lambda("x", checked_add(var("x"), lit(1_i32))));
        let list = list_hiding_max()?;
        if chunked {
            assert_read_matches(
                &list,
                &chunked_elements_list_strategy(),
                &expr,
                row_range,
                mask,
            )
            .await
        } else {
            assert_read_matches(&list, &flat_list_strategy(), &expr, row_range, mask).await
        }
    }

    /// Masked element reads skip element chunks no selected list references.
    #[tokio::test]
    async fn sparse_mask_skips_element_chunks_between_selected_lists() -> VortexResult<()> {
        let list = create_wider_list_array(false);
        let (segments, layout, session) =
            write_layout(&chunked_elements_list_strategy(), list.clone()).await?;
        let requested = Arc::new(Mutex::new(Vec::new()));
        let source = Arc::new(RecordingSegmentSource {
            inner: segments,
            requested: Arc::clone(&requested),
        });
        let reader = layout.new_reader("".into(), source, &session, &LayoutReaderContext::new())?;

        // Lists 0 and 4 hold elements [0, 2) and [8, 10); the chunks between hold only lists 1-3.
        let mask = Mask::from_iter([true, false, false, false, true]);
        let expr = root().bind(reader.dtype())?;
        let result = reader
            .projection_evaluation(&(0..5), &expr, MaskFuture::ready(mask.clone()))?
            .await?;
        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, list.filter(mask)?, &mut ctx);

        let mut segment_column = HashMap::new();
        segment_columns(&layout, "", &mut segment_column)?;
        let element_reads = requested
            .lock()
            .iter()
            .filter(|id| segment_column[*id] == ".elements")
            .count();
        assert_eq!(element_reads, 2);
        Ok(())
    }

    /// Random lists of `{a, b}` structs with garbage under their null lists, read through every
    /// kind of list use over random ranges and masks, match the same expressions in memory.
    #[tokio::test]
    async fn random_list_uses_match_memory() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(0);
        let exprs = [
            transform(get_item("a", var("x"))),
            transform(checked_add(
                get_item("a", var("x")),
                get_item("b", var("x")),
            )),
            pack(
                [
                    ("s", list_sum(transform(get_item("a", var("x"))))),
                    (
                        "t",
                        transform(checked_add(get_item("b", var("x")), lit(1_i32))),
                    ),
                ],
                Nullability::NonNullable,
            ),
            pack(
                [
                    (
                        "t",
                        transform(checked_add(get_item("a", var("x")), lit(1_i32))),
                    ),
                    ("n", list_length(col("l"))),
                    ("v", is_null(col("l"))),
                ],
                Nullability::NonNullable,
            ),
            pack(
                [
                    (
                        "t",
                        transform(checked_add(get_item("b", var("x")), lit(1_i32))),
                    ),
                    ("l", col("l")),
                ],
                Nullability::NonNullable,
            ),
        ];
        let strategy = TableStrategy::new(chunked_elements_strategy(), chunked_elements_strategy())
            .with_list_layout();
        for _ in 0..40 {
            let lists = rng.random_range(1..40usize);
            let table = random_struct_lists(&mut rng, lists)?;
            let start = rng.random_range(0..lists);
            let end = rng.random_range(start + 1..=lists);
            let mask = match rng.random_range(0..3u8) {
                0 => Mask::new_true(end - start),
                1 => Mask::from_iter((start..end).map(|_| rng.random_bool(0.5))),
                _ => Mask::from_iter((start..end).map(|_| rng.random_bool(0.1))),
            };
            for expr in &exprs {
                let range = u64::try_from(start)?..u64::try_from(end)?;
                assert_read_matches(&table, &strategy, expr, range, mask.clone()).await?;
            }
        }
        Ok(())
    }

    /// `{l: list<struct<a: i32, b: i32>>}` with up to four elements per list. Null lists hold
    /// `MAX`, which overflows any body adding to it; valid lists hold small values.
    fn random_struct_lists(rng: &mut StdRng, lists: usize) -> VortexResult<ArrayRef> {
        let mut offsets = vec![0u32];
        let mut valid = Vec::with_capacity(lists);
        let (mut field_a, mut field_b) = (Vec::new(), Vec::new());
        for _ in 0..lists {
            let is_valid = rng.random_bool(0.7);
            let len = rng.random_range(0..5u32);
            for _ in 0..len {
                let value = if is_valid {
                    rng.random_range(0..100)
                } else {
                    i32::MAX
                };
                field_a.push(value);
                field_b.push(value);
            }
            valid.push(is_valid);
            offsets.push(offsets[offsets.len() - 1] + len);
        }
        let elements = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter(field_a).into_array()),
            ("b", PrimitiveArray::from_iter(field_b).into_array()),
        ])?
        .into_array();
        let l = ListArray::try_new(
            elements,
            PrimitiveArray::from_iter(offsets).into_array(),
            Validity::Array(BoolArray::from_iter(valid).into_array()),
            &mut SESSION.create_execution_ctx(),
        )?
        .into_array();
        Ok(StructArray::from_fields(&[("l", l)])?.into_array())
    }
}
