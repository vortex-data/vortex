// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::not;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::list::ListLayout;
use crate::segments::SegmentSource;

type OptionalArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

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
        let elements = layout.elements().new_reader(
            format!("{name}.elements").into(),
            Arc::clone(&segment_source),
            &session,
            ctx,
        )?;
        let offsets = layout.offsets().new_reader(
            format!("{name}.offsets").into(),
            Arc::clone(&segment_source),
            &session,
            ctx,
        )?;
        let validity = layout
            .validity()
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

    /// Projection for [`ExprClass::Validity`] expressions (`is_null` / `is_not_null` of the list):
    /// reads only the validity child — synthesizing all-valid for a non-nullable list — and never
    /// touches the offsets or elements.
    fn project_validity(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
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
                Some(v) => Some(
                    v.projection_evaluation(&row_range, &root(), MaskFuture::ready(mask))?
                        .await?,
                ),
                None => None,
            };

            let validity = create_validity(validity_array, nullability).to_array(out_len);
            validity.apply(&rewritten)
        }
        .boxed())
    }

    /// Projection for [`ExprClass::Elements`] expressions (everything else): materializes the list
    /// (offsets + elements + validity) and applies the expression.
    fn project_elements(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        // Fire the offsets read before cloning so it overlaps the mask await below.
        let projection = ElementsProjection {
            reader: self.clone(),
            expr: expr.clone(),
            row_range: row_range.clone(),
            offsets: self.fetch_offsets(row_range)?,
        };

        Ok(async move {
            // Await the caller mask to decide the read shape. Offsets is already in flight and
            // overlaps this wait; for statically-resolved masks the await is free.
            let mask = mask.await?;
            let is_whole_chunk = projection.row_range.start == 0
                && projection.row_range.end == projection.reader.layout.row_count();

            if mask.all_true() && is_whole_chunk {
                projection.project_whole_chunk().await
            } else if mask.all_true() {
                projection.project_full_range().await
            } else {
                projection.project_sparse(mask).await
            }
        }
        .boxed())
    }

    /// Fire the offsets read for `row_range`. The offsets child has an extra entry, so reading
    /// `row_range` maps to offsets in `[row_range.start..row_range.end + 1)`.
    fn fetch_offsets(&self, row_range: &Range<u64>) -> VortexResult<ArrayFuture> {
        let offsets_range = row_range.start..(row_range.end + 1);
        let offsets_count = usize::try_from(offsets_range.end - offsets_range.start)?;
        self.offsets.projection_evaluation(
            &offsets_range,
            &root(),
            MaskFuture::new_true(offsets_count),
        )
    }
}

/// Read `offsets[0]` and `offsets[-1]` and return the elements-buffer range they describe.
fn calculate_elements_range(
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

/// Subtract `first` from every offset so the resulting offsets index into a sliced
/// `elements[first..]` buffer starting at zero. The constant array is cast to the offsets' dtype.
fn rebase_offsets(offsets: ArrayRef, first: u64) -> VortexResult<ArrayRef> {
    let constant = ConstantArray::new(first, offsets.len())
        .into_array()
        .cast(offsets.dtype().clone())?;
    offsets.binary(constant, Operator::Sub)
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

/// The deepest list child an expression needs, cheapest first.
///
/// Drives "fetch as little as possible": a projection/filter that only inspects the list's
/// null-ness needs the validity child; everything else needs the element values. The ordering
/// `Validity < Elements` lets us take the max over the operands of a compound expression.
// TODO: have `filter_evaluation` use this too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ExprClass {
    /// Only the list's validity is needed (`is_null` / `is_not_null` of the list itself).
    Validity,
    /// The element values are needed (everything else).
    Elements,
}

/// Classify `expr` by the deepest list child it touches, where `root()` is the list.
///
/// Only the exact shapes `is_null(root())` / `is_not_null(root())` (validity) are recognized. Every
/// other access to the list, including a bare `root()`, falls through to [`ExprClass::Elements`],
/// which is always correct.
fn classify(expr: &Expression) -> ExprClass {
    // `is_null(root())` / `is_not_null(root())` need only the list's own validity. Note this is
    // the list's null-ness, not the validity of some derived value, so the child must be `root()`.
    if (expr.is::<IsNull>() || expr.is::<IsNotNull>())
        && expr.children().len() == 1
        && is_root(expr.child(0))
    {
        return ExprClass::Validity;
    }

    // A bare reference to the list needs its elements.
    if is_root(expr) {
        return ExprClass::Elements;
    }

    // Otherwise the requirement is the max over the operands. Operands that never touch the list
    // (e.g. literals) contribute nothing, so an expression that never references `root()` is
    // treated as the cheapest class.
    expr.children()
        .iter()
        .map(classify)
        .max()
        .unwrap_or(ExprClass::Validity)
}

/// Rewrite a validity-class expression so it can be evaluated against the list's validity bool
/// array (`true` == valid row): `is_not_null(root())` becomes `root()` and `is_null(root())`
/// becomes `not(root())`. All other nodes are rebuilt with rewritten children.
fn rewrite_validity_expr(expr: &Expression) -> VortexResult<Expression> {
    if expr.is::<IsNotNull>() && expr.children().len() == 1 && is_root(expr.child(0)) {
        return Ok(root());
    }
    if expr.is::<IsNull>() && expr.children().len() == 1 && is_root(expr.child(0)) {
        return Ok(not(root()));
    }
    let children = expr
        .children()
        .iter()
        .map(rewrite_validity_expr)
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

/// Plan for fetching only the elements needed to materialize the kept list rows under a sparse
/// row mask, plus the offsets array we'll hand to `ListArray::try_new` for those kept rows.
///
/// When the row mask is sparse, the alternative (read full row_range, build full list, then
/// `array.filter(mask)`) wastes IO on elements that get thrown away. This plan tells the reader:
///
/// - which contiguous span of the elements buffer to fetch (`elements_range`),
/// - which positions inside that span belong to a kept row (`element_mask`),
/// - the offsets for the kept-row output, rebased to start at zero (`new_offsets`).
struct ScatterGather {
    /// Tightest absolute elements range covering all kept rows. Empty range when no rows kept.
    elements_range: Range<u64>,
    /// `element_mask.len() == elements_range.end - elements_range.start`. A bit is set iff its
    /// position in the elements buffer belongs to a kept list row.
    element_mask: Mask,
    /// Cumulative kept-list lengths starting at zero. `new_offsets.len() == kept_count + 1`.
    new_offsets: ArrayRef,
    /// Number of true bits in the input row mask. Read by unit tests only.
    #[cfg_attr(not(test), allow(dead_code))]
    kept_count: usize,
}

/// Walk the row mask and the (canonicalized) offsets to plan the elements fetch + output offsets
/// for the sparse-mask path of `projection_evaluation`. Single linear pass; no IO.
///
/// `offsets` is the offsets array we fetched for the full `row_range` (length n+1). `mask` is
/// the row-space mask (length n). Returns a plan suitable for handing the elements child a
/// bounded range + element-level mask, then constructing a kept-only `ListArray`.
// `usize::try_from` / `u64::try_from` are required by the macro arms whose `O` may be `u64` /
// `i64` (potentially fallible on 32-bit targets) but also expand to arms where `O` is `u8`,
// `u16`, etc. (where the conversion is trivially infallible). Suppress the resulting
// `unnecessary_fallible_conversions` lint from the latter arms — the uniform fallible form
// keeps the inner body identical across all expansions.
#[allow(clippy::unnecessary_fallible_conversions)]
fn compute_scatter_gather(
    offsets: &ArrayRef,
    mask: &Mask,
    session: &VortexSession,
) -> VortexResult<ScatterGather> {
    let kept_count = mask.true_count();
    let mut exec_ctx = session.create_execution_ctx();
    let prim_offsets = offsets.clone().execute::<PrimitiveArray>(&mut exec_ctx)?;
    let ptype = prim_offsets.ptype();

    if kept_count == 0 {
        // Empty result: no elements to fetch, new_offsets is a single zero.
        let new_offsets = vortex_array::match_each_integer_ptype!(ptype, |O| {
            Array::<Primitive>::new::<O>(
                Buffer::<O>::from(vec![O::default()]),
                Validity::NonNullable,
            )
            .into_array()
        });
        return Ok(ScatterGather {
            elements_range: 0..0,
            element_mask: Mask::new_false(0),
            new_offsets,
            kept_count: 0,
        });
    }

    vortex_array::match_each_integer_ptype!(ptype, |O| {
        compute_scatter_gather_typed::<O>(prim_offsets.as_slice::<O>(), mask, kept_count)
    })
}

fn compute_scatter_gather_typed<O>(
    offsets: &[O],
    mask: &Mask,
    kept_count: usize,
) -> VortexResult<ScatterGather>
where
    O: IntegerPType,
    usize: TryFrom<O>,
    VortexError: From<<usize as TryFrom<O>>::Error>,
{
    let mut new_off: Vec<O> = Vec::with_capacity(kept_count + 1);
    let mut element_slices: Vec<(usize, usize)> = Vec::with_capacity(kept_count);
    new_off.push(O::default());
    let mut cumulative: O = O::default();
    let mut range_start: Option<usize> = None;
    let mut range_end = 0usize;

    {
        let mut keep_row = |i: usize| -> VortexResult<()> {
            let start_offset = offsets[i];
            let end_offset = offsets[i + 1];
            cumulative += end_offset - start_offset;
            new_off.push(cumulative);

            let start = usize::try_from(start_offset)?;
            let end = usize::try_from(end_offset)?;
            if start < end {
                let start_base = *range_start.get_or_insert(start);
                element_slices.push((start - start_base, end - start_base));
                range_end = end;
            }
            Ok(())
        };

        // `mask.indices()` returns the set bit positions for `Values` masks; `AllTrue` is rare
        // here (caller checks density) but we handle it via fallback iteration.
        match mask.indices() {
            vortex_mask::AllOr::All => {
                for i in 0..mask.len() {
                    keep_row(i)?;
                }
            }
            vortex_mask::AllOr::None => {}
            vortex_mask::AllOr::Some(idxs) => {
                for &i in idxs {
                    keep_row(i)?;
                }
            }
        }
    }

    let range_start = range_start.unwrap_or(0);
    let element_mask = Mask::from_slices(range_end - range_start, element_slices);
    let new_offsets =
        Array::<Primitive>::new::<O>(Buffer::<O>::from(new_off), Validity::NonNullable)
            .into_array();

    Ok(ScatterGather {
        elements_range: u64::try_from(range_start)?..u64::try_from(range_end)?,
        element_mask,
        new_offsets,
        kept_count,
    })
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
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        self.offsets
            .register_splits(field_mask, split_range, splits)?;
        if let Some(validity) = &self.validity {
            validity.register_splits(field_mask, split_range, splits)?;
        }
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // All stats-based pruning should already be done upstream.
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        _mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        todo!()
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        // Read as little as possible based on which list children the expression needs.
        match classify(expr) {
            ExprClass::Validity => self.project_validity(row_range, expr, mask),
            ExprClass::Elements => self.project_elements(row_range, expr, mask),
        }
    }
}

/// Fetch the validity child for `row_range` under `mask`, yielding `None` for a non-nullable list
/// (which has no validity child).
fn fetch_validity(
    validity: Option<&LayoutReaderRef>,
    row_range: &Range<u64>,
    mask: MaskFuture,
) -> VortexResult<OptionalArrayFuture> {
    let fut = validity
        .map(|v| v.projection_evaluation(row_range, &root(), mask))
        .transpose()?;
    Ok(async move {
        match fut {
            Some(f) => f.await.map(Some),
            None => Ok(None),
        }
    }
    .boxed())
}

struct ListParts {
    elements: ArrayRef,
    offsets: ArrayRef,
    validity: Option<ArrayRef>,
}

/// Build the list array from its parts and apply the projection expression.
fn build_list(
    parts: ListParts,
    nullability: Nullability,
    expr: &Expression,
) -> VortexResult<ArrayRef> {
    let validity = create_validity(parts.validity, nullability);
    ListArray::try_new(parts.elements, parts.offsets, validity)?
        .into_array()
        .apply(expr)
}

struct ElementsProjection {
    reader: ListReader,
    expr: Expression,
    row_range: Range<u64>,
    offsets: ArrayFuture,
}

impl ElementsProjection {
    /// Path A1: whole-chunk read with an all-true mask. The elements bound is the whole elements
    /// buffer (`0..elements_row_count`) and `offsets[0] == 0` within a chunk, so we don't need to
    /// read offsets to know the bound and don't need to rebase. Fires elements + validity in
    /// parallel with the already-in-flight offsets — a single `try_join!` over all three children.
    async fn project_whole_chunk(self) -> VortexResult<ArrayRef> {
        let Self {
            reader,
            expr,
            row_range,
            offsets,
        } = self;
        let validity_row_count = usize::try_from(row_range.end - row_range.start)?;
        let elements_row_count = reader.elements.row_count();
        let elements_fut = reader.elements.projection_evaluation(
            &(0..elements_row_count),
            &root(),
            MaskFuture::new_true(usize::try_from(elements_row_count)?),
        )?;
        let validity_fut = fetch_validity(
            reader.validity.as_ref(),
            &row_range,
            MaskFuture::new_true(validity_row_count),
        )?;
        let (offsets, elements, validity) = try_join!(offsets, elements_fut, validity_fut)?;
        build_list(
            ListParts {
                elements,
                offsets,
                validity,
            },
            reader.layout.dtype().nullability(),
            &expr,
        )
    }

    /// Path A2: partial range with an all-true mask. The elements bound is
    /// `offsets[a]..offsets[b]`, so we await offsets before firing the elements read and rebase the
    /// offsets to start at zero.
    async fn project_full_range(self) -> VortexResult<ArrayRef> {
        let Self {
            reader,
            expr,
            row_range,
            offsets,
        } = self;
        let offsets = offsets.await?;
        let elements_range = calculate_elements_range(&offsets, &reader.session)?;
        let rebased_offsets = rebase_offsets(offsets, elements_range.start)?;
        let elements_len = elements_range.end - elements_range.start;
        let validity_row_count = usize::try_from(row_range.end - row_range.start)?;

        let elements_fut = reader.elements.projection_evaluation(
            &elements_range,
            &root(),
            MaskFuture::new_true(usize::try_from(elements_len)?),
        )?;
        let validity_fut = fetch_validity(
            reader.validity.as_ref(),
            &row_range,
            MaskFuture::new_true(validity_row_count),
        )?;
        let (elements, validity) = try_join!(elements_fut, validity_fut)?;
        build_list(
            ListParts {
                elements,
                offsets: rebased_offsets,
                validity,
            },
            reader.layout.dtype().nullability(),
            &expr,
        )
    }

    /// Path B: sparse mask. Bound the elements fetch to the tightest range covering the kept rows
    /// and pass an element-level mask so the elements child only materializes kept-row positions;
    /// validity is fetched for the kept rows by pushing the caller mask down directly.
    async fn project_sparse(self, mask: Mask) -> VortexResult<ArrayRef> {
        let Self {
            reader,
            expr,
            row_range,
            offsets,
        } = self;
        let validity_fut = fetch_validity(
            reader.validity.as_ref(),
            &row_range,
            MaskFuture::ready(mask.clone()),
        )?;
        let offsets = offsets.await?;
        let sg = compute_scatter_gather(&offsets, &mask, &reader.session)?;
        let elements_fut = reader.elements.projection_evaluation(
            &sg.elements_range,
            &root(),
            MaskFuture::ready(sg.element_mask),
        )?;
        let (elements, validity) = try_join!(elements_fut, validity_fut)?;
        build_list(
            ListParts {
                elements,
                offsets: sg.new_offsets,
                validity,
            },
            reader.layout.dtype().nullability(),
            &expr,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::expr::eq;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_buffer::buffer;

    use super::*;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::list::writer::ListLayoutStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    /// `classify` keys off the deepest list child an expression touches; `Elements` is the
    /// always-correct default for anything not specifically recognized.
    #[rstest]
    // `is_null` / `is_not_null` of the list itself need only validity.
    #[case::is_null(is_null(root()), ExprClass::Validity)]
    #[case::is_not_null(is_not_null(root()), ExprClass::Validity)]
    // Compound over validity-only operands stays validity.
    #[case::not_is_null(not(is_null(root())), ExprClass::Validity)]
    // A list-independent (constant) expression falls to the cheapest class.
    #[case::constant(lit(5), ExprClass::Validity)]
    // A bare list reference needs the elements.
    #[case::bare_root(root(), ExprClass::Elements)]
    // Any other fn over the list needs the elements.
    #[case::not_root(not(root()), ExprClass::Elements)]
    // `is_null` only short-circuits to validity when its argument is the list itself.
    #[case::is_null_of_derived(is_null(not(root())), ExprClass::Elements)]
    // Max over operands: validity + elements => elements.
    #[case::validity_and_elements(eq(is_null(root()), root()), ExprClass::Elements)]
    fn classify_expr_class(#[case] expr: Expression, #[case] expected: ExprClass) {
        assert_eq!(classify(&expr), expected);
    }

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
        let (segments, layout) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let not_null = reader
            .projection_evaluation(&(0..3), &is_not_null(root()), MaskFuture::new_true(3))?
            .await?;
        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(not_null, BoolArray::from_iter(valid.clone()), &mut exec_ctx);

        let is_null_res = reader
            .projection_evaluation(&(0..3), &is_null(root()), MaskFuture::new_true(3))?
            .await?;
        assert_arrays_eq!(
            is_null_res,
            BoolArray::from_iter(valid.iter().map(|v| !v).collect::<Vec<_>>()),
            &mut exec_ctx
        );

        Ok(())
    }

    fn flat_list_strategy() -> ListLayoutStrategy {
        ListLayoutStrategy::default()
    }

    async fn write_layout<S: LayoutStrategy>(
        strategy: &S,
        array: ArrayRef,
    ) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
        let segments = Arc::new(TestSegments::default());
        let segments_ref: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let layout = strategy
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await?;
        Ok((segments_ref, layout))
    }

    fn materialize_u32_array(array: ArrayRef) -> Vec<u32> {
        let mut ctx = SESSION.create_execution_ctx();
        array
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
            .as_slice::<u32>()
            .to_vec()
    }

    #[rstest]
    #[case::full(buffer![0u32, 2, 5, 5].into_array(), 0..5)]
    #[case::partial_slice(buffer![2u32, 5, 5, 8].into_array(), 2..8)]
    #[case::single_offset_is_empty(buffer![7u32].into_array(), 7..7)]
    #[case::u64_offsets(buffer![10u64, 12, 15, 15].into_array(), 10..15)]
    fn test_calculate_elements_range(
        #[case] offsets: ArrayRef,
        #[case] expected: Range<u64>,
    ) -> VortexResult<()> {
        assert_eq!(calculate_elements_range(&offsets, &SESSION)?, expected);
        Ok(())
    }

    #[test]
    fn calculate_elements_range_empty_offsets() -> VortexResult<()> {
        let offsets = PrimitiveArray::empty::<u32>(NonNullable).into_array();
        assert_eq!(calculate_elements_range(&offsets, &SESSION)?, 0..0);
        Ok(())
    }

    #[rstest]
    #[case::first_zero_is_identity(buffer![0u32, 2, 5, 5].into_array(), 0, vec![0, 2, 5, 5])]
    #[case::subtracts_first(buffer![3u32, 5, 8].into_array(), 3, vec![0, 2, 5])]
    fn test_rebase_offsets(
        #[case] offsets: ArrayRef,
        #[case] first: u64,
        #[case] expected: Vec<u32>,
    ) -> VortexResult<()> {
        let rebased = rebase_offsets(offsets, first)?;
        assert_eq!(materialize_u32_array(rebased), expected);
        Ok(())
    }

    // ---- compute_scatter_gather --------------------------------------------------------------

    /// Run `compute_scatter_gather` and unwrap the three derived fields plus the kept count.
    /// Returns the raw `new_offsets` ArrayRef so callers with non-u32 offsets can materialize
    /// the ptype themselves.
    fn run_scatter_gather(
        offsets: ArrayRef,
        mask: Mask,
    ) -> VortexResult<(Range<u64>, Vec<bool>, ArrayRef, usize)> {
        let sg = compute_scatter_gather(&offsets, &mask, &SESSION)?;
        let element_mask_bits: Vec<bool> = (0..sg.element_mask.len())
            .map(|i| sg.element_mask.value(i))
            .collect();
        Ok((
            sg.elements_range,
            element_mask_bits,
            sg.new_offsets,
            sg.kept_count,
        ))
    }

    /// Source layout for these tests: 5 lists with offsets `[0, 2, 5, 5, 8, 10]`, i.e.
    /// lengths `[2, 3, 0, 3, 2]`. Element positions for list i are `offsets[i]..offsets[i+1]`.
    fn five_list_offsets() -> ArrayRef {
        buffer![0u32, 2, 5, 5, 8, 10].into_array()
    }

    #[test]
    fn scatter_gather_single_middle_row() -> VortexResult<()> {
        // Keep only list 1 (positions 2..5).
        let mask = Mask::from_iter([false, true, false, false, false]);
        let (range, elem_mask, new_off, kept) = run_scatter_gather(five_list_offsets(), mask)?;
        assert_eq!(range, 2..5);
        assert_eq!(elem_mask, vec![true; 3]); // entire bounded range is the kept span
        assert_eq!(materialize_u32_array(new_off), vec![0, 3]);
        assert_eq!(kept, 1);
        Ok(())
    }

    #[test]
    fn scatter_gather_two_adjacent_rows() -> VortexResult<()> {
        // Keep lists 1 and 2 (positions 2..5 and 5..5 — second is empty).
        let mask = Mask::from_iter([false, true, true, false, false]);
        let (range, elem_mask, new_off, kept) = run_scatter_gather(five_list_offsets(), mask)?;
        assert_eq!(range, 2..5);
        assert_eq!(elem_mask, vec![true; 3]);
        assert_eq!(materialize_u32_array(new_off), vec![0, 3, 3]); // second kept row has length 0
        assert_eq!(kept, 2);
        Ok(())
    }

    #[test]
    fn scatter_gather_two_far_apart_rows() -> VortexResult<()> {
        // Keep lists 0 and 3 (positions 0..2 and 5..8). Element mask must skip position 2..5.
        let mask = Mask::from_iter([true, false, false, true, false]);
        let (range, elem_mask, new_off, kept) = run_scatter_gather(five_list_offsets(), mask)?;
        assert_eq!(range, 0..8);
        // positions 0..2 and 5..8 set, 2..5 unset.
        assert_eq!(
            elem_mask,
            vec![true, true, false, false, false, true, true, true]
        );
        assert_eq!(materialize_u32_array(new_off), vec![0, 2, 5]); // lengths 2 and 3
        assert_eq!(kept, 2);
        Ok(())
    }

    #[test]
    fn scatter_gather_at_boundaries() -> VortexResult<()> {
        // Keep first and last list (positions 0..2 and 8..10).
        let mask = Mask::from_iter([true, false, false, false, true]);
        let (range, elem_mask, new_off, kept) = run_scatter_gather(five_list_offsets(), mask)?;
        assert_eq!(range, 0..10);
        let mut expected = vec![false; 10];
        expected[0] = true;
        expected[1] = true;
        expected[8] = true;
        expected[9] = true;
        assert_eq!(elem_mask, expected);
        assert_eq!(materialize_u32_array(new_off), vec![0, 2, 4]);
        assert_eq!(kept, 2);
        Ok(())
    }

    #[test]
    fn scatter_gather_empty_mask_returns_empty_plan() -> VortexResult<()> {
        let mask = Mask::new_false(5);
        let (range, elem_mask, new_off, kept) = run_scatter_gather(five_list_offsets(), mask)?;
        assert_eq!(range, 0..0);
        assert!(elem_mask.is_empty());
        // single zero, ready to be a 0-row ListArray's offsets (offsets.len() - 1 == 0 rows)
        assert_eq!(materialize_u32_array(new_off), vec![0]);
        assert_eq!(kept, 0);
        Ok(())
    }

    #[test]
    fn scatter_gather_kept_row_is_empty_list() -> VortexResult<()> {
        // Keep only list 2, which has length 0 (offsets[2] == offsets[3] == 5).
        let mask = Mask::from_iter([false, false, true, false, false]);
        let (range, elem_mask, new_off, kept) = run_scatter_gather(five_list_offsets(), mask)?;
        assert_eq!(range, 0..0);
        assert!(elem_mask.is_empty());
        assert_eq!(materialize_u32_array(new_off), vec![0, 0]);
        assert_eq!(kept, 1);
        Ok(())
    }

    #[test]
    fn scatter_gather_ignores_empty_kept_boundary_rows() -> VortexResult<()> {
        // The first and last kept rows are empty. The read range should be anchored to the one
        // non-empty kept row, not widened across skipped rows.
        let offsets = buffer![0u32, 0, 100, 102, 200, 200].into_array();
        let mask = Mask::from_iter([true, false, true, false, true]);
        let (range, elem_mask, new_off, kept) = run_scatter_gather(offsets, mask)?;
        assert_eq!(range, 100..102);
        assert_eq!(elem_mask, vec![true, true]);
        assert_eq!(materialize_u32_array(new_off), vec![0, 0, 2, 2]);
        assert_eq!(kept, 3);
        Ok(())
    }

    #[test]
    fn scatter_gather_u64_offsets() -> VortexResult<()> {
        // Verify the ptype-dispatch path works for u64 offsets, not just u32.
        let offsets = buffer![0u64, 3, 7, 7, 12].into_array();
        let mask = Mask::from_iter([false, true, false, true]);
        let (range, elem_mask, new_off, kept) = run_scatter_gather(offsets, mask)?;
        assert_eq!(range, 3..12);
        // positions 3..7 (4 bits) and 7..12 (5 bits) — middle "gap" at 7..7 is zero-width.
        assert_eq!(elem_mask, vec![true; 9]);
        // Walk the new_offsets slice as u64.
        let mut ctx = SESSION.create_execution_ctx();
        let new_off_prim = new_off.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(new_off_prim.as_slice::<u64>(), &[0u64, 4, 9]);
        assert_eq!(kept, 2);
        Ok(())
    }

    fn create_basic_list_array(nullable: bool) -> ArrayRef {
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array())
        } else {
            Validity::NonNullable
        };

        ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 4, 5].into_array(),
            validity,
        )
        .expect("array is valid")
        .into_array()
    }

    #[tokio::test]
    async fn fetch_offsets_includes_extra_endpoint() -> VortexResult<()> {
        let list = create_basic_list_array(false);

        let (segments, layout) = write_layout(&flat_list_strategy(), list).await?;
        let ctx = LayoutReaderContext::new();
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;
        let reader = reader
            .as_any()
            .downcast_ref::<ListReader>()
            .expect("ListReader");

        let offsets = reader.fetch_offsets(&(1..3))?.await?;
        assert_eq!(materialize_u32_array(offsets), vec![2, 4, 5]);

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
        let (segments, layout) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let result = reader
            .projection_evaluation(&row_range, &root(), MaskFuture::new_true(len))?
            .await?;

        let expected =
            list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?;
        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_evaluation_applies_mask() -> VortexResult<()> {
        let list = create_basic_list_array(false);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let mask = Mask::from_iter([true, false, true]);
        let result = reader
            .projection_evaluation(&(0..3), &root(), MaskFuture::ready(mask.clone()))?
            .await?;

        let expected = list.filter(mask)?;
        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    /// Build a list with 5 rows and lengths [2, 3, 0, 3, 2]. Mirrors `five_list_offsets()`.
    fn create_wider_list_array(nullable: bool) -> ArrayRef {
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, true, false, true, true]).into_array())
        } else {
            Validity::NonNullable
        };
        ListArray::try_new(
            buffer![10i32, 11, 20, 21, 22, 30, 31, 32, 40, 41].into_array(),
            buffer![0u32, 2, 5, 5, 8, 10].into_array(),
            validity,
        )
        .expect("array is valid")
        .into_array()
    }

    #[rstest]
    // Single bit set far from start — exercises sparse path with tight elements range.
    #[case::single_middle(Mask::from_iter([false, false, false, true, false]), false)]
    // Two far-apart rows — element_mask has a gap between kept spans.
    #[case::two_far_apart(Mask::from_iter([true, false, false, true, false]), false)]
    // Boundary rows — first and last list.
    #[case::boundaries(Mask::from_iter([true, false, false, false, true]), false)]
    // Kept row is the empty list (zero-width span).
    #[case::kept_empty_row(Mask::from_iter([false, false, true, false, false]), false)]
    // Sparse with nullable elements/validity child — exercises validity push-down.
    #[case::sparse_nullable(Mask::from_iter([true, false, true, false, true]), true)]
    // No rows kept — degenerate empty output.
    #[case::all_false(Mask::new_false(5), false)]
    #[tokio::test]
    async fn projection_evaluation_sparse_mask_round_trips(
        #[case] mask: Mask,
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let list = create_wider_list_array(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let result = reader
            .projection_evaluation(&(0..5), &root(), MaskFuture::ready(mask.clone()))?
            .await?;

        let expected = list.filter(mask)?;
        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }
}
