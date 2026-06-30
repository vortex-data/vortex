// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::not;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::fixed_size_list::FixedSizeListLayout;
use crate::segments::SegmentSource;

type OptionalArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

#[derive(Clone)]
pub(super) struct FixedSizeListReader {
    layout: FixedSizeListLayout,
    name: Arc<str>,
    session: VortexSession,
    elements: LayoutReaderRef,
    validity: Option<LayoutReaderRef>,
}

impl FixedSizeListReader {
    pub(super) fn try_new(
        layout: FixedSizeListLayout,
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
            validity,
        })
    }

    fn project_validity(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let validity_reader = self.validity.clone();
        let nullability = self.layout.dtype().nullability();
        let row_range = row_range.clone();
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

            create_validity(validity_array, nullability)
                .to_array(out_len)
                .apply(&rewritten)
        }
        .boxed())
    }

    fn project_elements(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let projection = ElementsProjection {
            reader: self.clone(),
            expr: expr.clone(),
            row_range: row_range.clone(),
        };

        Ok(async move {
            let mask = mask.await?;
            if mask.all_true() {
                projection.project_full_range().await
            } else {
                projection.project_sparse(mask).await
            }
        }
        .boxed())
    }
}

impl LayoutReader for FixedSizeListReader {
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
        self.elements.register_splits(
            field_mask,
            &element_split_range(split_range, self.layout.list_size())?,
            splits,
        )?;
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
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
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

            let predicate = reader
                .projection_evaluation(&row_range, &expr, MaskFuture::ready(mask.clone()))?
                .await?;
            let mut ctx = session.create_execution_ctx();
            let predicate_mask = predicate.null_as_false().execute(&mut ctx)?;

            if mask.all_true() {
                Ok(predicate_mask)
            } else {
                Ok(mask.intersect_by_rank(&predicate_mask))
            }
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        match classify(expr) {
            ExprClass::Validity => self.project_validity(row_range, expr, mask),
            ExprClass::Elements => self.project_elements(row_range, expr, mask),
        }
    }
}

fn element_split_range(split_range: &SplitRange, list_size: u32) -> VortexResult<SplitRange> {
    let list_size = u64::from(list_size);
    let row_range = element_range(split_range.row_range(), list_size)?;
    let row_offset = split_range
        .row_offset()
        .checked_mul(list_size)
        .ok_or_else(|| vortex_err!("fixed-size-list split offset overflow"))?;
    SplitRange::try_new(row_offset, row_range)
}

fn element_range(row_range: &Range<u64>, list_size: u64) -> VortexResult<Range<u64>> {
    let start = row_range
        .start
        .checked_mul(list_size)
        .ok_or_else(|| vortex_err!("fixed-size-list element range overflow"))?;
    let end = row_range
        .end
        .checked_mul(list_size)
        .ok_or_else(|| vortex_err!("fixed-size-list element range overflow"))?;
    Ok(start..end)
}

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

fn create_validity(validity_array: Option<ArrayRef>, nullability: Nullability) -> Validity {
    match validity_array {
        Some(arr) => Validity::Array(arr),
        None => match nullability {
            Nullability::Nullable => Validity::AllValid,
            Nullability::NonNullable => Validity::NonNullable,
        },
    }
}

fn build_fixed_size_list(
    elements: ArrayRef,
    validity_array: Option<ArrayRef>,
    dtype: &DType,
    len: usize,
    expr: &Expression,
) -> VortexResult<ArrayRef> {
    let DType::FixedSizeList(_, list_size, nullability) = dtype else {
        return Err(vortex_err!(
            "FixedSizeListLayout requires FixedSizeList dtype, got {dtype}"
        ));
    };
    let validity = create_validity(validity_array, *nullability);
    FixedSizeListArray::try_new(elements, *list_size, validity, len)?
        .into_array()
        .apply(expr)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ExprClass {
    Validity,
    Elements,
}

fn classify(expr: &Expression) -> ExprClass {
    if (expr.is::<IsNull>() || expr.is::<IsNotNull>())
        && expr.children().len() == 1
        && is_root(expr.child(0))
    {
        return ExprClass::Validity;
    }

    if is_root(expr) {
        return ExprClass::Elements;
    }

    expr.children()
        .iter()
        .map(classify)
        .max()
        .unwrap_or(ExprClass::Validity)
}

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

struct SparseElementPlan {
    elements_range: Range<u64>,
    element_mask: Mask,
    kept_count: usize,
}

fn sparse_element_plan(
    row_range: &Range<u64>,
    mask: &Mask,
    list_size: u64,
) -> VortexResult<SparseElementPlan> {
    let kept_count = mask.true_count();
    if kept_count == 0 || list_size == 0 {
        return Ok(SparseElementPlan {
            elements_range: 0..0,
            element_mask: Mask::new_true(0),
            kept_count,
        });
    }

    let first = mask
        .first()
        .ok_or_else(|| vortex_err!("sparse mask has no true values"))?;
    let last = mask
        .last()
        .ok_or_else(|| vortex_err!("sparse mask has no true values"))?;
    let first_row = row_range
        .start
        .checked_add(u64::try_from(first)?)
        .ok_or_else(|| vortex_err!("fixed-size-list row range overflow"))?;
    let last_row_exclusive = row_range
        .start
        .checked_add(u64::try_from(last + 1)?)
        .ok_or_else(|| vortex_err!("fixed-size-list row range overflow"))?;
    let elements_range = element_range(&(first_row..last_row_exclusive), list_size)?;
    let element_mask_len = usize::try_from(elements_range.end - elements_range.start)?;

    let list_size_usize = usize::try_from(list_size)?;
    let mut element_slices = Vec::with_capacity(kept_count);
    match mask.indices() {
        AllOr::All => {
            return Ok(SparseElementPlan {
                elements_range,
                element_mask: Mask::new_true(element_mask_len),
                kept_count,
            });
        }
        AllOr::None => {}
        AllOr::Some(indices) => {
            for &idx in indices {
                let relative = idx - first;
                let start = relative
                    .checked_mul(list_size_usize)
                    .ok_or_else(|| vortex_err!("fixed-size-list element mask overflow"))?;
                element_slices.push((start, start + list_size_usize));
            }
        }
    }

    Ok(SparseElementPlan {
        elements_range,
        element_mask: Mask::from_slices(element_mask_len, element_slices),
        kept_count,
    })
}

struct ElementsProjection {
    reader: FixedSizeListReader,
    expr: Expression,
    row_range: Range<u64>,
}

impl ElementsProjection {
    async fn project_full_range(self) -> VortexResult<ArrayRef> {
        let Self {
            reader,
            expr,
            row_range,
        } = self;
        let len = usize::try_from(row_range.end - row_range.start)?;
        let list_size = u64::from(reader.layout.list_size());
        let elements_range = element_range(&row_range, list_size)?;
        let elements_len = usize::try_from(elements_range.end - elements_range.start)?;
        let elements_fut = reader.elements.projection_evaluation(
            &elements_range,
            &root(),
            MaskFuture::new_true(elements_len),
        )?;
        let validity_fut = fetch_validity(
            reader.validity.as_ref(),
            &row_range,
            MaskFuture::new_true(len),
        )?;
        let (elements, validity) = try_join!(elements_fut, validity_fut)?;
        build_fixed_size_list(elements, validity, reader.layout.dtype(), len, &expr)
    }

    async fn project_sparse(self, mask: Mask) -> VortexResult<ArrayRef> {
        let Self {
            reader,
            expr,
            row_range,
        } = self;
        let plan = sparse_element_plan(&row_range, &mask, u64::from(reader.layout.list_size()))?;
        let elements_fut = reader.elements.projection_evaluation(
            &plan.elements_range,
            &root(),
            MaskFuture::ready(plan.element_mask),
        )?;
        let validity_fut = fetch_validity(
            reader.validity.as_ref(),
            &row_range,
            MaskFuture::ready(mask),
        )?;
        let (elements, validity) = try_join!(elements_fut, validity_fut)?;
        build_fixed_size_list(
            elements,
            validity,
            reader.layout.dtype(),
            plan.kept_count,
            &expr,
        )
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::validity::Validity;

    use super::*;
    use crate::LayoutStrategy;
    use crate::layouts::fixed_size_list::writer::FixedSizeListLayoutStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    async fn write_layout(
        array: ArrayRef,
    ) -> VortexResult<(Arc<dyn SegmentSource>, crate::LayoutRef)> {
        let segments = Arc::new(TestSegments::default());
        let segments_ref: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let layout = FixedSizeListLayoutStrategy::default()
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await?;
        Ok((segments_ref, layout))
    }

    fn create_fsl(nullable: bool) -> ArrayRef {
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array())
        } else {
            Validity::NonNullable
        };
        FixedSizeListArray::new(
            PrimitiveArray::from_iter(0i32..8).into_array(),
            2,
            validity,
            4,
        )
        .into_array()
    }

    #[rstest]
    #[case::non_nullable(false)]
    #[case::nullable(true)]
    #[tokio::test]
    async fn projection_full_range(#[case] nullable: bool) -> VortexResult<()> {
        let fsl = create_fsl(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let result = reader
            .projection_evaluation(&(0..4), &root(), MaskFuture::new_true(4))?
            .await?;

        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, fsl, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_partial_range() -> VortexResult<()> {
        let fsl = create_fsl(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let result = reader
            .projection_evaluation(&(1..4), &root(), MaskFuture::new_true(3))?
            .await?;
        let expected = fsl.slice(1..4)?;

        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_sparse_mask() -> VortexResult<()> {
        let fsl = create_fsl(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;
        let mask = Mask::from_iter([true, false, false, true]);

        let result = reader
            .projection_evaluation(&(0..4), &root(), MaskFuture::ready(mask.clone()))?
            .await?;
        let expected = fsl.filter(mask)?;

        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_degenerate_list_size_zero() -> VortexResult<()> {
        let fsl = FixedSizeListArray::new(
            PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
            0,
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
            3,
        )
        .into_array();
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;
        let mask = Mask::from_iter([false, true, true]);

        let result = reader
            .projection_evaluation(&(0..3), &root(), MaskFuture::ready(mask.clone()))?
            .await?;
        let expected = fsl.filter(mask)?;

        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[rstest]
    #[case::nullable(true, vec![true, false, true, true])]
    #[case::non_nullable(false, vec![true, true, true, true])]
    #[tokio::test]
    async fn projection_validity_class(
        #[case] nullable: bool,
        #[case] valid: Vec<bool>,
    ) -> VortexResult<()> {
        let fsl = create_fsl(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let not_null = reader
            .projection_evaluation(&(0..4), &is_not_null(root()), MaskFuture::new_true(4))?
            .await?;
        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(not_null, BoolArray::from_iter(valid.clone()), &mut exec_ctx);

        let null = reader
            .projection_evaluation(&(0..4), &is_null(root()), MaskFuture::new_true(4))?
            .await?;
        assert_arrays_eq!(
            null,
            BoolArray::from_iter(valid.iter().map(|v| !v).collect::<Vec<_>>()),
            &mut exec_ctx
        );
        Ok(())
    }
}
