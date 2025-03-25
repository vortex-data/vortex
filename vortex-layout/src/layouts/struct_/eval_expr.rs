use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use futures::future::{BoxFuture, try_join_all};
use futures::stream::FuturesOrdered;
use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_expr::ExprRef;
use vortex_expr::transform::partition::PartitionedExpr;
use vortex_mask::Mask;

use crate::layouts::struct_::reader::StructReader;
use crate::{ArrayEvaluation, ExprEvaluator, LayoutReader, MaskEvaluation, MaskFuture};

#[async_trait]
impl ExprEvaluator for StructReader {
    fn evaluate_expr2(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<Option<ArrayRef>>>> {
        // Partition the expression into expressions that can be evaluated over individual fields
        let partitioned = self.partition_expr(expr.clone());

        // Short-circuit if there is only one partition
        if partitioned.partition_names.len() == 1 {
            return self.child(&partitioned.partition_names[0])?.evaluate_expr2(
                row_range,
                &partitioned.partitions[0],
                mask,
            );
        }

        // Construct readers for each child.
        let field_futures: Vec<_> = partitioned
            .partition_names
            .iter()
            .zip_eq(partitioned.partitions.iter())
            .map(|(name, expr)| {
                self.child(name)?
                    .evaluate_expr2(row_range, expr, mask.clone())
            })
            .try_collect()?;

        let name = self.layout().name().to_string();

        Ok(Box::pin(async move {
            let row_count = mask.await?.true_count();
            if row_count == 0 {
                // Short-circuit if the mask is all false
                return Ok(None);
            }

            let arrays = try_join_all(field_futures)
                .await?
                .into_iter()
                .zip(&*partitioned.partition_names.clone())
                .map(|(a, field_name)| {
                    if a.is_none() {
                        vortex_panic!(
                            "Layout {} child {} incorrectly returned None for non-empty mask",
                            name,
                            field_name,
                        )
                    }
                    a.vortex_expect("Layout incorrectly returned empty array for non-empty mask")
                })
                .collect::<Vec<_>>();
            assert!(
                arrays.iter().all(|a| a.len() == row_count),
                "Struct fields returned arrays of incorrect length"
            );

            let root_scope = StructArray::try_new(
                partitioned.partition_names.clone(),
                arrays,
                row_count,
                Validity::NonNullable,
            )?
            .into_array();

            Ok(Some(partitioned.root.evaluate(&root_scope)?))
        }))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        // Partition the expression into expressions that can be evaluated over individual fields
        let partitioned = self.partition_expr(expr.clone());

        // Short-circuit if there is only one partition
        if partitioned.partition_names.len() == 1 {
            return self
                .child(&partitioned.partition_names[0])?
                .filter_evaluation(row_range, &partitioned.partitions[0]);
        }

        // TODO(ngates): for any partition that returns a boolean, we can use a mask evaluation.

        // Construct evaluations for each child.
        let field_evals: Vec<_> = partitioned
            .partition_names
            .iter()
            .zip_eq(partitioned.partitions.iter())
            .map(|(name, expr)| self.child(name)?.projection_evaluation(row_range, expr))
            .try_collect()?;

        Ok(Box::new(StructMaskEvaluation {
            partitioned,
            field_evals,
        }))
    }

    fn projection_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        todo!()
    }
}

struct StructMaskEvaluation {
    partitioned: Arc<PartitionedExpr>,
    field_evals: Vec<Box<dyn ArrayEvaluation>>,
}

#[async_trait]
impl MaskEvaluation for StructMaskEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        let field_arrays: Vec<_> = FuturesOrdered::from_iter(
            self.field_evals
                .iter()
                .map(|eval| eval.invoke(Mask::new_true(mask.len()))),
        )
        .try_collect()
        .await?;

        let root_scope = StructArray::try_new(
            self.partitioned.partition_names.clone(),
            field_arrays,
            mask.len(),
            Validity::NonNullable,
        )?
        .into_array();

        Mask::try_from(self.partitioned.root.evaluate(&root_scope)?.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use rstest::{fixture, rstest};
    use vortex_array::arrays::StructArray;
    use vortex_array::{Array, ArrayContext, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability, StructDType};
    use vortex_error::VortexUnwrap;
    use vortex_expr::{get_item, gt, ident, pack};
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::layouts::struct_::writer::StructLayoutWriter;
    use crate::segments::AsyncSegmentReader;
    use crate::segments::test::TestSegments;
    use crate::writer::LayoutWriterExt;
    use crate::{Layout, RowMask};

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn struct_layout() -> (ArrayContext, Arc<dyn AsyncSegmentReader>, Layout) {
        let ctx = ArrayContext::empty();
        let mut segments = TestSegments::default();

        let layout = StructLayoutWriter::try_new(
            DType::Struct(
                Arc::new(StructDType::new(
                    vec!["a".into(), "b".into(), "c".into()].into(),
                    vec![I32.into(), I32.into(), I32.into()],
                )),
                Nullability::NonNullable,
            ),
            vec![
                Box::new(FlatLayoutWriter::new(
                    ctx.clone(),
                    I32.into(),
                    Default::default(),
                )),
                Box::new(FlatLayoutWriter::new(
                    ctx.clone(),
                    I32.into(),
                    Default::default(),
                )),
                Box::new(FlatLayoutWriter::new(
                    ctx.clone(),
                    I32.into(),
                    Default::default(),
                )),
            ],
        )
        .vortex_unwrap()
        .push_all(
            &mut segments,
            [Ok(StructArray::from_fields(
                [
                    ("a", buffer![7, 2, 3].into_array()),
                    ("b", buffer![4, 5, 6].into_array()),
                    ("c", buffer![4, 5, 6].into_array()),
                ]
                .as_slice(),
            )
            .unwrap()
            .into_array())],
        )
        .unwrap();
        (ctx, Arc::new(segments), layout)
    }

    #[rstest]
    fn test_struct_layout(
        #[from(struct_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn AsyncSegmentReader>,
            Layout,
        ),
    ) {
        let reader = layout.reader(segments, ctx).unwrap();
        let expr = gt(get_item("a", ident()), get_item("b", ident()));
        let result =
            block_on(reader.evaluate_expr(RowMask::new_valid_between(0, 3), expr)).unwrap();
        assert_eq!(
            vec![true, false, false],
            result
                .to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn test_struct_layout_row_mask(
        #[from(struct_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn AsyncSegmentReader>,
            Layout,
        ),
    ) {
        let reader = layout.reader(segments, ctx).unwrap();
        let expr = gt(get_item("a", ident()), get_item("b", ident()));
        let result = block_on(reader.evaluate_expr(
            // Take rows 0 and 1, skip row 2, and anything after that
            RowMask::new(Mask::from_iter([true, true, false]), 0),
            expr,
        ))
        .unwrap();

        assert_eq!(result.len(), 2);

        assert_eq!(
            vec![true, false],
            result
                .to_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn test_struct_layout_select(
        #[from(struct_layout)] (ctx, segments, layout): (
            ArrayContext,
            Arc<dyn AsyncSegmentReader>,
            Layout,
        ),
    ) {
        let reader = layout.reader(segments, ctx).unwrap();
        let expr = pack([("a", get_item("a", ident())), ("b", get_item("b", ident()))]);
        let result = block_on(reader.evaluate_expr(
            // Take rows 0 and 1, skip row 2, and anything after that
            RowMask::new(Mask::from_iter([true, true, false]), 0),
            expr,
        ))
        .unwrap();

        assert_eq!(result.len(), 2);

        assert_eq!(
            result
                .as_struct_typed()
                .unwrap()
                .maybe_null_field_by_name("a")
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [7, 2].as_slice()
        );

        assert_eq!(
            result
                .as_struct_typed()
                .unwrap()
                .maybe_null_field_by_name("b")
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [4, 5].as_slice()
        );
    }
}
