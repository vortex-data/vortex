use async_trait::async_trait;
use futures::future::try_join_all;
use futures::FutureExt;
use vortex_array::array::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_error::VortexResult;
use vortex_expr::transform::split_expression;
use vortex_expr::ExprRef;
use vortex_scan::RowMask;

use crate::layouts::struct_::reader::StructReader;
use crate::reader::LayoutReaderExt;
use crate::{ExprEvaluator, LayoutReader};

#[async_trait(?Send)]
impl ExprEvaluator for StructReader {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<ArrayData> {
        // TODO: apply validity mask to row_mask

        // Compute the result dtype of the expression.
        // let dtype = expr
        //     .evaluate(&Canonical::empty(self.dtype())?.into_array())?
        //     .dtype()
        //     .clone();

        let (combine_expr, expr_split) = split_expression(expr, self.dtype())?;

        println!("comb {:?}", combine_expr);
        println!("expr split {:?}", expr_split);

        let mut results = Vec::with_capacity(expr_split.len());
        let mut result_field_name = Vec::with_capacity(expr_split.len());

        for (field, (res, expr)) in &expr_split {
            // check if the field exists
            // chunks.push(chunk_reader.evaluate_expr(chunk_mask, expr).boxed_local());

            result_field_name.push(res.clone());

            results.push(
                self.child(field)?
                    .evaluate_expr(row_mask.clone(), expr.clone())
                    .boxed_local(),
            );
        }

        let arrays = try_join_all(results).await?;

        let row_count = self.layout().row_count();

        assert!(arrays.iter().all(|a| a.len() as u64 == row_count));

        let pack = StructArray::try_new(
            result_field_name.into(),
            arrays,
            row_count as usize,
            // TODO: handle validity
            Validity::NonNullable,
        )?;

        // Now we need to evaluate the expression
        combine_expr.evaluate(&pack.into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::array::StructArray;
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability, StructDType};
    use vortex_expr::{get_item, gt, ident};
    use vortex_scan::RowMask;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::layouts::struct_::writer::StructLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;
    use crate::LayoutData;

    /// Create a chunked layout with three chunks of primitive arrays.
    fn struct_layout() -> (Arc<TestSegments>, LayoutData) {
        let mut segments = TestSegments::default();

        let layout = StructLayoutWriter::new(
            DType::Struct(
                StructDType::new(
                    vec!["a".into(), "b".into()].into(),
                    vec![I32.into(), I32.into()],
                ),
                Nullability::NonNullable,
            ),
            vec![
                Box::new(FlatLayoutWriter::new(I32.into())),
                Box::new(FlatLayoutWriter::new(I32.into())),
            ],
        )
        .push_all(
            &mut segments,
            [StructArray::from_fields(
                [
                    ("a", buffer![7, 2, 3].into_array()),
                    ("b", buffer![4, 5, 6].into_array()),
                ]
                .as_slice(),
            )
            .map(IntoArrayData::into_array)],
        )
        .unwrap();
        (Arc::new(segments), layout)
    }

    #[test]
    fn test_struct_layout() {
        let (segments, layout) = struct_layout();

        let reader = layout.reader(segments, Default::default()).unwrap();
        let expr = gt(get_item("a", ident()), get_item("b", ident()));
        let result = futures::executor::block_on(
            reader.evaluate_expr(RowMask::new_valid_between(0, 3), expr),
        )
        .unwrap();
        println!(
            "res {:?}",
            result
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>()
        );
    }
}
