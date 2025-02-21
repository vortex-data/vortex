use async_trait::async_trait;
use futures::future::try_join_all;
use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, IntoArray};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::ExprRef;

use crate::layouts::struct_::reader::StructReader;
use crate::{ExprEvaluator, RowMask};

#[async_trait]
impl ExprEvaluator for StructReader {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<Array> {
        // Partition the expression into expressions that can be evaluated over individual fields
        let partitioned = self.partition_expr(expr.clone())?;
        let field_readers: Vec<_> = partitioned
            .partition_names
            .iter()
            .map(|name| self.child(name))
            .try_collect()?;

        // Short-circuit if there is only one partition
        if partitioned.partitions.len() == 1 {
            return self
                .child(&partitioned.partition_names[0])?
                .evaluate_expr(row_mask, partitioned.partitions[0].clone())
                .await;
        }

        // Otherwise, evaluate all partitions concurrently
        let arrays = try_join_all(
            field_readers
                .iter()
                .zip_eq(partitioned.partitions.iter())
                .map(|(reader, partition)| {
                    reader.evaluate_expr(row_mask.clone(), partition.clone())
                }),
        )
        .await?;

        let row_count = row_mask.true_count();
        debug_assert!(arrays.iter().all(|a| a.len() == row_count));

        let root_scope = StructArray::try_new(
            partitioned.partition_names.clone(),
            arrays,
            row_count,
            Validity::NonNullable,
        )?
        .into_array();

        partitioned.root.evaluate(&root_scope)
    }

    async fn prune_mask(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<RowMask> {
        // We currently can only perform pruning if the expression references a single field.
        // Otherwise, we have no good way to recombine the results.
        let partitioned = self.partition_expr(expr.clone())?;
        if partitioned.partitions.len() != 1 {
            log::debug!("Cannot push-down pruning for multi-field expr {}", expr);
            return Ok(row_mask);
        }

        let field_name = partitioned
            .partition_names
            .iter()
            .next()
            .vortex_expect("one partition");
        self.child(field_name)?
            .prune_mask(row_mask, partitioned.partitions[0].clone())
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use rstest::{fixture, rstest};
    use vortex_array::arrays::StructArray;
    use vortex_array::{IntoArray, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Nullability, StructDType};
    use vortex_expr::{get_item, gt, ident, pack};
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::layouts::struct_::writer::StructLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::segments::AsyncSegmentReader;
    use crate::writer::LayoutWriterExt;
    use crate::{Layout, RowMask};

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn struct_layout() -> (Arc<dyn AsyncSegmentReader>, Layout) {
        let mut segments = TestSegments::default();

        let layout = StructLayoutWriter::new(
            DType::Struct(
                Arc::new(StructDType::new(
                    vec!["a".into(), "b".into(), "c".into()].into(),
                    vec![I32.into(), I32.into(), I32.into()],
                )),
                Nullability::NonNullable,
            ),
            vec![
                Box::new(FlatLayoutWriter::new(I32.into(), Default::default())),
                Box::new(FlatLayoutWriter::new(I32.into(), Default::default())),
                Box::new(FlatLayoutWriter::new(I32.into(), Default::default())),
            ],
        )
        .push_all(
            &mut segments,
            [StructArray::from_fields(
                [
                    ("a", buffer![7, 2, 3].into_array()),
                    ("b", buffer![4, 5, 6].into_array()),
                    ("c", buffer![4, 5, 6].into_array()),
                ]
                .as_slice(),
            )
            .map(IntoArray::into_array)],
        )
        .unwrap();
        (Arc::new(segments), layout)
    }

    #[rstest]
    fn test_struct_layout(
        #[from(struct_layout)] (segments, layout): (Arc<dyn AsyncSegmentReader>, Layout),
    ) {
        let reader = layout.reader(segments, Default::default()).unwrap();
        let expr = gt(get_item("a", ident()), get_item("b", ident()));
        let result =
            block_on(reader.evaluate_expr(RowMask::new_valid_between(0, 3), expr)).unwrap();
        assert_eq!(
            vec![true, false, false],
            result
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn test_struct_layout_row_mask(
        #[from(struct_layout)] (segments, layout): (Arc<dyn AsyncSegmentReader>, Layout),
    ) {
        let reader = layout.reader(segments, Default::default()).unwrap();
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
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_struct_layout_select() {
        let (segments, layout) = struct_layout();

        let reader = layout.reader(segments, Default::default()).unwrap();
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
                .as_struct_array()
                .unwrap()
                .maybe_null_field_by_name("a")
                .unwrap()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [7, 2].as_slice()
        );

        assert_eq!(
            result
                .as_struct_array()
                .unwrap()
                .maybe_null_field_by_name("b")
                .unwrap()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [4, 5].as_slice()
        );
    }
}
