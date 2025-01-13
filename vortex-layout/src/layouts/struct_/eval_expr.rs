use async_trait::async_trait;
use futures::future::try_join_all;
use itertools::Itertools;
use vortex_array::array::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_expr::transform::partition::partition;
use vortex_expr::{ident, ExprRef};
use vortex_scan::RowMask;

use crate::layouts::struct_::reader::StructReader;
use crate::{ExprEvaluator, LayoutReaderExt};

#[async_trait]
impl ExprEvaluator for StructReader {
    async fn evaluate_expr(&self, row_mask: RowMask, expr: ExprRef) -> VortexResult<ArrayData> {
        // Partition the expression into expressions that can be evaluated over individual fields
        let partitioned = partition(expr, self.struct_dtype())?;
        let field_readers: Vec<_> = partitioned
            .partitions
            .iter()
            // TODO(joe): remove field from self.child
            .map(|partition| self.child(&partition.name.clone()))
            .try_collect()?;

        let arrays = try_join_all(
            field_readers
                .iter()
                .zip_eq(partitioned.partitions.iter())
                .map(|(reader, partition)| {
                    reader.evaluate_expr(row_mask.clone(), partition.expr.clone())
                }),
        )
        .await?;

        let validity = if self.dtype().nullability() == Nullability::Nullable {
            let validity: ArrayData = self
                .validity()?
                .evaluate_expr(row_mask.clone(), ident())
                .await?;
            let bool = validity.into_bool()?;
            Validity::from(bool.boolean_buffer())
        } else {
            Validity::NonNullable
        };

        let row_count = row_mask.true_count();
        debug_assert!(arrays.iter().all(|a| a.len() == row_count));

        let root_scope = StructArray::try_new(
            partitioned
                .partitions
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
                .into(),
            arrays,
            row_count,
            validity,
        )?
        .into_array();

        // Recombine the partitioned expressions into a single expression
        partitioned.root.evaluate(&root_scope)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;
    use itertools::Itertools;
    use vortex_array::array::{BoolArray, StructArray};
    use vortex_array::compute::FilterMask;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_array::validity::{ArrayValidity, Validity};
    use vortex_array::{IntoArrayData, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::DType::Bool;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, Field, Nullability, StructDType};
    use vortex_expr::{get_item, gt, ident, select};
    use vortex_scan::RowMask;

    use crate::layouts::flat::writer::FlatLayoutWriter;
    use crate::layouts::struct_::writer::StructLayoutWriter;
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;
    use crate::LayoutData;

    /// Create a chunked layout with three chunks of primitive arrays.
    fn struct_layout(validity: Validity) -> (Arc<TestSegments>, LayoutData) {
        let mut segments = TestSegments::default();

        let layout = StructLayoutWriter::new(
            DType::Struct(
                StructDType::new(
                    vec!["a".into(), "b".into(), "c".into()].into(),
                    vec![I32.into(), I32.into(), I32.into()],
                ),
                validity.nullability(),
            ),
            vec![
                Box::new(FlatLayoutWriter::new(I32.into())),
                Box::new(FlatLayoutWriter::new(I32.into())),
                Box::new(FlatLayoutWriter::new(I32.into())),
            ],
            Box::new(FlatLayoutWriter::new(Bool(Nullability::NonNullable))),
        )
        .push_all(
            &mut segments,
            [StructArray::try_new(
                ["a".into(), "b".into(), "c".into()].into(),
                vec![
                    buffer![7, 2, 3].into_array(),
                    buffer![4, 5, 6].into_array(),
                    buffer![4, 5, 6].into_array(),
                ],
                3,
                validity,
            )
            .map(IntoArrayData::into_array)],
        )
        .unwrap();
        (Arc::new(segments), layout)
    }

    #[test]
    fn test_struct_layout() {
        let (segments, layout) = struct_layout(NonNullable);

        let reader = layout.reader(segments, Default::default()).unwrap();

        let expr = get_item("a", ident());
        let result =
            block_on(reader.evaluate_expr(RowMask::new_valid_between(0, 3), expr)).unwrap();
        println!(
            "result {:?}",
            result.into_primitive().unwrap().as_slice::<i32>()
        );

        let expr = get_item("b", ident());
        let result =
            block_on(reader.evaluate_expr(RowMask::new_valid_between(0, 3), expr)).unwrap();
        println!(
            "result {:?}",
            result.into_primitive().unwrap().as_slice::<i32>()
        );

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

    #[test]
    fn test_struct_layout_row_mask() {
        let (segments, layout) = struct_layout(NonNullable);

        let reader = layout.reader(segments, Default::default()).unwrap();
        let expr = gt(get_item("a", ident()), get_item("b", ident()));
        let result = block_on(reader.evaluate_expr(
            // Take rows 0 and 1, skip row 2, and anything after that
            RowMask::new(FilterMask::from_iter([true, true, false]), 0),
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
        let (segments, layout) = struct_layout(NonNullable);

        let reader = layout.reader(segments, Default::default()).unwrap();
        let expr = select(vec!["a".into(), "b".into()], ident());
        let result = block_on(reader.evaluate_expr(
            // Take rows 0 and 1, skip row 2, and anything after that
            RowMask::new(FilterMask::from_iter([true, true, false]), 0),
            expr,
        ))
        .unwrap();

        assert_eq!(result.len(), 2);

        assert_eq!(
            result
                .as_struct_array()
                .unwrap()
                .maybe_null_field(&Field::Name("a".into()))
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
                .maybe_null_field(&Field::Name("b".into()))
                .unwrap()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [4, 5].as_slice()
        );
    }

    #[test]
    fn test_struct_nullable() {
        let (segments, layout) = struct_layout(Validity::Array(
            BoolArray::from_iter([false, true, true]).into_array(),
        ));

        let reader = layout.reader(segments, Default::default()).unwrap();
        let expr = get_item("a", ident());
        let result = block_on(reader.evaluate_expr(
            // Take rows 0 and 1, skip row 2, and anything after that
            RowMask::new(FilterMask::from_iter([true, true, true]), 0),
            expr,
        ))
        .unwrap();

        assert_eq!(result.len(), 3);

        assert_eq!(
            result
                .logical_validity()
                .into_array()
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect_vec(),
            vec![false, true, true]
        );

        assert_eq!(
            result.into_primitive().unwrap().as_slice::<i32>(),
            [7, 2, 3].as_slice()
        );
    }
}
