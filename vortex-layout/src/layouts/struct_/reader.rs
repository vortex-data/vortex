// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use dashmap::DashMap;
use itertools::Itertools;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask, FieldName, StructFields};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_expr::transform::immediate_access::annotate_scope_access;
use vortex_expr::transform::partition::{PartitionedExpr, partition};
use vortex_expr::transform::replace::{replace, replace_root_fields};
use vortex_expr::transform::simplify_typed::simplify_typed;
use vortex_expr::{ExactExpr, ExprRef, col, root};
use vortex_utils::aliases::hash_map::HashMap;

use crate::layouts::partitioned::{PartitionedArrayEvaluation, PartitionedMaskEvaluation};
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSource;
use crate::{
    ArrayEvaluation, LayoutReader, LayoutReaderRef, LazyReaderChildren, MaskEvaluation,
    NoOpPruningEvaluation, PruningEvaluation,
};

pub struct StructReader {
    layout: StructLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,

    /// A `pack` expression that holds each individual field of the root DType. This expansion
    /// ensures we can correctly partition expressions over the fields of the struct.
    expanded_root_expr: ExprRef,

    field_lookup: Option<HashMap<FieldName, usize>>,
    partitioned_expr_cache: DashMap<ExactExpr, Partitioned>,
}

impl StructReader {
    pub(super) fn try_new(
        layout: StructLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<Self> {
        let struct_dt = layout.struct_fields();

        // NOTE: This number is arbitrary and likely depends on the longest common prefix of field names
        let field_lookup = (struct_dt.nfields() > 80).then(|| {
            struct_dt
                .names()
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect()
        });

        let lazy_children =
            LazyReaderChildren::new(layout.children.clone(), segment_source.clone());

        // Create an expanded root expression that contains all fields of the struct.
        let expanded_root_expr = replace_root_fields(root(), struct_dt);

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self {
            layout,
            name,
            expanded_root_expr,
            lazy_children,
            field_lookup,
            partitioned_expr_cache: Default::default(),
        })
    }

    /// Return the [`StructFields`] of this layout.
    fn struct_fields(&self) -> &StructFields {
        self.layout.struct_fields()
    }

    /// Return the child reader for the field.
    fn child(&self, name: &FieldName) -> VortexResult<&LayoutReaderRef> {
        let idx = self
            .field_lookup
            .as_ref()
            .and_then(|lookup| lookup.get(name).copied())
            .or_else(|| self.struct_fields().find(name))
            .ok_or_else(|| vortex_err!("Field {} not found in struct layout", name))?;
        self.child_by_idx(idx)
    }

    /// Return the child reader for the field, by index.
    fn child_by_idx(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        let field_dtype = self
            .struct_fields()
            .field_by_index(idx)
            .ok_or_else(|| vortex_err!("Missing field {idx}"))?;
        let name = &self.struct_fields().names()[idx];
        self.lazy_children
            .get(idx, &field_dtype, &format!("{}.{}", self.name, name).into())
    }

    /// Utility for partitioning an expression over the fields of a struct.
    fn partition_expr(&self, expr: ExprRef) -> Partitioned {
        self.partitioned_expr_cache
            .entry(ExactExpr(expr.clone()))
            .or_insert_with(|| {
                // First, we expand the root scope into the fields of the struct to ensure
                // that partitioning works correctly.
                let expr = replace(expr.clone(), &root(), self.expanded_root_expr.clone());
                let expr = simplify_typed(expr, self.dtype())
                    .vortex_expect("We should not fail to simplify expression over struct fields");

                // Partition the expression into expressions that can be evaluated over individual fields
                let mut partitioned = partition(
                    expr.clone(),
                    self.dtype(),
                    annotate_scope_access(
                        self.dtype()
                            .as_struct()
                            .vortex_expect("We know it's a struct DType"),
                    ),
                )
                .vortex_expect("We should not fail to partition expression over struct fields");

                if partitioned.partitions.len() == 1 {
                    // If there's only one partition, we step into the field scope of the original
                    // expression by replacing any `$.a` with `$`.
                    return Partitioned::Single(
                        partitioned.partition_names[0].clone(),
                        replace(
                            expr.clone(),
                            &col(partitioned.partition_names[0].clone()),
                            root(),
                        ),
                    );
                }

                // We now need to process the partitioned expressions to rewrite the root scope
                // to be that of the field, rather than the struct. In other words, "stepping in"
                // to the field scope.
                partitioned.partitions = partitioned
                    .partitions
                    .iter()
                    .zip_eq(partitioned.partition_names.iter())
                    .map(|(e, name)| replace(e.clone(), &col(name.clone()), root()))
                    .collect();

                Partitioned::Multi(Arc::new(partitioned))
            })
            .clone()
    }
}

/// When partitioning an expression, in the case it only has a single partition we can avoid
/// some cost and just delegate to the child reader directly.
#[derive(Clone)]
enum Partitioned {
    Single(FieldName, ExprRef),
    Multi(Arc<PartitionedExpr<FieldName>>),
}

impl LayoutReader for StructReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(self.layout.row_count())
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        // In the case of an empty struct, we need to register the end split.
        splits.insert(row_offset + self.layout.row_count);

        self.layout.matching_fields(field_mask, |mask, idx| {
            self.child_by_idx(idx)?
                .register_splits(&[mask], row_offset, splits)
        })
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        // Partition the expression into expressions that can be evaluated over individual fields
        match &self.partition_expr(expr.clone()) {
            Partitioned::Single(name, partition) => {
                self.child(name)?.pruning_evaluation(row_range, partition)
            }
            Partitioned::Multi(_) => {
                // TODO(ngates): if all partitions are boolean, we can use a pruning evaluation. Otherwise
                //  there's not much we can do? Maybe... it's complicated...
                Ok(Box::new(NoOpPruningEvaluation))
            }
        }
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        // Partition the expression into expressions that can be evaluated over individual fields
        match &self.partition_expr(expr.clone()) {
            Partitioned::Single(name, partition) => {
                self.child(name)?.filter_evaluation(row_range, partition)
            }
            Partitioned::Multi(partitioned) => Ok(Box::new(PartitionedMaskEvaluation::try_new(
                partitioned.clone(),
                |name, expr| self.child(name)?.filter_evaluation(row_range, expr),
                |name, expr| self.child(name)?.projection_evaluation(row_range, expr),
            )?)),
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        // Partition the expression into expressions that can be evaluated over individual fields
        match &self.partition_expr(expr.clone()) {
            Partitioned::Single(name, partition) => self
                .child(name)?
                .projection_evaluation(row_range, partition),
            Partitioned::Multi(partitioned) => Ok(Box::new(PartitionedArrayEvaluation::try_new(
                partitioned.clone(),
                |name, expr| self.child(name)?.projection_evaluation(row_range, expr),
            )?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arcref::ArcRef;
    use futures::executor::block_on;
    use futures::stream;
    use itertools::Itertools;
    use rstest::{fixture, rstest};
    use vortex_array::arrays::StructArray;
    use vortex_array::{Array, ArrayContext, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, StructFields};
    use vortex_expr::{eq, get_item, get_item_scope, gt, lit, or, pack, root};
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::struct_::writer::StructStrategy;
    use crate::segments::{SegmentSource, SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{LayoutRef, LayoutStrategy, SequentialStreamAdapter, SequentialStreamExt as _};

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn struct_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = TestSegments::default();
        let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
        let strategy =
            StructStrategy::new(ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())));
        let layout = block_on(
            strategy.write_stream(
                &ctx,
                sequence_writer,
                SequentialStreamAdapter::new(
                    DType::Struct(
                        StructFields::new(
                            vec!["a".into(), "b".into(), "c".into()].into(),
                            vec![I32.into(), I32.into(), I32.into()],
                        ),
                        NonNullable,
                    ),
                    stream::once(async {
                        Ok((
                            SequenceId::root().downgrade(),
                            StructArray::from_fields(
                                [
                                    ("a", buffer![7, 2, 3].into_array()),
                                    ("b", buffer![4, 5, 6].into_array()),
                                    ("c", buffer![4, 5, 6].into_array()),
                                ]
                                .as_slice(),
                            )
                            .unwrap()
                            .into_array(),
                        ))
                    }),
                )
                .sendable(),
            ),
        )
        .unwrap();

        (Arc::new(segments), layout)
    }

    #[rstest]
    fn test_struct_layout_or(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout.new_reader("".into(), segments).unwrap();
        let filt = or(
            eq(get_item_scope("a"), lit(7)),
            or(
                eq(get_item_scope("b"), lit(5)),
                eq(get_item_scope("a"), lit(3)),
            ),
        );
        let result = block_on(
            reader
                .filter_evaluation(&(0..3), &filt)
                .unwrap()
                .invoke(Mask::new_true(3)),
        )
        .unwrap();
        assert_eq!(
            vec![true, true, true],
            result.to_boolean_buffer().iter().collect_vec()
        );
    }

    #[rstest]
    fn test_struct_layout(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout.new_reader("".into(), segments).unwrap();
        let expr = gt(get_item("a", root()), get_item("b", root()));
        let result = block_on(
            reader
                .projection_evaluation(&(0..3), &expr)
                .unwrap()
                .invoke(Mask::new_true(3)),
        )
        .unwrap();
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
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout.new_reader("".into(), segments).unwrap();
        let expr = gt(get_item("a", root()), get_item("b", root()));
        let result = block_on(
            reader
                .projection_evaluation(&(0..3), &expr)
                .unwrap()
                .invoke(Mask::from_iter([true, true, false])),
        )
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
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout.new_reader("".into(), segments).unwrap();
        let expr = pack(
            [("a", get_item("a", root())), ("b", get_item("b", root()))],
            NonNullable,
        );
        let result = block_on(
            reader
                .projection_evaluation(&(0..3), &expr)
                .unwrap()
                // Take rows 0 and 1, skip row 2, and anything after that
                .invoke(Mask::from_iter([true, true, false])),
        )
        .unwrap();

        assert_eq!(result.len(), 2);

        assert_eq!(
            result
                .to_struct()
                .unwrap()
                .field_by_name("a")
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [7, 2].as_slice()
        );

        assert_eq!(
            result
                .to_struct()
                .unwrap()
                .field_by_name("b")
                .unwrap()
                .to_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [4, 5].as_slice()
        );
    }
}
