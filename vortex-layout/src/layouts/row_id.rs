// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use dashmap::DashMap;
use vortex_array::arrays::{ConstantArray, StructArray};
use vortex_array::compute::filter;
use vortex_array::stats::{Precision, Stat};
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::PType::U64;
use vortex_dtype::{
    DType, Field, FieldMask, FieldName, FieldPath, FieldPathSet, Nullability, StructFields,
};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::transform::var_partition::{VarPartitionedExpr, var_partitions_with_map};
use vortex_expr::{
    ExactExpr, ExprRef, Identifier, Scope, ScopeDType, ScopeElement, ScopeFieldPathSetElement,
};
use vortex_mask::Mask;
use vortex_sequence::SequenceArray;

use crate::{ArrayEvaluation, LayoutReader, LayoutReaderRef, MaskEvaluation, PruningEvaluation};

pub struct RowIdLayoutReader {
    child: LayoutReaderRef,
    name: Arc<str>,
    partitioned_expr_cache: DashMap<ExactExpr, Arc<VarPartitionedExpr>>,
    file_index: u64,
    scope_dtype: ScopeDType,
}

pub static ROW_ID: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::Other(Arc::from("$vx.row_id")));
pub const FILE_ROW_NUMBER_FIELD: &str = "file_row_number";
pub const FILE_INDEX_FIELD: &str = "file_index";

impl RowIdLayoutReader {
    pub fn new(child: LayoutReaderRef) -> Self {
        Self::new_with_file_index(child, 0)
    }

    pub fn new_with_file_index(child: LayoutReaderRef, file_index: u64) -> Self {
        let scope_dtype = ScopeDType::new(child.dtype().clone()).with_dtype_element((
            ROW_ID.clone(),
            DType::Struct(
                StructFields::from_iter([
                    (FieldName::from(FILE_ROW_NUMBER_FIELD), DType::from(U64)),
                    (FILE_INDEX_FIELD.into(), U64.into()),
                ]),
                Nullability::NonNullable,
            ),
        ));
        Self {
            child,
            name: Arc::from("row_id_layout_reader"),
            partitioned_expr_cache: Default::default(),
            file_index,
            scope_dtype,
        }
    }
}

impl RowIdLayoutReader {
    /// Utility for partitioning an expression over the fields of a struct.
    fn partition_expr(&self, expr: &ExprRef) -> Arc<VarPartitionedExpr> {
        self.partitioned_expr_cache
            .entry(ExactExpr(expr.clone()))
            .or_insert_with(|| {
                // Partition the expression into expressions that can be evaluated over the row_id field
                // and all other fields that a delegated to their children.
                Arc::new(
                    var_partitions_with_map(expr, |id| {
                        if *id == *ROW_ID {
                            ROW_ID.clone()
                        } else {
                            Identifier::Identity
                        }
                    })
                    .vortex_expect("We should not fail to partition variables"),
                )
            })
            .clone()
    }

    fn row_id_scope(&self, row_range: &Range<u64>) -> ScopeElement {
        let arr_len = row_range.clone().count();

        (
            ROW_ID.clone(),
            StructArray::from_fields(&[
                (
                    FILE_ROW_NUMBER_FIELD,
                    SequenceArray::typed_new(row_range.start, 1, arr_len)
                        .vortex_expect("cannot be out of bounds")
                        .to_array(),
                ),
                (
                    FILE_INDEX_FIELD,
                    ConstantArray::new(self.file_index, arr_len).to_array(),
                ),
            ])
            .vortex_expect("valid struct array")
            .to_array(),
        )
    }

    pub fn row_id_stats_field_path_set() -> ScopeFieldPathSetElement {
        (
            ROW_ID.clone(),
            FieldPathSet::from_iter([
                FieldPath::from_iter([
                    Field::Name(FILE_ROW_NUMBER_FIELD.into()),
                    Field::Name(Stat::Max.name().into()),
                ]),
                FieldPath::from_iter([
                    Field::Name(FILE_ROW_NUMBER_FIELD.into()),
                    Field::Name(Stat::Min.name().into()),
                ]),
                FieldPath::from_iter([
                    Field::Name(FILE_INDEX_FIELD.into()),
                    Field::Name(Stat::Max.name().into()),
                ]),
                FieldPath::from_iter([
                    Field::Name(FILE_INDEX_FIELD.into()),
                    Field::Name(Stat::Min.name().into()),
                ]),
            ]),
        )
    }

    pub fn row_id_stats_set_scope(row_range: &Range<u64>, file_idx: u64) -> (Identifier, ArrayRef) {
        (
            ROW_ID.clone(),
            StructArray::from_fields(&[
                (
                    "file_row_number_max",
                    ConstantArray::new(row_range.end, 1).to_array(),
                ),
                (
                    "file_row_number_min",
                    ConstantArray::new(row_range.start, 1).to_array(),
                ),
                ("file_index_max", ConstantArray::new(file_idx, 1).to_array()),
                ("file_index_min", ConstantArray::new(file_idx, 1).to_array()),
            ])
            .vortex_expect("valid struct")
            .to_array(),
        )
    }
}

impl LayoutReader for RowIdLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn scope_dtype(&self) -> &ScopeDType {
        &self.scope_dtype
    }

    fn row_count(&self) -> Precision<u64> {
        self.child.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        self.child.register_splits(field_mask, row_offset, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        // TODO(joe): add variable and then row_id to the pruning eval
        self.child.pruning_evaluation(row_range, expr)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        let arr_len = row_range.clone().count();
        let partitioned = self.partition_expr(expr);
        let Some(row_id) = partitioned.find_partition(&ROW_ID) else {
            return self.child.filter_evaluation(row_range, expr);
        };
        let rest = partitioned.find_partition(&Identifier::Identity);

        let row_id_scope = Scope::empty(arr_len).with_array_pair(self.row_id_scope(row_range));

        let rest_eval = if let Some(rest) = &rest {
            let dtype = rest.return_dtype(&ScopeDType::new(self.dtype().clone()))?;
            Some(if matches!(dtype, DType::Bool(_)) {
                // If the partition evaluates to a boolean, we can evaluate it as a mask which
                // can often be more efficient since nulls are turned into `false` early on,
                // and layouts can perform predicate pruning / indexing.
                FieldEval::Mask(self.child.filter_evaluation(row_range, expr)?)
            } else {
                // Otherwise, we evaluate the projection as an array, and combine the results
                // at the end.
                FieldEval::Array(self.child.projection_evaluation(row_range, expr)?)
            })
        } else {
            None
        };

        Ok(Box::new(RowIdMaskEvaluation {
            row_id_partition_expr: row_id.clone(),
            row_id_partition_scope: row_id_scope,
            child: rest_eval,
            root: partitioned.root.clone(),
        }) as Box<_>)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        let arr_len = row_range.clone().count();
        let partitioned = self.partition_expr(expr);
        let Some(row_id_expr) = partitioned.find_partition(&ROW_ID) else {
            return self.child.projection_evaluation(row_range, expr);
        };
        let rest = partitioned.find_partition(&Identifier::Identity);

        let row_id_scope = Scope::empty(arr_len).with_array_pair(self.row_id_scope(row_range));

        let rest_eval = rest
            .map(|r| self.child.projection_evaluation(row_range, r))
            .transpose()?;

        Ok(Box::new(RowIdArrayEvaluation {
            row_id_partition_expr: row_id_expr.clone(),
            row_id_partition_scope: row_id_scope,
            child: rest_eval,
            root: partitioned.root.clone(),
        }) as Box<_>)
    }
}

enum FieldEval {
    Mask(Box<dyn MaskEvaluation>),
    Array(Box<dyn ArrayEvaluation>),
}

struct RowIdMaskEvaluation {
    row_id_partition_expr: ExprRef,
    row_id_partition_scope: Scope,
    child: Option<FieldEval>,
    root: ExprRef,
}

#[async_trait]
impl MaskEvaluation for RowIdMaskEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        let root_scope = if let Some(child) = &self.child {
            let mask = mask.clone();
            let arr = async move {
                match child {
                    FieldEval::Mask(eval) => Ok(eval.invoke(mask.clone()).await?.into_array()),
                    FieldEval::Array(eval) => eval.invoke(Mask::new_true(mask.len())).await,
                }
            }
            .await?;
            Scope::new(arr)
        } else {
            Scope::empty(mask.len())
        };

        let row_id_value = self
            .row_id_partition_expr
            .evaluate(&self.row_id_partition_scope)?;

        let root_scope = root_scope.with_array(ROW_ID.clone(), row_id_value);

        let root_mask = Mask::try_from(self.root.evaluate(&root_scope)?.as_ref())?;
        let mask = mask.bitand(&root_mask);

        Ok(mask)
    }
}

struct RowIdArrayEvaluation {
    row_id_partition_expr: ExprRef,
    row_id_partition_scope: Scope,
    child: Option<Box<dyn ArrayEvaluation>>,
    root: ExprRef,
}

#[async_trait]
impl ArrayEvaluation for RowIdArrayEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        let root_scope = if let Some(child) = &self.child {
            Scope::new(child.invoke(mask.clone()).await?.into_array())
        } else {
            Scope::empty(mask.len())
        };

        let row_id_result = self
            .row_id_partition_expr
            .evaluate(&self.row_id_partition_scope)?;
        let filtered = filter(&row_id_result, &mask)?;

        let root_scope = root_scope.with_array(ROW_ID.clone(), filtered.clone());

        self.root.evaluate(&root_scope)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_buffer::BooleanBuffer;
    use futures::executor::block_on;
    use futures::stream;
    use itertools::Itertools;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{ArrayContext, ToCanonical};
    use vortex_expr::{eq, get_item, gt, lit, or, root, var};
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::row_id::{FILE_ROW_NUMBER_FIELD, ROW_ID, RowIdLayoutReader};
    use crate::segments::{SegmentSource, SequenceWriter, TestSegments};
    use crate::sequence::SequenceId;
    use crate::{LayoutReader, LayoutStrategy, SequentialStreamAdapter, SequentialStreamExt};

    #[test]
    fn flat_expr_no_row_id() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::from_iter(1..=5).to_array();
            let array_clone = array.clone();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    sequence_writer.clone(),
                    SequentialStreamAdapter::new(
                        array.dtype().clone(),
                        stream::once(async { Ok((SequenceId::root().downgrade(), array_clone)) }),
                    )
                    .sendable(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let expr = eq(root(), lit(3i32));
            let result =
                RowIdLayoutReader::new(layout.new_reader("".into(), segments, ctx).unwrap())
                    .projection_evaluation(&(0..layout.row_count()), &expr)
                    .unwrap()
                    .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
                    .await
                    .unwrap()
                    .to_bool()
                    .unwrap();

            assert_eq!(
                &BooleanBuffer::from_iter([false, false, true, false, false]),
                result.boolean_buffer()
            );
        })
    }

    #[test]
    fn flat_expr_row_id() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::from_iter(1..=5).to_array();
            let array_clone = array.clone();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    sequence_writer.clone(),
                    SequentialStreamAdapter::new(
                        array.dtype().clone(),
                        stream::once(async { Ok((SequenceId::root().downgrade(), array_clone)) }),
                    )
                    .sendable(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let expr = gt(
                get_item(FILE_ROW_NUMBER_FIELD, var(ROW_ID.clone())),
                lit(3u64),
            );
            let result =
                RowIdLayoutReader::new(layout.new_reader("".into(), segments, ctx).unwrap())
                    .projection_evaluation(&(0..layout.row_count()), &expr)
                    .unwrap()
                    .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
                    .await
                    .unwrap()
                    .to_bool()
                    .unwrap();

            assert_eq!(
                &BooleanBuffer::from_iter([false, false, false, false, true]),
                result.boolean_buffer()
            );
        })
    }

    #[test]
    fn flat_expr_or() {
        block_on(async {
            let ctx = ArrayContext::empty();
            let segments = TestSegments::default();
            let sequence_writer = SequenceWriter::new(Box::new(segments.clone()));
            let array = PrimitiveArray::from_iter(1..=5).to_array();
            let array_clone = array.clone();
            let layout = FlatLayoutStrategy::default()
                .write_stream(
                    &ctx,
                    sequence_writer.clone(),
                    SequentialStreamAdapter::new(
                        array.dtype().clone(),
                        stream::once(async { Ok((SequenceId::root().downgrade(), array_clone)) }),
                    )
                    .sendable(),
                )
                .await
                .unwrap();
            let segments: Arc<dyn SegmentSource> = Arc::new(segments);

            let expr = or(
                eq(root(), lit(3i32)),
                or(
                    gt(
                        get_item(FILE_ROW_NUMBER_FIELD, var(ROW_ID.clone())),
                        lit(3u64),
                    ),
                    eq(root(), lit(1i32)),
                ),
            );

            let result =
                RowIdLayoutReader::new(layout.new_reader("".into(), segments, ctx).unwrap())
                    .projection_evaluation(&(0..layout.row_count()), &expr)
                    .unwrap()
                    .invoke(Mask::new_true(layout.row_count().try_into().unwrap()))
                    .await
                    .unwrap()
                    .to_bool()
                    .unwrap();

            assert_eq!(
                vec![true, false, true, false, true],
                result.boolean_buffer().iter().collect_vec()
            );
        })
    }
}
