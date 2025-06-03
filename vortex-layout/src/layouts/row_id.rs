use std::collections::BTreeSet;
use std::ops::{BitAnd, Range};
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use dashmap::DashMap;
use vortex_array::stats::Precision;
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::transform::var_partition::{VarPartitionedExpr, var_partitions};
use vortex_expr::{ExactExpr, ExprRef, IDENTITY_IDENTIFIER, Identifier, Scope, ScopeDType};
use vortex_mask::Mask;
use vortex_sequence::SequenceArray;

use crate::{ArrayEvaluation, LayoutReader, LayoutReaderRef, MaskEvaluation, PruningEvaluation};

pub struct RowIdLayoutReader {
    child: LayoutReaderRef,
    name: Arc<str>,
    partitioned_expr_cache: DashMap<ExactExpr, Arc<VarPartitionedExpr>>,
}

static ROW_ID: LazyLock<Identifier> = LazyLock::new(|| Arc::from("row_id"));

impl RowIdLayoutReader {
    pub fn new(child: LayoutReaderRef) -> Self {
        Self {
            child,
            name: Arc::from("row_id_layout_reader"),
            partitioned_expr_cache: Default::default(),
        }
    }
}

impl RowIdLayoutReader {
    /// Utility for partitioning an expression over the fields of a struct.
    fn partition_expr(&self, expr: &ExprRef) -> Arc<VarPartitionedExpr> {
        self.partitioned_expr_cache
            .entry(ExactExpr(expr.clone()))
            .or_insert_with(|| {
                // Partition the expression into expressions that can be evaluated over individual fields
                Arc::new(
                    var_partitions(expr).vortex_expect("We should not fail to partition variables"),
                )
            })
            .clone()
    }
}

impl LayoutReader for RowIdLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
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
        let rest = partitioned.find_partition(&Arc::from(IDENTITY_IDENTIFIER));

        let row_id_scope = Scope::empty(arr_len).with_value(
            ROW_ID.clone(),
            SequenceArray::typed_new(row_range.start, 1, arr_len)
                .vortex_expect("cannot be out of bounds")
                .to_array(),
        );

        let rest_eval = if let Some(rest) = &rest {
            let dtype = rest.return_dtype(&ScopeDType::new(self.child.dtype().clone()))?;
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

        let res = row_id.evaluate(&row_id_scope)?;
        Ok(Box::new(RowIdMaskEvaluation {
            row_id_partition: res,
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
        let Some(row_id) = partitioned.find_partition(&ROW_ID) else {
            return self.child.projection_evaluation(row_range, expr);
        };
        let rest = partitioned.find_partition(&Arc::from(IDENTITY_IDENTIFIER));

        let row_id_scope = Scope::empty(arr_len).with_value(
            ROW_ID.clone(),
            SequenceArray::typed_new(row_range.start, 1, arr_len)
                .vortex_expect("cannot be out of bounds")
                .to_array(),
        );

        let res = row_id.evaluate(&row_id_scope)?;

        let rest_eval = rest
            .map(|r| self.child.projection_evaluation(row_range, r))
            .transpose()?;

        Ok(Box::new(RowIdArrayEvaluation {
            row_id_partition: res,
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
    row_id_partition: ArrayRef,
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

        let root_scope = root_scope.with_value(ROW_ID.clone(), self.row_id_partition.clone());

        let root_mask = Mask::try_from(self.root.evaluate(&root_scope)?.as_ref())?;
        let mask = mask.bitand(&root_mask);

        Ok(mask)
    }
}

struct RowIdArrayEvaluation {
    row_id_partition: ArrayRef,
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

        let root_scope = root_scope.with_value(ROW_ID.clone(), self.row_id_partition.clone());

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
    use vortex_expr::{eq, gt, lit, or, root, var};
    use vortex_mask::Mask;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::row_id::RowIdLayoutReader;
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
                RowIdLayoutReader::new(layout.new_reader(&"".into(), &segments, &ctx).unwrap())
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

            let expr = gt(var("row_id"), lit(3u64));
            let result =
                RowIdLayoutReader::new(layout.new_reader(&"".into(), &segments, &ctx).unwrap())
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
                or(gt(var("row_id"), lit(3u64)), eq(root(), lit(1i32))),
            );

            let result =
                RowIdLayoutReader::new(layout.new_reader(&"".into(), &segments, &ctx).unwrap())
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
