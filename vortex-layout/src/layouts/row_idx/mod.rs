// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod expr;

use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use vortex_array::compute::filter;
use vortex_array::stats::Precision;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::{DType, FieldMask, PType};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::transform::partition::{PartitionedExpr, partition};
use vortex_expr::{ExactExpr, ExprRef, Scope, ScopeDType, is_root};
use vortex_mask::Mask;
use vortex_scalar::PValue;
use vortex_sequence::SequenceArray;

use crate::layouts::row_idx::expr::{RowIdxVTable, RowIdxVar};
use crate::{ArrayEvaluation, LayoutReader, MaskEvaluation, NoOpMaskEvaluation, PruningEvaluation};

pub struct RowIdxLayoutReader {
    name: Arc<str>,
    child: Arc<dyn LayoutReader>,
    row_offset: u64,

    partition_cache: DashMap<ExactExpr, Partitioned>,
}

impl RowIdxLayoutReader {
    fn partition_expr(&self, expr: &ExprRef) -> Partitioned {
        self.partition_cache
            .entry(ExactExpr(expr.clone()))
            .or_insert_with(|| {
                // Partition the expression into row idx and child expressions.
                let partitioned = partition(expr.clone(), self.dtype(), |expr| {
                    if expr.is::<RowIdxVTable>() {
                        vec![Partition::RowIdx]
                    } else if is_root(expr) {
                        vec![Partition::Child]
                    } else {
                        vec![]
                    }
                })
                .vortex_expect("We should not fail to partition expression over struct fields");

                // If there's only a single partition, we can directly return the expression.
                if partitioned.partitions.len() == 1 {
                    return match &partitioned.partition_annotations[0] {
                        Partition::RowIdx => Partitioned::RowIdx(expr.clone()),
                        Partition::Child => Partitioned::Child(expr.clone()),
                    };
                }

                assert_eq!(
                    partitioned.partitions.len(),
                    2,
                    "Expected exactly two partitions"
                );
                Partitioned::Partitioned(
                    partitioned
                        .find_partition(&Partition::RowIdx)
                        .vortex_expect("Missing RowIdx partition")
                        .clone(),
                    partitioned
                        .find_partition(&Partition::Child)
                        .vortex_expect("Missing Child partition")
                        .clone(),
                )
            })
            .clone()
    }
}

#[derive(Clone)]
enum Partitioned {
    // An expression that only references the row index (e.g., `row_idx == 5`).
    RowIdx(ExprRef),
    // An expression that does not reference the row index.
    Child(ExprRef),
    // Contains both the RowIdx and Child expressions.
    Partitioned(ExprRef, ExprRef),
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Partition {
    RowIdx,
    Child,
}

impl Display for Partition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Partition::RowIdx => write!(f, "row_idx"),
            Partition::Child => write!(f, "child"),
        }
    }
}

impl LayoutReader for RowIdxLayoutReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn scope_dtype(&self) -> &ScopeDType {
        self.child.scope_dtype()
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
        // Since RowIdx isn't a field, we only need to register splits for the child layout
        // if there are any fields in the mask at all.
        if !field_mask.is_empty() {
            self.child.register_splits(field_mask, row_offset, splits)?;
        }
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        match &self.partition_expr(expr) {
            Partitioned::RowIdx(expr) => Ok(Box::new(RowIdxEvaluation {
                row_offset: self.row_offset + row_range.start,
                expr: expr.clone(),
            })),
            Partitioned::Child(expr) => self.child.pruning_evaluation(row_range, expr),
            Partitioned::Partitioned(row_idx_expr, child_expr) => {
                Ok(Box::new(PartitionedEvaluation {
                    row_offset: self.row_offset + row_range.start,
                    row_idx_expr: row_idx_expr.clone(),
                    child_expr: child_expr.clone(),
                }) as _)
            }
        }
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        match &self.partition_expr(expr) {
            // Since this is run during pruning, we skip re-evaluating the row index expression
            // during the filter evaluation.
            Partitioned::RowIdx(_) => Ok(Box::new(NoOpMaskEvaluation)),
            Partitioned::Child(expr) => self.child.filter_evaluation(row_range, expr),
            Partitioned::Partitioned(p) => Ok(Box::new(PartitionedEvaluation {
                row_offset: self.row_offset + row_range.start,
                partition: p.clone(),
            }) as _),
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        match &self.partition_expr(expr) {
            Partitioned::RowIdx(expr) => Ok(Box::new(RowIdxEvaluation {
                row_offset: self.row_offset + row_range.start,
                expr: expr.clone(),
            }) as _),
            Partitioned::Child(expr) => self.child.projection_evaluation(row_range, expr),
            Partitioned::Partitioned(p) => Ok(Box::new(PartitionedEvaluation {
                row_offset: self.row_offset + row_range.start,
                partition: p.clone(),
            }) as _),
        }
    }
}

struct RowIdxEvaluation {
    row_offset: u64,
    expr: ExprRef,
}

#[async_trait]
impl PruningEvaluation for RowIdxEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        // Generate a sequence array for the row index.
        let row_idx = SequenceArray::new(
            PValue::U64(self.row_offset),
            PValue::U64(1),
            PType::U64,
            mask.len(),
        )?;

        let result = self.expr.evaluate(
            &Scope::empty(row_idx.len()).with_scope_var(RowIdxVar(row_idx.into_array())),
        )?;

        Mask::try_from(result.as_ref())
    }
}

#[async_trait]
impl ArrayEvaluation for RowIdxEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        // Generate a sequence array for the row index.
        let row_idx = SequenceArray::new(
            PValue::U64(self.row_offset),
            PValue::U64(1),
            PType::U64,
            mask.len(),
        )?;

        // Filter the row index based on the mask.
        let row_idx = filter(row_idx.as_ref(), &mask)?;

        self.expr
            .evaluate(&Scope::empty(row_idx.len()).with_scope_var(RowIdxVar(row_idx.into_array())))
    }
}

struct PartitionedEvaluation {
    row_offset: u64,
    child_eval: ExprRef,
}

#[async_trait]
impl PruningEvaluation for PartitionedEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        todo!()
    }
}

#[async_trait]
impl MaskEvaluation for PartitionedEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        todo!()
    }
}

#[async_trait]
impl ArrayEvaluation for PartitionedEvaluation {
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {}
}
