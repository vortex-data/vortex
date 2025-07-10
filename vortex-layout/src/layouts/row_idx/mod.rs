// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod expr;

use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::sync::Arc;

use dashmap::DashMap;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::transform::partition::{PartitionedExpr, partition};
use vortex_expr::transform::replace::replace;
use vortex_expr::{ExactExpr, ExprRef, ScopeDType, col, root};

use crate::{ArrayEvaluation, LayoutReader, MaskEvaluation, PruningEvaluation};

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
                // Partition the expression into expressions that can be evaluated over individual fields
                let mut partitioned = partition(expr.clone(), self.dtype(), |expr| expr)
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

#[derive(Clone)]
enum Partitioned {
    // An expression that only references the row index (e.g., `row_idx == 5`).
    RowIdx(ExprRef),
    // An expression that does not reference the row index.
    Child(ExprRef),
    // An expression that references both the row index and other fields.
    Partitioned(Arc<PartitionedExpr<Partition>>),
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
        self.child.register_splits(field_mask, row_offset, splits)
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn PruningEvaluation>> {
        self.child.pruning_evaluation(row_range, expr)
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn MaskEvaluation>> {
        self.child.filter_evaluation(row_range, expr)
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
    ) -> VortexResult<Box<dyn ArrayEvaluation>> {
        self.child.projection_evaluation(row_range, expr)
    }
}
