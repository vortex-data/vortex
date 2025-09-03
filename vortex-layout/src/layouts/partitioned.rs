// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::ops::BitAnd;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use futures::try_join;
use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::pipeline::operators::MaskFuture;
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexResult};
use vortex_expr::transform::PartitionedExpr;
use vortex_expr::{ExprRef, Scope};
use vortex_mask::Mask;

use crate::{ArrayEvaluation, MaskEvaluation};

/// An implementation of [`MaskEvaluation`] for partitioned expressions.
pub struct PartitionedMaskEvaluation<P> {
    partitioned: Arc<PartitionedExpr<P>>,
    field_evals: Vec<PartitionEval>,
}

impl<P> PartitionedMaskEvaluation<P> {
    pub fn try_new(
        partitioned: Arc<PartitionedExpr<P>>,
        filter_evaluation: impl Fn(&P, &ExprRef) -> VortexResult<Box<dyn MaskEvaluation>>,
        projection_evaluation: impl Fn(&P, &ExprRef) -> VortexResult<Box<dyn ArrayEvaluation>>,
    ) -> VortexResult<Self> {
        // Construct evaluations for each child.
        let field_evals: Vec<_> = partitioned
            .partition_annotations
            .iter()
            .zip_eq(partitioned.partitions.iter())
            .zip_eq(partitioned.partition_dtypes.iter())
            .map(|((annotation, expr), dtype)| {
                Ok::<_, VortexError>(if matches!(dtype, DType::Bool(Nullability::NonNullable)) {
                    // If the partition evaluates to a boolean, we can evaluate it as a mask which
                    // can often be more efficient since nulls are turned into `false` early on,
                    // and layouts can perform predicate pruning / indexing.
                    PartitionEval::Mask(filter_evaluation(annotation, expr)?)
                } else {
                    // Otherwise, we evaluate the projection as an array, and combine the results
                    // at the end.
                    PartitionEval::Array(projection_evaluation(annotation, expr)?)
                })
            })
            .try_collect()?;

        Ok(Self {
            partitioned,
            field_evals,
        })
    }
}

enum PartitionEval {
    Mask(Box<dyn MaskEvaluation>),
    Array(Box<dyn ArrayEvaluation>),
}

#[async_trait]
impl<P: 'static + Send + Sync> MaskEvaluation for PartitionedMaskEvaluation<P> {
    async fn invoke(&self, mask: MaskFuture) -> VortexResult<Mask> {
        // TODO(ngates): ideally we'd spawn these so the CPU can be utilized more effectively.
        let field_arrays = try_join_all(self.field_evals.iter().map(|eval| {
            let mask = mask.clone();
            async move {
                match eval {
                    PartitionEval::Mask(eval) => Ok(eval.invoke(mask.clone()).await?.into_array()),
                    PartitionEval::Array(eval) => {
                        eval.invoke(MaskFuture::new_true(mask.len())).await
                    }
                }
            }
        }));
        let (field_arrays, mask) = try_join!(field_arrays, mask)?;

        let root_scope = StructArray::try_new(
            self.partitioned.partition_names.clone(),
            field_arrays,
            mask.len(),
            Validity::NonNullable,
        )?
        .into_array();

        let root_mask = self
            .partitioned
            .root
            .evaluate(&Scope::new(root_scope))?
            .try_to_mask_fill_null_false()?;
        let mask = mask.bitand(&root_mask);

        Ok(mask)
    }
}

/// An implementation of [`ArrayEvaluation`] for partitioned expressions.
pub struct PartitionedArrayEvaluation<P> {
    partitioned: Arc<PartitionedExpr<P>>,
    field_evals: Vec<Box<dyn ArrayEvaluation>>,
}

impl<P> PartitionedArrayEvaluation<P> {
    pub fn try_new(
        partitioned: Arc<PartitionedExpr<P>>,
        projection_evaluation: impl Fn(&P, &ExprRef) -> VortexResult<Box<dyn ArrayEvaluation>>,
    ) -> VortexResult<Self> {
        // Construct evaluations for each child.
        let field_evals: Vec<_> = partitioned
            .partition_annotations
            .iter()
            .zip_eq(partitioned.partitions.iter())
            .map(|(annotation, expr)| projection_evaluation(annotation, expr))
            .try_collect()?;

        Ok(Self {
            partitioned,
            field_evals,
        })
    }
}

#[async_trait]
impl<P: 'static + Send + Sync + Display> ArrayEvaluation for PartitionedArrayEvaluation<P> {
    async fn invoke(&self, mask: MaskFuture) -> VortexResult<ArrayRef> {
        let field_arrays = try_join_all(
            self.field_evals
                .iter()
                .map(|eval| eval.invoke(mask.clone())),
        );
        let (field_arrays, mask) = try_join!(field_arrays, mask)?;

        let root_scope = StructArray::try_new(
            self.partitioned.partition_names.clone(),
            field_arrays,
            mask.true_count(),
            Validity::NonNullable,
        )?
        .into_array();

        self.partitioned.root.evaluate(&Scope::new(root_scope))
    }
}
