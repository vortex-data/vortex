// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::ops::BitAnd;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::FuturesOrdered;
use futures::TryStreamExt;
use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexResult};
use vortex_expr::transform::PartitionedExpr;
use vortex_expr::{ExprRef, Scope};
use vortex_mask::Mask;

use crate::{ArrayEvaluation, MaskEvaluation};

/// An implementation of [`MaskEvaluation`] for partitioned expressions.
pub struct PartitionedMaskEvaluation<'handle, P> {
    partitioned: Arc<PartitionedExpr<P>>,
    field_evals: Vec<PartitionEval<'handle>>,
}

impl<'handle, P> PartitionedMaskEvaluation<'handle, P> {
    pub fn try_new(
        partitioned: Arc<PartitionedExpr<P>>,
        filter_evaluation: impl Fn(&P, &ExprRef) -> VortexResult<Box<dyn MaskEvaluation<'handle>>>,
        projection_evaluation: impl Fn(&P, &ExprRef) -> VortexResult<Box<dyn ArrayEvaluation<'handle>>>,
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

enum PartitionEval<'handle> {
    Mask(Box<dyn MaskEvaluation<'handle>>),
    Array(Box<dyn ArrayEvaluation<'handle>>),
}

#[async_trait]
impl<'handle, P: 'static + Send + Sync> MaskEvaluation<'handle>
    for PartitionedMaskEvaluation<'handle, P>
{
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        // TODO(ngates): ideally we'd spawn these so the CPU can be utilized more effectively.
        let field_arrays: Vec<_> = FuturesOrdered::from_iter(self.field_evals.iter().map(|eval| {
            let mask = mask.clone();
            async move {
                match eval {
                    PartitionEval::Mask(eval) => Ok(eval.invoke(mask.clone()).await?.into_array()),
                    PartitionEval::Array(eval) => eval.invoke(Mask::new_true(mask.len())).await,
                }
            }
        }))
        .try_collect()
        .await?;

        let root_scope = StructArray::try_new(
            self.partitioned.partition_names.clone(),
            field_arrays,
            mask.len(),
            Validity::NonNullable,
        )?
        .into_array();

        let root_mask = Mask::try_from(
            self.partitioned
                .root
                .evaluate(&Scope::new(root_scope))?
                .as_ref(),
        )?;
        let mask = mask.bitand(&root_mask);

        Ok(mask)
    }
}

/// An implementation of [`ArrayEvaluation`] for partitioned expressions.
pub struct PartitionedArrayEvaluation<'handle, P> {
    partitioned: Arc<PartitionedExpr<P>>,
    field_evals: Vec<Box<dyn ArrayEvaluation<'handle>>>,
}

impl<'handle, P> PartitionedArrayEvaluation<'handle, P> {
    pub fn try_new(
        partitioned: Arc<PartitionedExpr<P>>,
        projection_evaluation: impl Fn(&P, &ExprRef) -> VortexResult<Box<dyn ArrayEvaluation<'handle>>>,
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
impl<'handle, P: 'static + Send + Sync + Display> ArrayEvaluation<'handle>
    for PartitionedArrayEvaluation<'handle, P>
{
    async fn invoke(&self, mask: Mask) -> VortexResult<ArrayRef> {
        let field_arrays: Vec<_> = FuturesOrdered::from_iter(
            self.field_evals
                .iter()
                .map(|eval| eval.invoke(mask.clone())),
        )
        .try_collect()
        .await?;

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
