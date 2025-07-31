// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryStreamExt;
use futures::stream::FuturesOrdered;
use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, IntoArray};
use vortex_error::VortexResult;
use vortex_expr::transform::partition::PartitionedExpr;
use vortex_expr::{ExprRef, Scope};
use vortex_mask::Mask;

use crate::ArrayEvaluation;

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
