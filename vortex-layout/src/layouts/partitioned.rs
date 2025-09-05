// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::sync::Arc;

use futures::future::try_join_all;
use futures::try_join;
use itertools::Itertools;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::pipeline::operators::MaskFuture;
use vortex_array::validity::Validity;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexError, VortexResult};
use vortex_expr::transform::PartitionedExpr;
use vortex_expr::{ExprRef, Scope};

use crate::ArrayFuture;

pub trait PartitionedExprEval<P> {
    fn into_mask_future(
        self: Arc<Self>,
        mask: MaskFuture,
        mask_fn: impl Fn(&P, &ExprRef, MaskFuture) -> VortexResult<MaskFuture>,
        array_fn: impl Fn(&P, &ExprRef, MaskFuture) -> VortexResult<ArrayFuture>,
    ) -> VortexResult<MaskFuture>;

    fn into_array_future(
        self: Arc<Self>,
        mask: MaskFuture,
        array_fn: impl Fn(&P, &ExprRef, MaskFuture) -> VortexResult<ArrayFuture>,
    ) -> VortexResult<ArrayFuture>;
}

impl<P: Send + Sync + 'static> PartitionedExprEval<P> for PartitionedExpr<P> {
    fn into_mask_future(
        self: Arc<Self>,
        mask: MaskFuture,
        mask_fn: impl Fn(&P, &ExprRef, MaskFuture) -> VortexResult<MaskFuture>,
        array_fn: impl Fn(&P, &ExprRef, MaskFuture) -> VortexResult<ArrayFuture>,
    ) -> VortexResult<MaskFuture> {
        // Construct evaluations for each child.
        let field_evals: Vec<_> = self
            .partition_annotations
            .iter()
            .zip_eq(self.partitions.iter())
            .zip_eq(self.partition_dtypes.iter())
            .map(|((annotation, expr), dtype)| {
                Ok::<_, VortexError>(if matches!(dtype, DType::Bool(Nullability::NonNullable)) {
                    // If the partition evaluates to a boolean, we can evaluate it as a mask which
                    // can often be more efficient since nulls are turned into `false` early on,
                    // and layouts can perform predicate pruning / indexing.
                    PartitionEval::Mask(mask_fn(annotation, expr, mask.clone())?)
                } else {
                    // Otherwise, we evaluate the projection as an array, and combine the results
                    // at the end.
                    PartitionEval::Array(array_fn(
                        annotation,
                        expr,
                        MaskFuture::new_true(mask.len()),
                    )?)
                })
            })
            .try_collect()?;

        Ok(MaskFuture::new(mask.len(), async move {
            // TODO(ngates): ideally we'd spawn these so the CPU can be utilized more effectively.
            let field_arrays = try_join_all(field_evals.into_iter().map(|eval| async move {
                match eval {
                    PartitionEval::Mask(eval) => Ok(eval.await?.into_array()),
                    PartitionEval::Array(eval) => eval.await,
                }
            }));
            let (field_arrays, mask) = try_join!(field_arrays, mask)?;

            let root_scope = StructArray::try_new(
                self.partition_names.clone(),
                field_arrays,
                mask.len(),
                Validity::NonNullable,
            )?
            .into_array();

            let root_mask = self
                .root
                .evaluate(&Scope::new(root_scope))?
                .try_to_mask_fill_null_false()?;
            let mask = mask.bitand(&root_mask);

            Ok(mask)
        }))
    }

    fn into_array_future(
        self: Arc<Self>,
        mask: MaskFuture,
        array_fn: impl Fn(&P, &ExprRef, MaskFuture) -> VortexResult<ArrayFuture>,
    ) -> VortexResult<ArrayFuture> {
        // Construct evaluations for each child.
        let field_evals: Vec<_> = self
            .partition_annotations
            .iter()
            .zip_eq(self.partitions.iter())
            .map(|(annotation, expr)| array_fn(annotation, expr, mask.clone()))
            .try_collect()?;

        Ok(Box::pin(async move {
            // TODO(ngates): ideally we'd spawn these so the CPU can be utilized more effectively.
            let field_arrays = try_join_all(field_evals);
            let (field_arrays, mask) = try_join!(field_arrays, mask)?;

            let root_scope = StructArray::try_new(
                self.partition_names.clone(),
                field_arrays,
                mask.true_count(),
                Validity::NonNullable,
            )?
            .into_array();

            self.root.evaluate(&Scope::new(root_scope))
        }))
    }
}

enum PartitionEval {
    Mask(MaskFuture),
    Array(ArrayFuture),
}
