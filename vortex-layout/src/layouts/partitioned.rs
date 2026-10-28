// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;
use std::sync::Arc;

use futures::future::try_join_all;
use futures::try_join;
use itertools::Itertools;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::validity::Validity;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_expr::analysis::Annotation;
use vortex_expr::transform::BoundPartitionedExpr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayFuture;

/// A bound partitioned expression, cached together with what its evaluation needs, so that
/// neither has to be rebuilt per split.
pub(crate) struct PartitionedEval<P> {
    expr: BoundPartitionedExpr<P>,
    /// Set when the root only packs partition results, see [`RootFields`].
    root_fields: Option<RootFields>,
}

/// A root of the form `pack(name: $.partition, ...)`, which assembles partition results into a
/// non-nullable struct without computing anything.
///
/// Evaluating it as an expression would build and reduce one scalar function array per field on
/// every split, which dominates projections of wide structs.
struct RootFields {
    names: FieldNames,
    /// The index of the partition backing each output field.
    partitions: Vec<usize>,
}

impl<P> PartitionedEval<P> {
    pub(crate) fn new(mut expr: BoundPartitionedExpr<P>) -> VortexResult<Self>
    where
        P: Annotation,
        FieldName: From<P>,
    {
        expr.collapse_single_partitions()?;
        let root_fields = RootFields::try_new(&expr);
        Ok(Self { expr, root_fields })
    }
}

impl RootFields {
    fn try_new<P>(expr: &BoundPartitionedExpr<P>) -> Option<Self> {
        let pack = expr.root.as_opt::<Pack>()?;
        if pack.nullability.is_nullable() {
            return None;
        }
        let positions: HashMap<&FieldName, usize> = expr
            .partition_names
            .iter()
            .enumerate()
            .map(|(idx, name)| (name, idx))
            .collect();
        let partitions = expr
            .root
            .children()
            .iter()
            .map(|child| {
                let name = child.as_opt::<GetItem>()?;
                if !child.children()[0].is_root() {
                    return None;
                }
                positions.get(name).copied()
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            names: pack.names.clone(),
            partitions,
        })
    }
}

impl<P: Send + Sync + 'static> PartitionedEval<P> {
    pub(crate) fn into_mask_future(
        self: Arc<Self>,
        mask: MaskFuture,
        mask_fn: impl Fn(&P, &BoundExpression, MaskFuture) -> VortexResult<MaskFuture>,
        array_fn: impl Fn(&P, &BoundExpression, MaskFuture) -> VortexResult<ArrayFuture>,
        session: VortexSession,
    ) -> VortexResult<MaskFuture> {
        // Construct evaluations for each child.
        let field_evals: Vec<_> = self
            .expr
            .partition_annotations
            .iter()
            .zip_eq(self.expr.partitions.iter())
            .map(|(annotation, expr)| {
                Ok::<_, VortexError>(
                    if matches!(expr.dtype(), DType::Bool(Nullability::NonNullable)) {
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
                    },
                )
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
                self.expr.partition_names.clone(),
                field_arrays,
                mask.len(),
                Validity::NonNullable,
            )?
            .into_array();

            let mut ctx = session.create_execution_ctx();
            let root_mask = root_scope
                .apply_bound(&self.expr.root)?
                .fill_null(false)?
                .execute::<Mask>(&mut ctx)?;

            let mask = mask.bitand(&root_mask);

            Ok(mask)
        }))
    }

    pub(crate) fn into_array_future(
        self: Arc<Self>,
        mask: MaskFuture,
        array_fn: impl Fn(&P, &BoundExpression, MaskFuture) -> VortexResult<ArrayFuture>,
    ) -> VortexResult<ArrayFuture> {
        // Construct evaluations for each child.
        let field_evals: Vec<_> = self
            .expr
            .partition_annotations
            .iter()
            .zip_eq(self.expr.partitions.iter())
            .map(|(annotation, expr)| array_fn(annotation, expr, mask.clone()))
            .try_collect()?;

        Ok(Box::pin(async move {
            // TODO(ngates): ideally we'd spawn these so the CPU can be utilized more effectively.
            let field_arrays = try_join_all(field_evals);
            let (field_arrays, mask) = try_join!(field_arrays, mask)?;

            if let Some(root_fields) = &self.root_fields {
                let fields = root_fields
                    .partitions
                    .iter()
                    .map(|&idx| field_arrays[idx].clone())
                    .collect::<Vec<_>>();
                return Ok(StructArray::try_new(
                    root_fields.names.clone(),
                    fields,
                    mask.true_count(),
                    Validity::NonNullable,
                )?
                .into_array());
            }

            let root_scope = StructArray::try_new(
                self.expr.partition_names.clone(),
                field_arrays,
                mask.true_count(),
                Validity::NonNullable,
            )?
            .into_array();

            root_scope.apply_bound(&self.expr.root)
        }))
    }
}

enum PartitionEval {
    Mask(MaskFuture),
    Array(ArrayFuture),
}
