// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Range;

use vortex_array::MaskFuture;
use vortex_array::ProstMetadata;
use vortex_array::dtype::FieldName;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::transform::partition_bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::pack::Pack as PackFn;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::layouts::row_idx::RowIdx as RowIdxFn;
use crate::plan::Eval;
use crate::plan::EvalPlan;
use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;
use crate::plan::optimizer::PlanParentReduceRule;
use crate::plan::plans::eval::rewrite_partition_root;
use crate::plan::plans::row_idx_partition::CHILD_PARTITION_NAME;
use crate::plan::plans::row_idx_partition::ROW_IDX_PARTITION_NAME;
use crate::plan::plans::row_idx_partition::RowIdxPartitionPlan;
use crate::plan::plans::row_idx_values::RowIdxValuesPlan;
use crate::plan::plans::row_idx_values::row_idx_dtype;

/// Adds row-index support to its child, offsetting row numbers into the file's row domain.
#[derive(Clone, Debug)]
pub struct RowIdx;

/// The row offset applied to the child domain.
#[derive(Clone, Debug)]
pub struct RowIdxData {
    row_offset: u64,
}

/// A plan that adds row-index support to its child.
pub type RowIdxPlan = Plan<RowIdx>;

impl RowIdxPlan {
    /// Creates a row-index plan with `row_offset` applied to its child domain.
    pub fn new(row_offset: u64, child: PlanRef) -> Self {
        PlanParts {
            vtable: RowIdx,
            dtype: child.dtype().clone(),
            row_count: child.row_count(),
            children: vec![child].into(),
            data: RowIdxData { row_offset },
        }
        .into_typed()
    }

    /// Returns the row offset applied to the child domain.
    pub fn row_offset(&self) -> u64 {
        self.data().row_offset
    }

    /// Returns the child plan.
    pub fn child_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(0)
    }
}

impl PlanVTable for RowIdx {
    type PlanData = RowIdxData;
    type Metadata = ProstMetadata<RowIdxPlanMetadata>;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.row_idx");
        *ID
    }

    fn metadata(plan: &Plan<Self>) -> Option<Self::Metadata> {
        Some(ProstMetadata(RowIdxPlanMetadata {
            row_offset: plan.data().row_offset,
        }))
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("RowIdx", children, 1)?;
        let child = children
            .get(0)?
            .ok_or_else(|| vortex_error::vortex_err!("RowIdx child is absent"))?;
        if child.dtype() != plan.dtype() || child.row_count() != plan.row_count() {
            vortex_error::vortex_bail!(
                "RowIdx child shape changed from ({}, {}) to ({}, {})",
                plan.dtype(),
                plan.row_count(),
                child.dtype(),
                child.row_count()
            );
        }
        Ok(())
    }

    fn execute(
        plan: &Plan<Self>,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        plan.child_plan()?.execute(ctx, row_range, mask)
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        if index == 0 {
            Cow::Borrowed("child")
        } else {
            Cow::Owned(format!("child[{index}]"))
        }
    }
}

/// Serialized metadata for a [`RowIdx`] plan.
#[derive(Clone, PartialEq, Eq, ::prost::Message)]
pub struct RowIdxPlanMetadata {
    /// The row offset applied to the child domain.
    #[prost(uint64, tag = "1")]
    pub row_offset: u64,
}

/// Partitions an expression between generated row indices and the data child.
#[derive(Debug)]
pub(crate) struct ExpressionRowIdxRule;

impl PlanParentReduceRule<RowIdx> for ExpressionRowIdxRule {
    type Parent = Eval;

    fn reduce_parent(
        &self,
        child: &Plan<RowIdx>,
        parent: &Plan<Eval>,
        _child_idx: usize,
    ) -> VortexResult<Option<PlanRef>> {
        let expression = parent.expression();
        let partitioned = partition_bound(expression.clone(), |node| {
            if node
                .as_scalar()
                .is_some_and(|scalar_fn| scalar_fn.is::<RowIdxFn>())
            {
                vec![RowIdxExpressionPartition::RowIdx]
            } else if node.is_root() {
                vec![RowIdxExpressionPartition::Child]
            } else {
                vec![]
            }
        })?;

        // A root-independent expression does not need either side of RowIdx.
        if partitioned.partition_annotations.is_empty() {
            return Ok(Some(
                EvalPlan::try_new(expression.clone(), child.child_plan()?)?.into_plan(),
            ));
        }

        if partitioned.partition_annotations.len() == 1 {
            return match partitioned.partition_annotations[0] {
                RowIdxExpressionPartition::RowIdx => {
                    let expression = replace_row_idx(expression.clone())?;
                    let values =
                        RowIdxValuesPlan::new(child.row_offset(), child.row_count()).into_plan();
                    Ok(Some(EvalPlan::try_new(expression, values)?.into_plan()))
                }
                RowIdxExpressionPartition::Child => Ok(Some(
                    EvalPlan::try_new(expression.clone(), child.child_plan()?)?.into_plan(),
                )),
            };
        }

        if partitioned.partition_annotations.len() != 2 {
            return Ok(None);
        }
        let Some(row_idx_index) = partitioned
            .partition_annotations
            .iter()
            .position(|partition| *partition == RowIdxExpressionPartition::RowIdx)
        else {
            return Ok(None);
        };
        let Some(child_index) = partitioned
            .partition_annotations
            .iter()
            .position(|partition| *partition == RowIdxExpressionPartition::Child)
        else {
            return Ok(None);
        };

        let row_idx_partition = &partitioned.partitions[row_idx_index];
        let child_partition = &partitioned.partitions[child_index];
        let (Some(row_idx_pack), Some(child_pack)) = (
            row_idx_partition
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<PackFn>()),
            child_partition
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<PackFn>()),
        ) else {
            return Ok(None);
        };
        let row_idx_partition_name = partitioned.partition_names[row_idx_index].clone();
        let child_partition_name = partitioned.partition_names[child_index].clone();
        let mut collapsed = Vec::with_capacity(2);

        let row_idx_expression = if row_idx_partition.children().len() == 1 {
            let Some(value_name) = row_idx_pack.names.get(0) else {
                return Ok(None);
            };
            collapsed.push((row_idx_partition_name, value_name.clone()));
            row_idx_partition.children()[0].clone()
        } else {
            row_idx_partition.clone()
        };
        let child_expression = if child_partition.children().len() == 1 {
            let Some(value_name) = child_pack.names.get(0) else {
                return Ok(None);
            };
            collapsed.push((child_partition_name, value_name.clone()));
            child_partition.children()[0].clone()
        } else {
            child_partition.clone()
        };

        let row_idx_expression = replace_row_idx(row_idx_expression)?;
        let row_idx_plan = EvalPlan::try_new(
            row_idx_expression,
            RowIdxValuesPlan::new(child.row_offset(), child.row_count()).into_plan(),
        )?
        .into_plan();
        let child_plan = EvalPlan::try_new(child_expression, child.child_plan()?)?.into_plan();
        let partitions = RowIdxPartitionPlan::try_new(row_idx_plan, child_plan)?;
        let residual =
            rewrite_partition_root(partitioned.root, partitions.dtype().clone(), &collapsed)?;

        Ok(Some(
            EvalPlan::try_new(residual, partitions.into_plan())?.into_plan(),
        ))
    }
}

fn replace_row_idx(expression: BoundExpression) -> VortexResult<BoundExpression> {
    Ok(expression
        .transform_down(|node| {
            if node
                .as_scalar()
                .is_some_and(|scalar_fn| scalar_fn.is::<RowIdxFn>())
            {
                Ok(Transformed {
                    value: BoundExpression::new_root(row_idx_dtype()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum RowIdxExpressionPartition {
    RowIdx,
    Child,
}

impl RowIdxExpressionPartition {
    fn name(self) -> &'static str {
        match self {
            Self::RowIdx => ROW_IDX_PARTITION_NAME,
            Self::Child => CHILD_PARTITION_NAME,
        }
    }
}

impl Display for RowIdxExpressionPartition {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl From<RowIdxExpressionPartition> for FieldName {
    fn from(partition: RowIdxExpressionPartition) -> Self {
        FieldName::from(partition.name())
    }
}
