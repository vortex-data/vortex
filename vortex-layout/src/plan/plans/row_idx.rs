// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::BoundExpressionRef;
use vortex_array::expr::transform::partition_bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::pack::Pack as PackFn;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::layouts::row_idx::RowIdx as RowIdxFn;
use crate::plan::EvalPlan;
use crate::plan::PackPlan;
use crate::plan::Plan;
use crate::plan::PlanChildren;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;
use crate::plan::plans::pack::rewrite_partition_root;

const ROW_IDX_PARTITION_NAME: &str = "row_idx";
const CHILD_PARTITION_NAME: &str = "child";

/// Generates global row indices for the current execution row domain.
#[derive(Clone, Debug)]
pub struct RowIdx;

/// Operator-specific data for a [`RowIdx`] plan.
#[derive(Clone, Debug)]
pub struct RowIdxData;

/// A childless source of global row indices supplied by its execution row domain.
pub type RowIdxPlan = Plan<RowIdx>;

impl RowIdxPlan {
    /// Creates a row-index source covering `row_count` rows.
    pub fn new(row_count: u64) -> Self {
        PlanParts {
            vtable: RowIdx,
            dtype: row_idx_dtype(),
            row_count,
            children: PlanChildren::default(),
            data: RowIdxData,
        }
        .into_typed()
    }
}

/// Returns the dtype of a generated row index.
pub fn row_idx_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

impl PlanVTable for RowIdx {
    type PlanData = RowIdxData;
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.row_idx");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn with_children(
        _plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("RowIdx", children, 0)
    }
}

/// Plans an expression over a data source and its global row-index domain.
///
/// Expressions that only use data bypass the row-index source, expressions that only use
/// `#row_idx` bypass `child`, and mixed expressions combine independently planned branches with a
/// [`PackPlan`]. The file row offset is supplied by the execution row domain.
pub fn plan_row_idx_expression(
    expression: BoundExpressionRef,
    child: PlanRef,
) -> VortexResult<PlanRef> {
    let partitioned = partition_bound(Arc::clone(&expression), |node| {
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

    if partitioned.partition_annotations.is_empty() {
        let row_domain = PackPlan::try_new(
            StructFields::empty(),
            Nullability::NonNullable,
            child.row_count(),
            Vec::new(),
            None,
        )?
        .into_plan();
        return Ok(EvalPlan::try_new(expression, row_domain)?.into_plan());
    }

    if partitioned.partition_annotations.len() == 1 {
        return match partitioned.partition_annotations[0] {
            RowIdxExpressionPartition::RowIdx => {
                let expression = replace_row_idx(expression)?;
                let values = RowIdxPlan::new(child.row_count()).into_plan();
                Ok(EvalPlan::try_new(expression, values)?.into_plan())
            }
            RowIdxExpressionPartition::Child => {
                Ok(EvalPlan::try_new(expression, child)?.into_plan())
            }
        };
    }

    vortex_ensure!(
        partitioned.partition_annotations.len() == 2,
        "Row-index expression produced more than two partitions"
    );
    let row_idx_index = partitioned
        .partition_annotations
        .iter()
        .position(|partition| *partition == RowIdxExpressionPartition::RowIdx)
        .ok_or_else(|| vortex_err!("Row-index expression has no row-index partition"))?;
    let child_index = partitioned
        .partition_annotations
        .iter()
        .position(|partition| *partition == RowIdxExpressionPartition::Child)
        .ok_or_else(|| vortex_err!("Row-index expression has no data partition"))?;

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
        return Err(vortex_err!(
            "Row-index expression partitions must be struct packs"
        ));
    };
    let row_idx_partition_name = partitioned.partition_names[row_idx_index].clone();
    let child_partition_name = partitioned.partition_names[child_index].clone();
    let mut collapsed = Vec::with_capacity(2);

    let row_idx_expression = if row_idx_partition.children().len() == 1 {
        let Some(value_name) = row_idx_pack.names.get(0) else {
            return Err(vortex_err!("Row-index expression partition is empty"));
        };
        collapsed.push((row_idx_partition_name, value_name.clone()));
        Arc::clone(&row_idx_partition.children()[0])
    } else {
        Arc::clone(row_idx_partition)
    };
    let child_expression = if child_partition.children().len() == 1 {
        let Some(value_name) = child_pack.names.get(0) else {
            return Err(vortex_err!("Data expression partition is empty"));
        };
        collapsed.push((child_partition_name, value_name.clone()));
        Arc::clone(&child_partition.children()[0])
    } else {
        Arc::clone(child_partition)
    };

    let row_count = child.row_count();
    let row_idx_expression = replace_row_idx(row_idx_expression)?;
    let row_idx_plan =
        EvalPlan::try_new(row_idx_expression, RowIdxPlan::new(row_count).into_plan())?.into_plan();
    let child_plan = EvalPlan::try_new(child_expression, child)?.into_plan();
    let fields = StructFields::from_iter([
        (ROW_IDX_PARTITION_NAME, row_idx_plan.dtype().clone()),
        (CHILD_PARTITION_NAME, child_plan.dtype().clone()),
    ]);
    let partitions = PackPlan::try_new(
        fields,
        Nullability::NonNullable,
        row_count,
        vec![row_idx_plan, child_plan],
        None,
    )?;
    let residual =
        rewrite_partition_root(partitioned.root, partitions.dtype().clone(), &collapsed)?;

    Ok(EvalPlan::try_new(residual, partitions.into_plan())?.into_plan())
}

fn replace_row_idx(expression: BoundExpressionRef) -> VortexResult<BoundExpressionRef> {
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
