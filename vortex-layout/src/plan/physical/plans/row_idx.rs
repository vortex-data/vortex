// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::layouts::row_idx::RowIdx as RowIdxFn;
use crate::partition::PartitionedExpression;
use crate::partition::Partitions;
use crate::partition::partition_expression;
use crate::partition::positional_name;
use crate::partition_session::FieldIndex;
use crate::partition_session::NodeTouches;
use crate::partition_session::PartitionRule;
use crate::partition_session::PartitionSession;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::PackPlan;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanChildren;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanVTable;
use crate::plan::physical::check_child_count;
use crate::plan::physical::operators::OnceSource;
use crate::plan::physical::pipeline::Chain;
use crate::plan::physical::pipeline::Compiler;

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

    fn compile(
        plan: &Plan<Self>,
        rows: Range<u64>,
        mask: &Mask,
        compiler: &mut Compiler<'_>,
    ) -> VortexResult<Option<Chain>> {
        let _ = plan;
        if mask.all_false() {
            return Ok(None);
        }
        let offset = compiler.row_offset();
        let indices = Buffer::from_iter(rows.start + offset..rows.end + offset).into_array();
        let indices = if mask.all_true() {
            indices
        } else {
            indices.filter(mask.clone())?
        };
        Ok(Some(Chain::new(OnceSource::new(indices))))
    }
}

const CHILD_SLOT: FieldIndex = 0;
const ROW_IDX_SLOT: FieldIndex = FieldIndex::MAX;

pub(crate) fn register_row_idx_partition_rules(session: &PartitionSession) {
    session.register_root(RowIdx.id(), &RowIdxRootRule);
    session.register(RowIdx.id(), RowIdxFn.id(), &RowIdxFnRule);
}

/// row_idx as $ reads its child
struct RowIdxRootRule;
impl PartitionRule for RowIdxRootRule {
    fn apply(&self, node: &BoundExpression) -> VortexResult<Option<NodeTouches>> {
        let slots = vec![CHILD_SLOT];
        let replacement = BoundExpression::new_root(node.dtype().clone());
        Ok(Some(NodeTouches { slots, replacement }))
    }
}

/// row_idx as a non-root reads nothing and generates a virtual output slot
struct RowIdxFnRule;
impl PartitionRule for RowIdxFnRule {
    fn apply(&self, _node: &BoundExpression) -> VortexResult<Option<NodeTouches>> {
        let slots = vec![ROW_IDX_SLOT];
        let replacement = BoundExpression::new_root(row_idx_dtype());
        Ok(Some(NodeTouches { slots, replacement }))
    }
}

/// Plans an expression over a data source and its global row-index domain.
///
/// Expressions that only use data bypass the row-index source, expressions that only use
/// `#row_idx` bypass `child`, and mixed expressions combine independently planned branches with a
/// [`PackPlan`]. The file row offset is supplied by the execution row domain.
pub fn plan_row_idx_expression(
    expression: BoundExpression,
    child: PlanRef,
    session: &VortexSession,
) -> VortexResult<PlanRef> {
    let row_count = child.row_count();
    let partitioned = match partition_expression(&expression, RowIdx.id(), session)? {
        PartitionedExpression::One(slot, expr) => {
            let source = match slot {
                Some(ROW_IDX_SLOT) => RowIdxPlan::new(row_count).into_plan(),
                Some(_) => child,
                None => PackPlan::try_new(
                    StructFields::empty(),
                    Nullability::NonNullable,
                    row_count,
                    Vec::new(),
                    None,
                )?
                .into_plan(),
            };
            return Ok(EvalPlan::try_new(expr, source)?.into_plan());
        }
        PartitionedExpression::Many(partitioned) => partitioned,
    };
    let Partitions { root, partitions } = partitioned;

    let mut plans = Vec::with_capacity(partitions.len());
    for partition in partitions {
        let source = if partition.slot == ROW_IDX_SLOT {
            RowIdxPlan::new(row_count).into_plan()
        } else {
            child.clone()
        };
        plans.push(EvalPlan::try_new(partition.expr, source)?.into_plan());
    }

    let fields = StructFields::from_iter(
        plans
            .iter()
            .enumerate()
            .map(|(position, plan)| (positional_name(position), plan.dtype().clone())),
    );
    let partitions = PackPlan::try_new(fields, Nullability::NonNullable, row_count, plans, None)?;

    Ok(EvalPlan::try_new(root, partitions.into_plan())?.into_plan())
}
