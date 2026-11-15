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
use vortex_array::expr::Expression;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::layouts::row_idx::RowIdx as RowIdxFn;
use crate::partition::PartitionRule;
use crate::partition::PartitionSession;
use crate::partition::PartitionedExpression;
use crate::partition::PartitionedPlan;
use crate::partition::Partitions;
use crate::partition::PlanSlot;
use crate::partition::Slots;
use crate::partition::positional_name;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::PackPlan;
use crate::plan::physical::Plan;
use crate::plan::physical::PlanId;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::plan::physical::PlanSlots;
use crate::plan::physical::PlanVTable;
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
            slots: PlanSlots::default(),
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

    fn with_slots(
        _plan: &Plan<Self>,
        slots: &PlanSlots,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        slots.check_len("RowIdx", 0)
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

#[derive(Clone, Copy)]
pub enum RowIdxSlot {
    Child,
    RowIdx,
}
impl From<RowIdxSlot> for PlanSlot {
    fn from(slot: RowIdxSlot) -> Self {
        match slot {
            RowIdxSlot::Child => 0,
            RowIdxSlot::RowIdx => 1,
        }
    }
}

impl PartitionedPlan for RowIdx {
    type Slot = RowIdxSlot;
    fn slot_dtype(scope: &DType, slot: RowIdxSlot) -> VortexResult<DType> {
        Ok(match slot {
            RowIdxSlot::Child => scope.clone(),
            RowIdxSlot::RowIdx => row_idx_dtype(),
        })
    }
}

pub fn register_row_idx_partition_rules(session: &PartitionSession) {
    session.register_root_rule(RowIdx, &RootRule);
    session.register_rule(RowIdx, RowIdxFn, &RowIdxFnRule);
}

/// row_idx as $ reads its child
struct RootRule;
impl PartitionRule for RootRule {
    type Plan = RowIdx;
    fn apply(
        &self,
        slots: &Slots<'_, RowIdx>,
        _node: &BoundExpression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(slots.read(RowIdxSlot::Child)))
    }
}

/// row_idx not as $ reads a virtual output column
struct RowIdxFnRule;
impl PartitionRule for RowIdxFnRule {
    type Plan = RowIdx;
    fn apply(
        &self,
        slots: &Slots<'_, RowIdx>,
        _node: &BoundExpression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(slots.read(RowIdxSlot::RowIdx)))
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
    let partitioned = match RowIdx.partition(child.dtype(), &expression, session)? {
        PartitionedExpression::One(slot, expr) => {
            let source = if slot == PlanSlot::from(RowIdxSlot::RowIdx) {
                RowIdxPlan::new(row_count).into_plan()
            } else {
                child
            };
            return Ok(EvalPlan::try_new(expr, source)?.into_plan());
        }
        PartitionedExpression::Any(expr) => {
            let source = PackPlan::try_new(
                StructFields::empty(),
                Nullability::NonNullable,
                row_count,
                Vec::new(),
                None,
            )?;
            return Ok(EvalPlan::try_new(expr, source.into_plan())?.into_plan());
        }
        PartitionedExpression::Many(partitioned) => partitioned,
    };
    let Partitions {
        combiner: root,
        partitions,
    } = partitioned;

    let mut plans = Vec::with_capacity(partitions.len());
    for partition in partitions {
        let source = if partition.slot == PlanSlot::from(RowIdxSlot::RowIdx) {
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
