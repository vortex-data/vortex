// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::ops::Range;

use futures::FutureExt;
use futures::try_join;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanRef;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;

pub(crate) const ROW_IDX_PARTITION_NAME: &str = "row_idx";
pub(crate) const CHILD_PARTITION_NAME: &str = "child";

const ROW_IDX: usize = 0;
const CHILD: usize = 1;

/// Combines independently evaluated row-index and data expression partitions into a struct.
#[derive(Clone, Debug)]
pub struct RowIdxPartition;

/// A plan that pairs a row-index partition with a data partition.
pub type RowIdxPartitionPlan = Plan<RowIdxPartition>;

impl RowIdxPartitionPlan {
    /// Creates a partition plan whose branches share a row domain.
    pub fn try_new(row_idx: PlanRef, child: PlanRef) -> VortexResult<Self> {
        if row_idx.row_count() != child.row_count() {
            vortex_bail!(
                "Row-index partition row count {} does not match child row count {}",
                row_idx.row_count(),
                child.row_count()
            )
        }
        let dtype = DType::Struct(
            StructFields::from_iter([
                (ROW_IDX_PARTITION_NAME, row_idx.dtype().clone()),
                (CHILD_PARTITION_NAME, child.dtype().clone()),
            ]),
            Nullability::NonNullable,
        );
        let row_count = child.row_count();
        Ok(PlanParts {
            vtable: RowIdxPartition,
            dtype,
            row_count,
            children: vec![row_idx, child].into(),
            data: (),
        }
        .into_typed())
    }

    /// Returns the plan evaluating the row-index expression partition.
    pub fn row_idx_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(ROW_IDX)
    }

    /// Returns the plan evaluating the data-child expression partition.
    pub fn child_plan(&self) -> VortexResult<PlanRef> {
        self.child_required(CHILD)
    }
}

impl PlanVTable for RowIdxPartition {
    type PlanData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.row_idx_partition");
        *ID
    }

    fn metadata(_plan: &Plan<Self>) -> Option<Self::Metadata> {
        // The partition dtype is recoverable from the children.
        Some(EmptyMetadata)
    }

    fn with_children(
        plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("RowIdxPartition", children, 2)?;
        let row_idx = children
            .get(ROW_IDX)?
            .ok_or_else(|| vortex_error::vortex_err!("Row-index partition is absent"))?;
        let child = children
            .get(CHILD)?
            .ok_or_else(|| vortex_error::vortex_err!("Data partition is absent"))?;
        if row_idx.row_count() != plan.row_count() || child.row_count() != plan.row_count() {
            vortex_bail!("Row-index partition child row count does not match the plan");
        }
        Ok(())
    }

    fn execute(
        plan: &Plan<Self>,
        ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        let row_idx = plan.row_idx_plan()?.execute(ctx, row_range, mask.clone())?;
        let child = plan.child_plan()?.execute(ctx, row_range, mask)?;
        let names = plan
            .dtype()
            .as_struct_fields_opt()
            .vortex_expect("RowIdxPartition dtype must be a struct")
            .names()
            .clone();
        Ok(async move {
            let (row_idx, child) = try_join!(row_idx, child)?;
            let len = child.len();
            Ok(
                StructArray::try_new(names, vec![row_idx, child], len, Validity::NonNullable)?
                    .into_array(),
            )
        }
        .boxed())
    }

    fn child_name(_plan: &Plan<Self>, index: usize) -> Cow<'_, str> {
        match index {
            ROW_IDX => Cow::Borrowed(ROW_IDX_PARTITION_NAME),
            CHILD => Cow::Borrowed(CHILD_PARTITION_NAME),
            _ => Cow::Owned(format!("child[{index}]")),
        }
    }
}
