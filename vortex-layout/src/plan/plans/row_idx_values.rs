// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::FutureExt;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::layouts::row_idx::idx_array;
use crate::plan::Plan;
use crate::plan::PlanArrayFuture;
use crate::plan::PlanChildren;
use crate::plan::PlanExecutionContext;
use crate::plan::PlanId;
use crate::plan::PlanParts;
use crate::plan::PlanVTable;
use crate::plan::check_child_count;

/// Generates the global row indices covering a row domain.
#[derive(Clone, Debug)]
pub struct RowIdxValues;

/// The first global row index and the number of rows generated.
#[derive(Clone, Debug)]
pub struct RowIdxValuesData {
    row_offset: u64,
}

/// A plan that generates global row indices.
pub type RowIdxValuesPlan = Plan<RowIdxValues>;

impl RowIdxValuesPlan {
    /// Creates a row-index values plan covering `row_count` rows from `row_offset`.
    pub fn new(row_offset: u64, row_count: u64) -> Self {
        PlanParts {
            vtable: RowIdxValues,
            dtype: row_idx_dtype(),
            row_count,
            children: PlanChildren::default(),
            data: RowIdxValuesData { row_offset },
        }
        .into_typed()
    }

    /// Returns the global row index assigned to the first row.
    pub fn row_offset(&self) -> u64 {
        self.data().row_offset
    }
}

/// Returns the dtype of a generated row index.
pub fn row_idx_dtype() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

impl PlanVTable for RowIdxValues {
    type PlanData = RowIdxValuesData;
    type Metadata = ProstMetadata<RowIdxValuesPlanMetadata>;

    fn id(&self) -> PlanId {
        static ID: CachedId = CachedId::new("vortex.plan.row_idx_values");
        *ID
    }

    fn metadata(plan: &Plan<Self>) -> Option<Self::Metadata> {
        Some(ProstMetadata(RowIdxValuesPlanMetadata {
            row_offset: plan.data().row_offset,
        }))
    }

    fn with_children(
        _plan: &Plan<Self>,
        children: &PlanChildren,
        _data: &mut Self::PlanData,
    ) -> VortexResult<()> {
        check_child_count("RowIdxValues", children, 0)
    }

    fn execute(
        plan: &Plan<Self>,
        _ctx: &PlanExecutionContext,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<PlanArrayFuture> {
        vortex_ensure!(
            row_range.start <= row_range.end && row_range.end <= plan.row_count(),
            "RowIdxValues row range {:?} is outside 0..{}",
            row_range,
            plan.row_count()
        );
        vortex_ensure!(
            mask.len() == usize::try_from(row_range.end - row_range.start)?,
            "RowIdxValues mask length mismatch"
        );
        let row_offset = plan.row_offset();
        vortex_ensure!(
            row_offset.checked_add(row_range.start).is_some()
                && (row_range.is_empty() || row_offset.checked_add(row_range.end - 1).is_some()),
            "RowIdxValues offset overflows u64"
        );
        let array = idx_array(row_offset, row_range).into_array();
        Ok(async move {
            let mask = mask.await?;
            if mask.all_true() {
                Ok(array)
            } else {
                array.filter(mask)
            }
        }
        .boxed())
    }
}

/// Serialized metadata for a [`RowIdxValues`] plan.
#[derive(Clone, PartialEq, Eq, ::prost::Message)]
pub struct RowIdxValuesPlanMetadata {
    /// The global row index assigned to the first row.
    #[prost(uint64, tag = "1")]
    pub row_offset: u64,
}
