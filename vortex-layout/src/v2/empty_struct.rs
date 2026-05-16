// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`EmptyStructPlan`] — terminal node that emits empty struct arrays
//! of the requested row count without reading any field data.
//!
//! Used by `StructLayout::plan` when an expression doesn't reference
//! any fields (e.g. `pack([], …)` or `lit(1)`). The expression's
//! result is row-count-shaped but field-free; planning the children
//! would do per-field I/O for nothing.

use std::ops::Range;
use std::sync::Arc;

use futures::stream;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::v2::dataflow::OutputFrontier;
use crate::v2::demand::RowDemand;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Single-partition plan that emits one chunk of empty struct of the
/// requested length per `execute` call. Schema is `Struct({},
/// NonNullable)`.
pub struct EmptyStructPlan {
    row_count: u64,
    output_dtype: DType,
}

impl EmptyStructPlan {
    pub fn new(row_count: u64) -> Self {
        let output_dtype = DType::Struct(StructFields::empty(), Nullability::NonNullable);
        Self {
            row_count,
            output_dtype,
        }
    }
}

impl PartialEq for EmptyStructPlan {
    fn eq(&self, other: &Self) -> bool {
        self.row_count == other.row_count && self.output_dtype == other.output_dtype
    }
}

impl Eq for EmptyStructPlan {}

impl std::hash::Hash for EmptyStructPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.row_count.hash(state);
        self.output_dtype.hash(state);
    }
}

impl LayoutPlan for EmptyStructPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= 1 {
            vortex_bail!("EmptyStructPlan partition out of range: {partition}");
        }
        Ok(PartitionStats::for_range(0..self.row_count))
    }

    fn output_ordered(&self) -> bool {
        true
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        &[]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if !children.is_empty() {
            vortex_bail!("EmptyStructPlan has no children");
        }
        Ok(self)
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        _demand: &RowDemand,
        _frontier: &OutputFrontier,
        _ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if row_range.start > self.row_count || row_range.end > self.row_count {
            vortex_bail!(
                "EmptyStructPlan::execute row range {row_range:?} exceeds row count {}",
                self.row_count
            );
        }
        let len = usize::try_from(row_range.end - row_range.start).map_err(|_| {
            vortex_error::vortex_err!(
                "EmptyStructPlan::execute row range too large for usize: {row_range:?}",
            )
        })?;
        let array = StructArray::try_new(
            FieldNames::default(),
            Vec::new(),
            len,
            Validity::NonNullable,
        )?
        .into_array();
        let dtype = self.output_dtype.clone();
        let inner = stream::iter(vec![Ok(array)]);
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, inner)))
    }
}
