// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`AndBoolStreamsPlan`] — k-way per-element AND of bool-stream
//! children. Used by `Scan::build` to combine the per-conjunct mask
//! streams into a single mask that `FilterPlan` can apply.
//!
//! See `LAYOUT_PLAN.md` § Scan construction.

use std::ops::BitAnd as _;
use std::ops::Range;
use std::sync::Arc;

use futures::StreamExt;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;

use crate::v2::aligned::AlignedArrayStream;
use crate::v2::demand::RowDemand;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Combines N bool-stream children into a single bool stream by
/// AND-ing per row. Children's chunk boundaries don't have to line
/// up — the underlying [`AlignedArrayStream`] re-aligns first.
pub struct AndBoolStreamsPlan {
    children: Vec<LayoutPlanRef>,
    output_dtype: DType,
    row_count: u64,
}

impl AndBoolStreamsPlan {
    pub fn new(children: Vec<LayoutPlanRef>, row_count: u64) -> Self {
        debug_assert!(
            !children.is_empty(),
            "AndBoolStreamsPlan needs at least one child"
        );
        debug_assert!(
            children
                .iter()
                .all(|c| matches!(c.schema(), DType::Bool(_))),
            "AndBoolStreamsPlan: every child must produce a Bool stream",
        );
        // The result is always a non-nullable Bool — input nulls
        // are absorbed into the mask (None values are treated as
        // not-matching, same as the v1 filter pipeline).
        let output_dtype = DType::Bool(Nullability::NonNullable);
        Self {
            children,
            output_dtype,
            row_count,
        }
    }
}

impl PartialEq for AndBoolStreamsPlan {
    fn eq(&self, other: &Self) -> bool {
        self.children == other.children
            && self.output_dtype == other.output_dtype
            && self.row_count == other.row_count
    }
}

impl Eq for AndBoolStreamsPlan {}

impl std::hash::Hash for AndBoolStreamsPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.children.hash(state);
        self.output_dtype.hash(state);
        self.row_count.hash(state);
    }
}

impl LayoutPlan for AndBoolStreamsPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= 1 {
            vortex_bail!("AndBoolStreamsPlan partition out of range: {partition}");
        }
        Ok(PartitionStats::for_range(0..self.row_count))
    }

    fn output_ordered(&self) -> bool {
        self.children.iter().all(|c| c.output_ordered())
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true; self.children.len()]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true; self.children.len()]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        &self.children
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != self.children.len() {
            vortex_bail!(
                "AndBoolStreamsPlan::with_new_children expected {} children, got {}",
                self.children.len(),
                children.len()
            );
        }
        Ok(Arc::new(Self {
            children,
            output_dtype: self.output_dtype.clone(),
            row_count: self.row_count,
        }))
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        let mut child_streams = Vec::with_capacity(self.children.len());
        for child in &self.children {
            child_streams.push(child.execute(row_range.clone(), demand, ctx)?);
        }

        let dtype = self.output_dtype.clone();
        let session = ctx.session().clone();
        let aligned = AlignedArrayStream::new(child_streams, ctx.session().handle());
        let mapped = aligned.map(move |result| {
            let arrays = result?;
            // Convert each bool-array chunk to a `Mask`, AND them
            // together, then materialise back to a `BoolArray` so
            // downstream `FilterPlan` can take it as a `Mask`. The
            // Mask round-trip is what the v1 filter pipeline does
            // (see `FlatReader::filter_evaluation`); doing it here
            // keeps the AND operation cheap (bit-level) and lets us
            // shrink the resulting array to its true bits.
            let mut iter = arrays.into_iter();
            let first = iter.next().vortex_expect(
                "AlignedArrayStream output preserves child count and we required >= 1 child",
            );
            let mut ctx = session.create_execution_ctx();
            let mut acc: Mask = first.execute::<Mask>(&mut ctx)?;
            for next in iter {
                let next_mask: Mask = next.execute::<Mask>(&mut ctx)?;
                acc = (&acc).bitand(&next_mask);
            }
            let bits = acc.to_bit_buffer();
            Ok(BoolArray::new(bits, Validity::NonNullable).into_array())
        });

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, mapped)))
    }
}
