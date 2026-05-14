// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`DictDecodePlan`] — wraps a child plan that produces dict codes,
//! materialises a [`DictArray`] per chunk against the dict values,
//! and applies the caller's expression.
//!
//! Within one `execute` call the values are awaited once at the start
//! of the output stream and reused for every codes chunk that follows
//! — single producer, single consumer, no shared-future plumbing
//! required. The plan struct itself holds only the values layout and
//! segment source (cheap clones), not any I/O state.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `DictLayout::plan`.

use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
use futures::StreamExt;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::SharedArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::LayoutRef;
use crate::segments::SegmentSource;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Per-execute call: take codes from `child`, await the dict values
/// once at the start of the output stream, then materialise a
/// [`DictArray`] per chunk and apply `expr`.
pub struct DictDecodePlan {
    child: LayoutPlanRef,
    values_layout: LayoutRef,
    segment_source: Arc<dyn SegmentSource>,
    expr: Expression,
    output_dtype: DType,
    all_values_referenced: bool,
}

impl DictDecodePlan {
    pub fn new(
        child: LayoutPlanRef,
        values_layout: LayoutRef,
        segment_source: Arc<dyn SegmentSource>,
        expr: Expression,
        output_dtype: DType,
        all_values_referenced: bool,
    ) -> Self {
        Self {
            child,
            values_layout,
            segment_source,
            expr,
            output_dtype,
            all_values_referenced,
        }
    }
}

impl LayoutPlan for DictDecodePlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        self.child.partition_count()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        // Decoding is row-preserving; row range passes through.
        self.child.partition_stats(partition)
    }

    fn output_ordered(&self) -> bool {
        self.child.output_ordered()
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        std::slice::from_ref(&self.child)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != 1 {
            vortex_bail!(
                "DictDecodePlan::with_new_children expected 1 child, got {}",
                children.len()
            );
        }
        let child = children
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("DictDecodePlan::with_new_children: empty vec"))?;
        Ok(Arc::new(Self {
            child,
            values_layout: Arc::clone(&self.values_layout),
            segment_source: Arc::clone(&self.segment_source),
            expr: self.expr.clone(),
            output_dtype: self.output_dtype.clone(),
            all_values_referenced: self.all_values_referenced,
        }))
    }

    fn execute(&self, row_range: Range<u64>, ctx: &ScanCtx) -> VortexResult<SendableArrayStream> {
        let inner = self.child.execute(row_range, ctx)?;
        let values_reader = self.values_layout.new_reader(
            "v2.dict.values".into(),
            Arc::clone(&self.segment_source),
            ctx.session(),
        )?;
        let values_len = usize::try_from(self.values_layout.row_count())?;
        let values_fut = values_reader.projection_evaluation(
            &(0..self.values_layout.row_count()),
            &root(),
            MaskFuture::new_true(values_len),
        )?;

        let expr = self.expr.clone();
        let dtype = self.output_dtype.clone();
        let all_values_referenced = self.all_values_referenced;
        let stream = try_stream! {
            // Await the values once for the whole execute call. Wrap
            // in `SharedArray` so each chunk's `DictArray::new_unchecked`
            // gets a cheap Arc-clone rather than re-canonicalising.
            let values = SharedArray::new(values_fut.await?).into_array();
            let mut inner = inner;
            while let Some(codes_res) = inner.next().await {
                let codes = codes_res?;
                // SAFETY: matches the v1 `DictReader::projection_evaluation`
                // contract (`vortex-layout/src/layouts/dict/reader.rs:243`):
                // codes dtype is enforced by the codes child reader, and
                // `all_values_referenced` is purely a correctness hint.
                let array = unsafe {
                    DictArray::new_unchecked(codes, values.clone())
                        .set_all_values_referenced(all_values_referenced)
                }
                .into_array()
                .optimize()?;
                yield array.apply(&expr)?;
            }
        };
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}
