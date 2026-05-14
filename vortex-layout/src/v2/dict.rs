// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`DictDecodePlan`] — wraps a child plan that produces dict codes,
//! materialises a [`DictArray`] per chunk against the dict values,
//! and applies the caller's expression.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `DictLayout::plan`.
//! The full design rewrites value predicates into the codes domain
//! (`col = "Alice"` → `codes IN {17}`); `DictDecodePlan` here is the
//! projection-only half — it only handles the projection / value
//! materialisation path.
//!
//! **No values caching here.** Per `LAYOUT_PLAN.md` § Model "Plans
//! are pure descriptions", we don't cache the values across execute
//! calls. Each codes-range execute re-reads the values; the proper
//! fix is `Let` / `Use` (see `LAYOUT_PLAN.md` § Tee and CSE).

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::TryFutureExt;
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
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentSource;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;

/// Per-execute call: take codes from `child`, decode against the
/// freshly-read `values` array, then evaluate `expr` on the
/// materialised `DictArray`.
///
/// `values_layout` is held by reference; reads happen lazily inside
/// `execute`. The values are read once per execute call (not cached
/// across calls, by design).
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
        let child = children.into_iter().next().ok_or_else(|| {
            vortex_error::vortex_err!(
                "DictDecodePlan::with_new_children: len-1 children vec was empty"
            )
        })?;
        Ok(Arc::new(Self {
            child,
            values_layout: Arc::clone(&self.values_layout),
            segment_source: Arc::clone(&self.segment_source),
            expr: self.expr.clone(),
            output_dtype: self.output_dtype.clone(),
            all_values_referenced: self.all_values_referenced,
        }))
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        session: &VortexSession,
    ) -> VortexResult<SendableArrayStream> {
        let inner = self.child.execute(row_range, session)?;

        // Read the dictionary values once for this execute call.
        // The `SharedArrayFuture` shares the read across the chunks
        // emitted by `inner` within this call. We do **not** share
        // across multiple `execute` calls — that's the job of `Let`
        // / `Use` (see `LAYOUT_PLAN.md` § Tee and CSE).
        let values_reader = self.values_layout.new_reader(
            "v2.dict.values".into(),
            Arc::clone(&self.segment_source),
            session,
        )?;
        let values_len = usize::try_from(self.values_layout.row_count())?;
        let values_fut: SharedArrayFuture = values_reader
            .projection_evaluation(
                &(0..self.values_layout.row_count()),
                &root(),
                MaskFuture::new_true(values_len),
            )?
            .map_err(Arc::new)
            .map_ok(|values| SharedArray::new(values).into_array())
            .boxed()
            .shared();

        let expr = self.expr.clone();
        let dtype = self.output_dtype.clone();
        let all_values_referenced = self.all_values_referenced;
        let mapped = inner.then(move |codes_res| {
            let values_fut = values_fut.clone();
            let expr = expr.clone();
            async move {
                let values = values_fut.await.map_err(VortexError::from)?;
                let codes = codes_res?;
                // SAFETY: matches the v1 `DictReader::projection_evaluation`
                // contract (`vortex-layout/src/layouts/dict/reader.rs:243`):
                // codes dtype is enforced by the codes child reader, and
                // `all_values_referenced` is purely a correctness hint.
                let array = unsafe {
                    DictArray::new_unchecked(codes, values)
                        .set_all_values_referenced(all_values_referenced)
                }
                .into_array()
                .optimize()?;
                array.apply(&expr)
            }
        });
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, mapped)))
    }
}
