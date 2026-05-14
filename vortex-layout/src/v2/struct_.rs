// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`StructPlan`] — field-routing node over a struct layout. Zips
//! per-field child plans positionally.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `StructLayout::plan`.

use std::sync::Arc;

use futures::StreamExt;
use futures::stream;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;

/// Composes child plans positionally; each child produces values for
/// one field, and `StructPlan::execute` zips them into struct arrays.
pub struct StructPlan {
    children: Vec<LayoutPlanRef>,
    field_names: Vec<FieldName>,
    output_dtype: DType,
}

impl StructPlan {
    pub fn new(
        children: Vec<LayoutPlanRef>,
        field_names: Vec<FieldName>,
        output_dtype: DType,
    ) -> Self {
        debug_assert_eq!(
            children.len(),
            field_names.len(),
            "StructPlan: children and field_names must agree"
        );
        Self {
            children,
            field_names,
            output_dtype,
        }
    }

    pub fn field_names(&self) -> &[FieldName] {
        &self.field_names
    }
}

impl LayoutPlan for StructPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        // Children are positionally aligned. Partition counts must
        // match across children; the plan's count is any of them.
        self.children
            .first()
            .map(|c| c.partition_count())
            .unwrap_or(1)
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        // First child's row count is authoritative since children
        // are positionally aligned.
        self.children
            .first()
            .map(|c| c.partition_stats(partition))
            .unwrap_or_else(|| Ok(PartitionStats::unknown()))
    }

    fn output_ordered(&self) -> bool {
        self.children.iter().all(|c| c.output_ordered())
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        // Positional zip — all children must agree on partition order.
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
                "StructPlan::with_new_children expected {} children, got {}",
                self.children.len(),
                children.len()
            );
        }
        Ok(Arc::new(Self {
            children,
            field_names: self.field_names.clone(),
            output_dtype: self.output_dtype.clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        session: &VortexSession,
    ) -> VortexResult<SendableArrayStream> {
        if self.output_dtype.is_nullable() {
            // Nullable structs need a validity child; that wiring lives in
            // StructLayout::plan and arrives in a later PR alongside the
            // expression-routing rewrite.
            vortex_bail!("StructPlan does not yet support nullable structs");
        }

        let mut child_streams = Vec::with_capacity(self.children.len());
        for child in &self.children {
            child_streams.push(child.execute(partition, session)?);
        }

        let names: FieldNames = FieldNames::from(self.field_names.as_slice());
        let dtype = self.output_dtype.clone();

        let zipped = stream::unfold(
            (child_streams, names, dtype.clone()),
            |(mut streams, names, dtype)| async move {
                let mut next_arrays = Vec::with_capacity(streams.len());
                for stream in &mut streams {
                    match stream.next().await {
                        Some(Ok(a)) => next_arrays.push(a),
                        Some(Err(e)) => return Some((Err(e), (streams, names, dtype))),
                        None if next_arrays.is_empty() => return None,
                        None => {
                            return Some((
                                Err(vortex_err!(
                                    "StructPlan child streams emitted different numbers of batches"
                                )),
                                (streams, names, dtype),
                            ));
                        }
                    }
                }
                let len = next_arrays[0].len();
                for a in &next_arrays[1..] {
                    if a.len() != len {
                        return Some((
                            Err(vortex_err!(
                                "StructPlan child arrays have mismatched lengths {} vs {}",
                                len,
                                a.len()
                            )),
                            (streams, names, dtype),
                        ));
                    }
                }
                let result =
                    StructArray::try_new(names.clone(), next_arrays, len, Validity::NonNullable)
                        .map(|s| s.into_array());
                Some((result, (streams, names, dtype)))
            },
        );

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, zipped)))
    }
}
