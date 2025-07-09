//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitAnd, Range};
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::compute::filter;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_expr::{ExprRef, Scope};
use vortex_mask::{AllOr, Mask};
use vortex_utils::aliases::hash_set::HashSet;

use crate::MaskEvaluation;
use crate::layouts::view::reader::{BinaryViewFuture, FetchBuffers};

/// Filter execution for ViewLayout.
///
/// Filter evaluation is only needed using a mask over the views buffer, and then the
/// string buffers can be deserialized independently.
pub(crate) struct ViewFilter {
    pub(crate) row_range: Range<usize>,
    pub(crate) name: Arc<str>,
    pub(crate) dtype: DType,
    pub(crate) expr: ExprRef,
    pub(crate) views: BinaryViewFuture,
    pub(crate) fetch_buffers: FetchBuffers,
}

#[async_trait]
impl MaskEvaluation for ViewFilter {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        let mut views_buffer = self.views.clone().await?;
        if self.row_range.start > 0 || self.row_range.end < views_buffer.len() {
            // Slice the views buffer down to this portion of the split
            views_buffer = views_buffer.slice(self.row_range.start..self.row_range.end);
        }

        // Make a set of all of the buffers that we know we need.
        let mut required_buffers = HashSet::<u32>::new();
        match mask.slices() {
            AllOr::All => {
                for &view in views_buffer.iter() {
                    if !view.is_inlined() {
                        required_buffers.insert(view.as_view().buffer_index());
                    }
                }
            }
            // Check only the sliced elements
            AllOr::Some(slices) => {
                for &(start, end) in slices {
                    for &view in &views_buffer[start..end] {
                        if !view.is_inlined() {
                            required_buffers.insert(view.as_view().buffer_index());
                        }
                    }
                }
            }
            // No buffers needed
            AllOr::None => {}
        }

        // Force each required buffer to be loaded before executing the filter.
        let buffer_count = required_buffers.iter().copied().max().vortex_expect("");
        let mut resolved_buffers = Vec::new();
        for i in 0..buffer_count {
            let idx = i;
            if required_buffers.contains(&idx) {
                resolved_buffers.push(self.fetch_buffers.fetch_buffer(i as usize).await?);
            } else {
                resolved_buffers.push(ByteBuffer::empty());
            }
        }

        let mut array = VarBinViewArray::try_new(
            views_buffer,
            resolved_buffers,
            self.dtype.clone(),
            // TODO(aduffy): FEED IN THE CHILD LAYOUT TO READ THE VALIDITY
            Validity::NonNullable,
        )?
        .into_array();

        let array_mask = if mask.density() < 0.2 {
            // Evaluate only the selected rows of the mask.
            array = filter(&array, &mask)?;
            let array_mask = Mask::try_from(self.expr.evaluate(&Scope::new(array))?.as_ref())?;
            mask.intersect_by_rank(&array_mask)
        } else {
            // Evaluate all rows, avoiding the more expensive rank intersection.
            array = self.expr.evaluate(&Scope::new(array))?;
            let array_mask = Mask::try_from(array.as_ref())?;
            mask.bitand(&array_mask)
        };

        log::debug!(
            "Flat mask evaluation {} - {} (mask = {}) => {}",
            self.name,
            self.expr,
            mask.density(),
            array_mask.density(),
        );

        Ok(array_mask)
    }
}
