// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::{StreamExec, StreamExecRef, StreamNode};
use async_trait::async_trait;
use std::sync::Arc;
use vortex_array::compute::filter;
use vortex_array::serde::ArrayParts;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};
use vortex_mask::Mask;

pub struct FlatLayoutStreamNode {
    len: usize,
    dtype: DType,
    ctx: ArrayContext,

    segment_id: SegmentId,
    segment_source: Arc<dyn SegmentSource>,
}

impl StreamNode for FlatLayoutStreamNode {
    fn row_count(&self) -> u64 {
        self.len as u64
    }

    fn execute(&self) -> VortexResult<StreamExecRef> {
        let fut = self.segment_source.request(self.segment_id);
        Ok(Box::new(FlatLayoutStreamExec {
            len: self.len,
            dtype: self.dtype.clone(),
            ctx: self.ctx.clone(),
            segment_future: Some(fut),
            array: None,
            offset: 0,
        }))
    }
}

pub struct FlatLayoutStreamExec {
    len: usize,
    dtype: DType,
    ctx: ArrayContext,
    segment_future: Option<SegmentFuture>,

    array: Option<ArrayRef>,
    offset: usize,
}

#[async_trait]
impl StreamExec for FlatLayoutStreamExec {
    fn next_batch_size(&self) -> usize {
        self.len
    }

    async fn next_batch(&mut self, mask: &Mask) -> VortexResult<ArrayRef> {
        let array = match &self.array {
            Some(array) => array,
            None => {
                let fut = self
                    .segment_future
                    .take()
                    .vortex_expect("Segment future must be present for first batch");
                let segment = fut.await?;
                let array =
                    ArrayParts::try_from(segment)?.decode(&self.ctx, &self.dtype, self.len)?;
                self.array = Some(array);
                self.array.as_ref().vortex_expect("just written")
            }
        };

        let array = array.slice(self.offset..self.offset + mask.len());
        self.offset += mask.len();

        let array = filter(&array, &mask)?;
        Ok(array)
    }
}
