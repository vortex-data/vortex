// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::{StreamExec, StreamExecRef, StreamNode, StreamNodeRef};
use async_trait::async_trait;
use itertools::Itertools;
use std::collections::VecDeque;
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};
use vortex_mask::Mask;

pub struct ChunkedStreamNode {
    dtype: DType,
    // Does this need to be lazy? It could be I suppose.
    children: Vec<StreamNodeRef>,
}

impl StreamNode for ChunkedStreamNode {
    fn row_count(&self) -> u64 {
        self.children.iter().map(|c| c.row_count()).sum()
    }

    fn execute(&self) -> VortexResult<StreamExecRef> {
        let children: VecDeque<_> = self
            .children
            .iter()
            .map(|child| child.execute())
            .try_collect()?;
        Ok(Box::new(ChunkedStreamExec { children }))
    }
}

pub struct ChunkedStreamExec {
    children: VecDeque<StreamExecRef>,
}

#[async_trait]
impl StreamExec for ChunkedStreamExec {
    fn next_batch_size(&self) -> usize {
        self.children
            .front()
            .map(|child| child.next_batch_size())
            .unwrap_or(0)
    }

    async fn next_batch(&mut self, mask: &Mask) -> VortexResult<ArrayRef> {
        // TODO(ngates): we need to figure out if we need to split the mask or not? We could
        //  guarantee that mask.len() <= next_batch_size() and avoid splitting here.
        assert!(mask.len() <= self.next_batch_size());
        assert!(mask.len() > 0);

        let mut child = self
            .children
            .pop_front()
            .ok_or_else(|| vortex_err!("ChunkedStreamExec has no more children"))?;
        let batch = child.next_batch(mask).await?;

        if child.next_batch_size() > 0 {
            // Put back non-exhausted child
            self.children.push_front(child);
        }

        Ok(batch)
    }
}
