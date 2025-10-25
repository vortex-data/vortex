// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::{StreamExec, StreamExecRef, StreamNode, StreamNodeRef};
use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_expr::{ExprRef, Scope};
use vortex_mask::Mask;

pub struct ProjectStreamNode {
    expr: ExprRef,
    child: StreamNodeRef,
}

impl StreamNode for ProjectStreamNode {
    fn row_count(&self) -> u64 {
        self.child.row_count()
    }

    fn execute(&self) -> VortexResult<StreamExecRef> {
        Ok(Box::new(ProjectStreamExec {
            expr: self.expr.clone(),
            child: self.child.execute()?,
        }))
    }
}

pub struct ProjectStreamExec {
    expr: ExprRef,
    child: StreamExecRef,
}

#[async_trait]
impl StreamExec for ProjectStreamExec {
    fn next_batch_size(&self) -> usize {
        self.child.next_batch_size()
    }

    async fn next_batch(&mut self, mask: &Mask) -> VortexResult<ArrayRef> {
        let array = self.child.next_batch(mask).await?;
        self.expr.evaluate(&Scope::new(array))
    }
}
