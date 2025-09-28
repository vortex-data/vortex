// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, Operator, OperatorEq,
    OperatorHash, OperatorId, OperatorRef,
};
use crate::{Array, Canonical, IntoArray};

#[derive(Debug, Clone)]
pub struct SliceOperator {
    child: OperatorRef,
    range: Range<usize>,
}

impl SliceOperator {
    pub fn try_new(child: OperatorRef, range: Range<usize>) -> VortexResult<Self> {
        if range.start > range.end {
            vortex_bail!(
                "invalid slice range: start > end ({} > {})",
                range.start,
                range.end
            );
        }
        if range.end > child.len() {
            vortex_bail!(
                "slice range end out of bounds: {} > {}",
                range.end,
                child.len()
            );
        }
        Ok(SliceOperator { child, range })
    }

    pub fn range(&self) -> &Range<usize> {
        &self.range
    }
}

impl OperatorHash for SliceOperator {
    fn operator_hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.child.operator_hash(state);
        self.range.hash(state);
    }
}

impl OperatorEq for SliceOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.range == other.range && self.child.operator_eq(&other.child)
    }
}

impl Operator for SliceOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.slice")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn len(&self) -> usize {
        self.range.end - self.range.start
    }

    fn children(&self) -> &[OperatorRef] {
        std::slice::from_ref(&self.child)
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(SliceOperator::try_new(
            children.into_iter().next().vortex_expect("missing child"),
            self.range.clone(),
        )?))
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for SliceOperator {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        let child_exec = ctx.bind_project(&self.child, Some(mask))?;
        Ok(Box::new(SliceExecution {
            child: child_exec,
            range: self.range.clone(),
        }))
    }
}

struct SliceExecution {
    child: BatchExecutionRef,
    range: Range<usize>,
}

#[async_trait]
impl BatchExecution for SliceExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let child = self.child.execute().await?;
        Ok(child.into_array().slice(self.range).to_canonical())
    }
}
