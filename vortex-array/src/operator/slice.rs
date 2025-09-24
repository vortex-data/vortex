// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail};

use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, LengthBounds, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
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
        if range.end > child.bounds().max {
            vortex_bail!(
                "slice range end out of bounds: {} > {}",
                range.end,
                child.bounds().max
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

    fn bounds(&self) -> LengthBounds {
        (self.range.end - self.range.start).into()
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

    fn reduce_children(&self) -> VortexResult<Option<OperatorRef>> {
        // We push down the slice operator to any child that is aligned to the parent.
        let children = (0..self.nchildren())
            .map(|i| {
                let child = self.child.children()[i].clone();

                if self.child.is_selection_target(i).unwrap_or_default() {
                    // Push-down the filter to this child.
                    Ok::<_, VortexError>(Arc::new(SliceOperator::try_new(
                        child,
                        self.range.clone(),
                    )?) as OperatorRef)
                } else {
                    Ok(child)
                }
            })
            .try_collect()?;

        Ok(Some(self.child.clone().with_children(children)?))
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for SliceOperator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        let child_exec = ctx.child(0)?;
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
