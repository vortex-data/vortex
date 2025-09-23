// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::hash::Hasher;
use std::slice;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::compute::filter;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, Operator, OperatorEq,
    OperatorHash, OperatorId, OperatorRef,
};
use crate::{Array, Canonical, IntoArray};

#[derive(Debug)]
pub struct FilterOperator {
    child: OperatorRef,
    mask: Mask,
}

impl OperatorEq for FilterOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.child.operator_eq(&other.child) && self.mask.operator_eq(&other.mask)
    }
}

impl OperatorHash for FilterOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.child.operator_hash(state);
        self.mask.operator_hash(state);
    }
}

impl FilterOperator {
    pub fn new(child: OperatorRef, mask: Mask) -> FilterOperator {
        assert_eq!(
            child.len(),
            mask.len(),
            "Mask length must match child length"
        );
        FilterOperator { child, mask }
    }

    pub fn mask(&self) -> &Mask {
        &self.mask
    }
}

impl Operator for FilterOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.filter")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn len(&self) -> usize {
        self.mask.true_count()
    }

    fn children(&self) -> &[OperatorRef] {
        slice::from_ref(&self.child)
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(Arc::new(FilterOperator {
            child: children.into_iter().next().vortex_expect("missing child"),
            mask: self.mask.clone(),
        }))
    }

    fn reduce_children(&self) -> VortexResult<Option<OperatorRef>> {
        // If none of the children are position-preserving, we cannot push down the filter.
        if !self
            .child
            .children()
            .iter()
            .enumerate()
            .any(|(i, child)| child.is_position_preserving(i).unwrap_or_default())
        {
            return Ok(None);
        }

        // We push down the filter operator to any child that is aligned to the parent.
        let children = (0..self.child.nchildren())
            .map(|i| {
                let child = self.child.children()[i].clone();

                if child.is_position_preserving(i).unwrap_or_default() {
                    // Push-down the filter to this child.
                    Arc::new(FilterOperator::new(child, self.mask.clone()))
                } else {
                    child
                }
            })
            .collect();

        Ok(Some(self.child.clone().with_children(children)?))
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }

    // fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
    //     TODO(ngates): do we decide if we're super sparse that we should be batch executed?
    //      Seems like a weird place to make that decision though... Although we do have all the
    //      information here. Pushdown has already happened, so we know exactly which is faster.
    // Some(self)
    // }
}

impl BatchOperator for FilterOperator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        Ok(Box::new(FilterExecution {
            child: ctx.take_child(0)?,
            mask: self.mask.clone(),
        }) as BatchExecutionRef)
    }
}

struct FilterExecution {
    child: BatchExecutionRef,
    mask: Mask,
}

#[async_trait]
impl BatchExecution for FilterExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let child = self.child.execute().await?;
        // TODO(ngates): obviously inline all canonical implementations here
        Ok(filter(child.into_array().as_ref(), &self.mask)?.to_canonical())
    }
}
