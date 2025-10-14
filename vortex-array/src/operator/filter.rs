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
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, LengthBounds, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
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
        assert!(
            child.bounds().contains(mask.len()),
            "Mask length must be within child bounds"
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

    fn bounds(&self) -> LengthBounds {
        self.mask.true_count().into()
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
        // We need selection target information to be defined for all children.
        let Some(selection_targets): Option<Vec<_>> = self
            .child
            .children()
            .iter()
            .enumerate()
            .map(|(i, child)| child.is_selection_target(i))
            .collect()
        else {
            return Ok(None);
        };

        // Selection is defined to be false for all children, so we cannot push down the
        // filter.
        if selection_targets.iter().all(|s| !s) {
            return Ok(None);
        }

        // Otherwise, we push down the filter to all children that are selection targets.
        let children = self
            .child
            .children()
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, child)| {
                if selection_targets[i] {
                    // Push-down the filter to this child.
                    Arc::new(FilterOperator::new(child, self.mask.clone())) as OperatorRef
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
}

impl BatchOperator for FilterOperator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        Ok(Box::new(FilterExecution {
            child: ctx.child(0)?,
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
