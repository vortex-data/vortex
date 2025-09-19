// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::{
    BatchBindCtx, BatchExecutionRef, BatchOperator, BindContext, Operator, OperatorId, OperatorRef,
    PipelinedOperator,
};
use crate::pipeline::Kernel;
use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::slice;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

#[derive(Debug)]
pub struct FilterOperator {
    child: OperatorRef,
    mask: Mask,
}

impl PartialEq for FilterOperator {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child) && self.mask.eq(&other.mask)
    }
}
impl Eq for FilterOperator {}

impl Hash for FilterOperator {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.child.hash(state);
        // Hash the discriminant first
        std::mem::discriminant(&self.mask).hash(state);
        match &self.mask {
            Mask::AllTrue(len) => len.hash(state),
            Mask::AllFalse(len) => len.hash(state),
            Mask::Values(values) => {
                Arc::as_ptr(values).hash(state);
            }
        }
    }
}

impl FilterOperator {
    pub fn try_new(child: OperatorRef, mask: Mask) -> VortexResult<OperatorRef> {
        Ok(Arc::new(FilterOperator { child, mask }))
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
}

impl BatchOperator for FilterOperator {
    fn bind(&self, _ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        todo!()
    }
}

impl PipelinedOperator for FilterOperator {
    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        todo!()
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![0]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
    }
}
