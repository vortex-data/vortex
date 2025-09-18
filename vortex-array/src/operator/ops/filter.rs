// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::slice;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::operator::{
    BatchBindCtx, BatchExecutionRef, BatchOperator, BindContext, LengthBounds, Operator,
    OperatorId, OperatorRef, PipelinedOperator,
};
use crate::pipeline::Kernel;
use crate::MaskFuture;

#[derive(Debug)]
pub struct FilterOperator {
    child: OperatorRef,
    mask: Box<LazyMask>,
}

impl Hash for FilterOperator {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.child.hash(state);
        (self.mask.as_ref() as *const LazyMask).hash(state);
    }
}

impl PartialEq for FilterOperator {
    fn eq(&self, other: &Self) -> bool {
        self.child.eq(&other.child) && std::ptr::eq(self.mask.as_ref(), other.mask.as_ref())
    }
}

impl Eq for FilterOperator {}

impl FilterOperator {
    pub fn try_new(child: OperatorRef, mask: LazyMask) -> VortexResult<OperatorRef> {
        // Attempt to push down the filter through any scalar children.
        if child.is_scalar() {
            return child.with_children();
        }

        Ok(Arc::new(FilterOperator {
            child,
            mask: Box::new(mask),
        }))
    }
}

/// A lazy mask that is either ready or pending computation.
///
/// We distinguish between ready and pending masks so that operators can make use of density
/// statistics when making optimization decisions in the case where the mask is known.
#[derive(Clone)]
pub enum LazyMask {
    Ready(Mask),
    Pending(MaskFuture),
}

impl Debug for LazyMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LazyMask::Ready(mask) => f
                .debug_tuple("Ready")
                .field(&format!(
                    "Mask(len={}, count={})",
                    mask.len(),
                    mask.true_count()
                ))
                .finish(),
            LazyMask::Pending(_) => f.debug_tuple("Pending").finish(),
        }
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

    fn length(&self) -> LengthBounds {
        self.child.length()
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
