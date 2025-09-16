// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::{
    ArrayOperator, BatchBindCtx, BatchExecution, BatchOperator, PipelinedOperator,
};
use crate::pipeline::operators::{BindContext, MaskFuture};
use crate::pipeline::Kernel;
use std::any::Any;
use std::slice;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

pub struct FilterOperator {
    child: Arc<dyn ArrayOperator>,
    mask: LazyMask,
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

impl ArrayOperator for FilterOperator {
    fn id(&self) -> Arc<str> {
        Arc::from("vortex.filter")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    fn len(&self) -> usize {
        self.child.len()
    }

    fn children(&self) -> &[Arc<dyn ArrayOperator>] {
        slice::from_ref(&self.child)
    }

    fn with_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ArrayOperator>>,
    ) -> VortexResult<Arc<dyn ArrayOperator>> {
        Ok(Arc::new(FilterOperator {
            child: children.into_iter().next().vortex_expect("missing child"),
            mask: self.mask.clone(),
        }))
    }
}

impl BatchOperator for FilterOperator {
    fn bind(&self, ctx: &dyn BatchBindCtx) -> VortexResult<Box<dyn BatchExecution>> {
        todo!()
    }
}

impl PipelinedOperator for FilterOperator {
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        todo!()
    }
}
