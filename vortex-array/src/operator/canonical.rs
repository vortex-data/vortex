// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::compute::filter;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, MaskExecution, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use crate::{Array, Canonical};

impl Operator for Canonical {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.canonical")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.as_ref().dtype()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl OperatorHash for Canonical {
    fn operator_hash<H: Hasher>(&self, _state: &mut H) {
        // FIXME(ngates)
    }
}

impl OperatorEq for Canonical {
    fn operator_eq(&self, _other: &Self) -> bool {
        // FIXME(ngates): do something
        false
    }
}

impl BatchOperator for Canonical {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        Ok(Box::new(CanonicalExecution {
            canonical: self.clone(),
            mask: ctx.bind_mask(mask)?,
        }))
    }
}

pub struct CanonicalExecution {
    canonical: Canonical,
    mask: MaskExecution,
}

impl CanonicalExecution {
    pub fn new(canonical: Canonical, mask: MaskExecution) -> Self {
        Self { canonical, mask }
    }
}

#[async_trait]
impl BatchExecution for CanonicalExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let mask = self.mask.await?;
        Ok(if !mask.all_true() {
            filter(self.canonical.as_ref(), &mask)?.to_canonical()
        } else {
            self.canonical
        })
    }
}
