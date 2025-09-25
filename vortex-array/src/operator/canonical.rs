// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::compute::filter;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, LengthBounds, Operator,
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

    fn bounds(&self) -> LengthBounds {
        self.as_ref().len().into()
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
            mask: ctx.project_all(mask)?,
        }))
    }
}

pub struct CanonicalExecution {
    canonical: Canonical,
    mask: BatchExecutionRef,
}

impl CanonicalExecution {
    pub fn new(canonical: Canonical, mask: BatchExecutionRef) -> Self {
        Self { canonical, mask }
    }
}

#[async_trait]
impl BatchExecution for CanonicalExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let mask = self.mask.execute().await?.into_bool().to_mask();
        Ok(if !mask.all_true() {
            filter(self.canonical.as_ref(), &mask)?.to_canonical()
        } else {
            self.canonical
        })
    }
}
