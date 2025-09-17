// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use log;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::operator::canonical::CanonicalExecution;
use crate::operator::{
    BatchBindCtx, BatchExecution, BatchOperator, Operator, OperatorId, OperatorRef,
};
use crate::validity::Validity;
use crate::vtable::{PipelineVTable, ValidityHelper};
use crate::Canonical;

impl PipelineVTable<PrimitiveVTable> for PrimitiveVTable {
    fn to_operator(array: &PrimitiveArray) -> VortexResult<Option<OperatorRef>> {
        if !array.validity().all_valid(array.len()) {
            log::debug!(
                "PipelineVTable::to_operator is not supported for arrays with invalid values"
            );
            return Ok(None);
        }
        Ok(Some(Arc::new(array.clone())))
    }
}

impl Operator for PrimitiveArray {
    fn id(&self) -> OperatorId {
        self.encoding_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.buffer.len() / self.dtype.as_ptype().byte_width()
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }
}

impl BatchOperator for PrimitiveArray {
    fn bind(&self, _ctx: &dyn BatchBindCtx) -> VortexResult<Box<dyn BatchExecution>> {
        Ok(Box::new(CanonicalExecution(Canonical::Primitive(
            PrimitiveArray::from_byte_buffer(
                self.buffer.clone(),
                self.dtype.as_ptype(),
                Validity::AllValid,
            ),
        ))))
    }
}
