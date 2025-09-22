// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Formatter;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::operator::canonical::CanonicalExecution;
use crate::operator::{
    BatchBindCtx, BatchExecutionRef, BatchOperator, DisplayFormat, Operator, OperatorHash,
    OperatorId, OperatorRef,
};
use crate::vtable::PipelineVTable;
use crate::Canonical;

impl PipelineVTable<PrimitiveVTable> for PrimitiveVTable {
    fn to_operator(array: &PrimitiveArray) -> VortexResult<Option<OperatorRef>> {
        Ok(Some(Arc::new(array.clone())))
    }
}

impl Hash for PrimitiveArray {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        OperatorHash(&self.buffer).hash(state);
        OperatorHash(&self.validity).hash(state);
    }
}

impl PartialEq for PrimitiveArray {
    fn eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && OperatorHash(&self.buffer) == OperatorHash(&other.buffer)
            && OperatorHash(&self.validity) == OperatorHash(&other.validity)
    }
}

impl Eq for PrimitiveArray {}

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

    fn fmt_as(&self, _df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "PrimitiveArray(ptype={})", self.ptype())
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for PrimitiveArray {
    fn bind(&self, _ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        Ok(Box::new(CanonicalExecution(Canonical::Primitive(
            self.clone(),
        ))))
    }
}
