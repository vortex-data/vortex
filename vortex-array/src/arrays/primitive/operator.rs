// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Formatter;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::operator::canonical::CanonicalExecution;
use crate::operator::{
    BatchBindCtx, BatchExecutionRef, BatchOperator, DisplayFormat, LengthBounds, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use crate::vtable::PipelineVTable;

impl PipelineVTable<PrimitiveVTable> for PrimitiveVTable {
    fn to_operator(array: &PrimitiveArray) -> VortexResult<Option<OperatorRef>> {
        Ok(Some(Arc::new(array.clone())))
    }
}

impl OperatorHash for PrimitiveArray {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.buffer.operator_hash(state);
        self.validity.operator_hash(state);
    }
}

impl OperatorEq for PrimitiveArray {
    fn operator_eq(&self, other: &Self) -> bool {
        self.dtype == other.dtype
            && self.buffer.operator_eq(&other.buffer)
            && self.validity.operator_eq(&other.validity)
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

    fn bounds(&self) -> LengthBounds {
        LengthBounds::from(self.buffer.len() / self.dtype.as_ptype().byte_width())
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
