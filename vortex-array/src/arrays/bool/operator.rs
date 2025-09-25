// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Formatter;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{BoolArray, BoolVTable};
use crate::operator::canonical::CanonicalExecution;
use crate::operator::{
    BatchBindCtx, BatchExecutionRef, BatchOperator, DisplayFormat, LengthBounds, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use crate::vtable::PipelineVTable;
use crate::{Array, Canonical};

impl PipelineVTable<BoolVTable> for BoolVTable {
    fn to_operator(array: &BoolArray) -> VortexResult<Option<OperatorRef>> {
        Ok(Some(Arc::new(array.clone())))
    }
}

impl OperatorHash for BoolArray {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.hash(state);
        self.buffer.offset().hash(state);
        self.buffer.inner().as_ptr().hash(state);
        self.validity.operator_hash(state);
    }
}

impl OperatorEq for BoolArray {
    fn operator_eq(&self, other: &Self) -> bool {
        self.dtype.eq(other.dtype())
            && self.buffer.offset() == other.buffer.offset()
            && self.buffer.inner().as_ptr() == other.buffer.inner().as_ptr()
            && self.validity.operator_eq(&other.validity)
    }
}

impl Operator for BoolArray {
    fn id(&self) -> OperatorId {
        self.encoding_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        Array::dtype(self.as_ref())
    }

    fn bounds(&self) -> LengthBounds {
        Array::len(self.as_ref()).into()
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn fmt_as(&self, _df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "BoolArray")
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for BoolArray {
    fn project(
        &self,
        mask: &OperatorRef,
        ctx: &mut dyn BatchBindCtx,
    ) -> VortexResult<BatchExecutionRef> {
        let mask = ctx.project_all(mask)?;
        Ok(Box::new(CanonicalExecution::new(
            Canonical::Bool(self.clone()),
            mask,
        )))
    }
}
