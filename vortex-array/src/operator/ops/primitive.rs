// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PrimitiveArray;
use crate::operator::ops::canonical::CanonicalExecution;
use crate::operator::{ArrayOperator, BatchBindCtx, BatchExecution, BatchOperator};
use crate::validity::Validity;
use crate::Canonical;
use std::any::Any;
use std::sync::Arc;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;

/// An operator for a primitive array.
struct PrimitiveOperator {
    dtype: DType,
    buffer: ByteBuffer,
}

impl ArrayOperator for PrimitiveOperator {
    fn id(&self) -> Arc<str> {
        Arc::from("vortex.primitive")
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

    fn children(&self) -> &[Arc<dyn ArrayOperator>] {
        &[]
    }

    fn with_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ArrayOperator>>,
    ) -> VortexResult<Arc<dyn ArrayOperator>> {
        Ok(self)
    }

    fn as_batch(&self) -> Option<&dyn BatchOperator> {
        Some(self)
    }
}

impl BatchOperator for PrimitiveOperator {
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
