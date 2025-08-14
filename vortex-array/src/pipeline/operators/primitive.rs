// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::task::{Poll, ready};

use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::pipeline::bits::BitView;
use crate::pipeline::buffers::BufferHandle;
use crate::pipeline::operators::{BindContext, Operator};
use crate::pipeline::types::{Element, VType};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext, N};
use crate::vtable::PipelineVTable;

impl PipelineVTable<PrimitiveVTable> for PrimitiveVTable {
    fn to_operator(array: &PrimitiveArray) -> VortexResult<Arc<dyn Operator>> {
        Ok(Arc::new(array.clone()))
    }

    fn to_pipeline(array: &PrimitiveArray) -> VortexResult<Box<dyn Kernel>> {
        todo!()
    }
}

impl Operator for PrimitiveArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vtype(&self) -> VType {
        VType::Primitive(self.ptype())
    }

    fn children(&self) -> &[Arc<dyn Operator>] {
        &[]
    }

    fn with_children(&self, children: Vec<Arc<dyn Operator>>) -> Arc<dyn Operator> {
        Arc::new(self.clone())
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        match_each_native_ptype!(self.ptype(), |T| {
            Ok(Box::new(PrimitiveKernel::<T> {
                buffer: BufferHandle::new(self.buffer()),
                offset: 0,
            }) as Box<dyn Kernel>)
        })
    }
}

impl Hash for PrimitiveArray {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.byte_buffer().as_ptr().hash(state);
        self.ptype().hash(state);
    }
}

/// A kernel that produces primitive values from a byte buffer.
pub struct PrimitiveKernel<T: NativePType> {
    buffer: BufferHandle<T>,
    offset: usize,
}

impl<T: Element + NativePType> Kernel for PrimitiveKernel<T> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.offset = chunk_idx * N;
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn KernelContext,
        mask: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        // FIXME(ngates): support mask.
        assert_eq!(mask.true_count(), N, "Mask must have exactly N true bits");

        let buffer = ready!(self.buffer.get_or_load(ctx))?;
        let remaining = buffer.len() - self.offset;

        if remaining > N {
            out.as_slice_mut::<T>()
                .copy_from_slice(&buffer[self.offset..][..N]);
            self.offset += N;
        } else {
            out.as_slice_mut::<T>()[..remaining].copy_from_slice(&buffer[self.offset..]);
            self.offset += remaining;
        }

        Poll::Ready(Ok(()))
    }
}
