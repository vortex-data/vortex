// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{SequenceArray, SequenceVTable};
use num_traits::{ConstOne, PrimInt};
use std::any::Any;
use std::sync::Arc;
use vortex_array::operator::slice::SliceOperator;
use vortex_array::operator::{Operator, OperatorId, OperatorRef};
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{BindContext, Element, Kernel, KernelContext, PipelinedOperator, N};
use vortex_array::vtable::PipelineVTable;
use vortex_array::Array;
use vortex_dtype::{match_each_integer_ptype, DType, NativePType};
use vortex_error::{vortex_err, VortexResult};

impl PipelineVTable<SequenceVTable> for SequenceVTable {
    fn to_operator(array: &SequenceArray) -> VortexResult<Option<OperatorRef>> {
        Ok(Some(Arc::new(array.clone())))
    }
}

impl Operator for SequenceArray {
    fn id(&self) -> OperatorId {
        self.encoding_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        Array::dtype(self.as_ref())
    }

    fn len(&self) -> usize {
        Array::len(self.as_ref())
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }

    fn reduce_parent(
        &self,
        parent: OperatorRef,
        _child_idx: usize,
    ) -> VortexResult<Option<OperatorRef>> {
        // Push down slice
        if let Some(slice) = parent.as_any().downcast_ref::<SliceOperator>() {
            let range = slice.range();
            return Ok(Some(Arc::new(SequenceArray::unchecked_new(
                self.index_value(range.start),
                self.multiplier(),
                self.ptype(),
                self.dtype().nullability(),
                range.len(),
            ))));
        }

        Ok(None)
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        Some(self)
    }
}

impl PipelinedOperator for SequenceArray {
    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        Ok(match_each_integer_ptype!(self.ptype(), |T| {
            if self.multiplier().as_primitive::<T>() == <T as ConstOne>::ONE {
                Box::new(SequenceKernel::<T> {
                    base: self.base().as_primitive::<T>(),
                    len: self.len(),
                    offset: 0,
                })
            } else {
                Box::new(MultiplierSequenceKernel::<T> {
                    base: self.base().as_primitive::<T>(),
                    multiplier: self.multiplier().as_primitive::<T>(),
                    len: self.len(),
                    offset: 0,
                })
            }
        }))
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
    }
}

struct SequenceKernel<T> {
    base: T,
    len: usize,
    offset: usize,
}

impl<T: Element + NativePType + PrimInt> Kernel for SequenceKernel<T> {
    fn step(&mut self, _ctx: &KernelContext, out: &mut ViewMut) -> VortexResult<()> {
        // TODO(ngates): benchmark and optimize this
        let values = out.as_slice_mut::<T>();
        let len = (self.len - self.offset).min(N);
        for i in 0..len {
            values[i] = self.base
                + T::from_usize(self.offset + i)
                    .ok_or_else(|| vortex_err!("Overflow converting usize to ptype"))?;
        }
        out.set_len(len);
        self.offset += len;
        Ok(())
    }
}

struct MultiplierSequenceKernel<T> {
    base: T,
    multiplier: T,
    len: usize,
    offset: usize,
}

impl<T: Element + NativePType + PrimInt> Kernel for MultiplierSequenceKernel<T> {
    fn step(&mut self, _ctx: &KernelContext, out: &mut ViewMut) -> VortexResult<()> {
        // TODO(ngates): benchmark and optimize this. We should use addition not multiplication
        let values = out.as_slice_mut::<T>();
        let len = (self.len - self.offset).min(N);
        for i in 0..len {
            values[i] = self.base
                + self
                    .multiplier
                    .checked_mul(
                        &T::from_usize(self.offset + i)
                            .ok_or_else(|| vortex_err!("Overflow converting usize to ptype"))?,
                    )
                    .ok_or_else(|| vortex_err!("Overflow computing sequence value"))?;
        }
        out.set_len(len);
        self.offset += len;
        Ok(())
    }
}
