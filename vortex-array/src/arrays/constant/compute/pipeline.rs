// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::operator::{
    BindContext, Operator, OperatorId, OperatorRef, PipelinedOperator,
};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, N};
use crate::vtable::PipelineVTable;

impl PipelineVTable<ConstantVTable> for ConstantVTable {
    fn to_operator(array: &ConstantArray) -> VortexResult<Option<OperatorRef>> {
        Ok(Some(Arc::new(array.clone())))
    }
}

impl Operator for ConstantArray {
    fn id(&self) -> OperatorId {
        self.encoding_id()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.scalar.dtype()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }
}

impl PipelinedOperator for ConstantArray {
    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        debug_assert!(matches!(
            self.dtype(),
            DType::Bool(_) | DType::Primitive(..)
        ));
        match self.scalar.dtype() {
            DType::Bool(_) => Ok(Box::new(BoolConstantKernel {
                remaining: self.len,
                value: self
                    .scalar
                    .as_bool()
                    .value()
                    .vortex_expect("scalar value not bool"),
            })),
            DType::Primitive(..) => Ok(match_each_native_ptype!(
                self.scalar.as_primitive().ptype(),
                |T| {
                    Box::new(ConstantKernel::<T> {
                        remaining: self.len,
                        value: self
                            .scalar
                            .as_primitive()
                            .typed_value::<T>()
                            .vortex_expect("scalar value not of type T"),
                    })
                }
            )),
            _ => todo!(
                "Unsupported scalar type for constant: {:?}",
                self.scalar.dtype()
            ),
        }
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
    }
}

/// Kernel that produces constant primitive values.
pub struct ConstantKernel<T: NativePType> {
    remaining: usize,
    value: T,
}

/// Kernel that produces constant boolean values.
pub struct BoolConstantKernel {
    remaining: usize,
    value: bool,
}

impl<T: Element + NativePType> Kernel for ConstantKernel<T> {
    fn step(&mut self, _ctx: &KernelContext, out: &mut ViewMut) -> VortexResult<()> {
        out.as_slice_mut::<T>()[..N].fill(self.value);
        let len = self.remaining.min(N);
        out.set_len(len);
        Ok(())
    }
}

impl Kernel for BoolConstantKernel {
    fn step(&mut self, _ctx: &KernelContext, out: &mut ViewMut) -> VortexResult<()> {
        out.as_slice_mut::<bool>()[..N].fill(self.value);
        let len = self.remaining.min(N);
        out.set_len(len);
        Ok(())
    }
}
