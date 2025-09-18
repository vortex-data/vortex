// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::operator::{
    BindContext, LengthBounds, Operator, OperatorId, OperatorRef, PipelinedOperator,
};
use crate::pipeline::bits::BitView;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext};
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

    fn length(&self) -> LengthBounds {
        self.len.into()
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
    value: T,
}

/// Kernel that produces constant boolean values.
pub struct BoolConstantKernel {
    value: bool,
}

impl<T: Element + NativePType> Kernel for ConstantKernel<T> {
    fn step(
        &mut self,
        _ctx: &KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let out_slice = out.as_slice_mut::<T>();
        for i in 0..selected.true_count() {
            out_slice[i] = self.value;
        }
        Ok(())
    }
}

impl Kernel for BoolConstantKernel {
    fn step(
        &mut self,
        _ctx: &KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let out_slice = out.as_slice_mut::<bool>();
        for i in 0..selected.true_count() {
            out_slice[i] = self.value;
        }
        Ok(())
    }
}
