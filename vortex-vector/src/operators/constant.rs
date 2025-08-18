// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::task::Poll;

use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::bits::BitView;
use crate::operators::{BindContext, Operator};
use crate::types::{Element, VType};
use crate::view::ViewMut;
use crate::{Kernel, KernelContext};

#[derive(Debug)]
pub struct ConstantOperator {
    pub(crate) scalar: Scalar,
}

impl ConstantOperator {
    pub fn new(scalar: Scalar) -> Self {
        assert!(!scalar.is_null());
        Self { scalar }
    }
}

impl Hash for ConstantOperator {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.scalar.as_ref().hash(state);
    }
}

impl Operator for ConstantOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vtype(&self) -> VType {
        match self.scalar.dtype() {
            DType::Bool(_) => VType::Bool,
            DType::Primitive(p, _) => VType::Primitive(*p),
            DType::Binary(_) => VType::Binary,
            _ => todo!(),
        }
    }

    fn children(&self) -> &[Arc<dyn Operator>] {
        &[]
    }

    fn with_children(&self, children: Vec<Arc<dyn Operator>>) -> Arc<dyn Operator> {
        Arc::new(ConstantOperator::new(self.scalar.clone()))
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        Ok(match_each_native_ptype!(
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
        ))
    }
}

pub struct ConstantKernel<T: NativePType> {
    value: T,
}

impl<T: Element + NativePType> Kernel for ConstantKernel<T> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let out_slice = out.as_slice_mut::<T>();
        for i in 0..selected.true_count() {
            out_slice[i] = self.value;
        }
        Poll::Ready(Ok(()))
    }
}
