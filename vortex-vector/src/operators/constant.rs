// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::rc::Rc;

use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::bits::BitView;
use crate::operators::{BindContext, Operator};
use crate::types::{Element, VType};
use crate::view::ViewMut;
use crate::{Kernel, KernelContext};

#[derive(Debug, Hash)]
pub struct ConstantOperator {
    pub(crate) scalar: Scalar,
}

impl ConstantOperator {
    pub fn maybe_new(scalar: Scalar) -> Option<Self> {
        if scalar.is_null() {
            None
        } else {
            Some(Self { scalar })
        }
    }

    pub fn new(scalar: Scalar) -> Self {
        Self::maybe_new(scalar).vortex_expect("scalar cannot be null")
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

    fn children(&self) -> &[Rc<dyn Operator>] {
        &[]
    }

    fn with_children(&self, children: Vec<Rc<dyn Operator>>) -> Rc<dyn Operator> {
        Rc::new(ConstantOperator::new(self.scalar.clone()))
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
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
}

pub struct ConstantKernel<T: NativePType> {
    value: T,
}

pub struct BoolConstantKernel {
    value: bool,
}

impl<T: Element + NativePType> Kernel for ConstantKernel<T> {
    fn step(
        &mut self,
        ctx: &KernelContext,
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
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &KernelContext,
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
