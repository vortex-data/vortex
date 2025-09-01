// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::sync::Arc;

use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::pipeline::bits::BitView;
use crate::pipeline::operators::{BindContext, Operator, OperatorRef};
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext, PipelineVTable, VType};

impl PipelineVTable<ConstantVTable> for ConstantVTable {
    fn to_operator(array: &ConstantArray) -> VortexResult<Option<OperatorRef>> {
        Ok(ConstantOperator::maybe_new(array.scalar.clone()).map(|c| Arc::new(c) as OperatorRef))
    }
}

/// Pipeline operator for constant arrays that produces the same scalar value for all elements.
#[derive(Debug, Hash)]
pub struct ConstantOperator {
    pub(crate) scalar: Scalar,
}

impl ConstantOperator {
    pub fn maybe_new(scalar: Scalar) -> Option<Self> {
        if scalar.is_null() || !matches!(scalar.dtype(), DType::Bool(_) | DType::Primitive(..)) {
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

    fn children(&self) -> &[OperatorRef] {
        &[]
    }

    fn with_children(&self, _children: Vec<OperatorRef>) -> OperatorRef {
        Arc::new(ConstantOperator::new(self.scalar.clone()))
    }

    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        debug_assert!(matches!(self.vtype(), VType::Bool | VType::Primitive(_)));
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
