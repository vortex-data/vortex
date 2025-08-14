// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;
use std::task::Poll;

use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::pipeline::bits::BitView;
use crate::pipeline::operators::compare::CompareOp;
use crate::pipeline::operators::{BindContext, Operator};
use crate::pipeline::types::{Element, VType};
use crate::pipeline::vector::VectorId;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext};
use crate::{compute, match_each_compare_op};

#[derive(Debug, Hash)]
pub struct ScalarCompareOperator {
    children: [Arc<dyn Operator>; 1],
    pub op: compute::Operator,
    pub scalar: Scalar,
}

impl ScalarCompareOperator {
    pub fn new(child: Arc<dyn Operator>, op: compute::Operator, scalar: Scalar) -> Self {
        assert_eq!(child.vtype(), VType::Primitive(scalar.dtype().as_ptype()));
        Self {
            children: [child],
            op,
            scalar,
        }
    }
}

impl Operator for ScalarCompareOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn children(&self) -> &[Arc<dyn Operator>] {
        &self.children
    }

    fn vtype(&self) -> VType {
        VType::Bool
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        match self.children[0].vtype() {
            VType::Primitive(ptype) => {
                match_each_native_ptype!(ptype, |T| {
                    match_each_compare_op!(self.op, |Op| {
                        Ok(Box::new(ScalarComparePrimitiveKernel::<T, Op> {
                            lhs: ctx.children()[0],
                            rhs: self
                                .scalar
                                .as_primitive()
                                .typed_value::<T>()
                                .vortex_expect("scalar value not of type T"),
                            _phantom: PhantomData,
                        }) as Box<dyn Kernel>)
                    })
                })
            }
            _ => vortex_bail!(
                "Unsupported type for comparison: {}",
                self.children[0].vtype()
            ),
        }
    }

    fn with_children(&self, mut children: Vec<Arc<dyn Operator>>) -> Arc<dyn Operator> {
        Arc::new(ScalarCompareOperator::new(
            children.remove(0),
            self.op,
            self.scalar.clone(),
        ))
    }
}

struct ScalarComparePrimitiveKernel<T: Element + NativePType, Op: CompareOp<T>> {
    lhs: VectorId,
    rhs: T,
    _phantom: PhantomData<Op>,
}

impl<T: Element + NativePType, Op: CompareOp<T>> Kernel for ScalarComparePrimitiveKernel<T, Op> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let lhs_vec = ctx.vector(self.lhs);
        let lhs = lhs_vec.as_slice::<T>();
        let bools = out.as_slice_mut::<bool>();

        for i in 0..selected.true_count() {
            bools[i] = unsafe { Op::compare(lhs.get_unchecked(i), &self.rhs) };
        }

        Poll::Ready(Ok(()))
    }
}
