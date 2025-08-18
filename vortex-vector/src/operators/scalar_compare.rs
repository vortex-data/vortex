// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::marker::PhantomData;
use std::rc::Rc;
use std::task::Poll;

use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::bits::BitView;
use crate::operators::compare::{BinaryOperator, CompareOp};
use crate::operators::{BindContext, Operator};
use crate::types::{Element, VType};
use crate::vector::VectorId;
use crate::view::ViewMut;
use crate::{Kernel, KernelContext, match_each_compare_op};

#[derive(Debug, Hash)]
pub struct ScalarCompareOperator {
    children: [Rc<dyn Operator>; 1],
    pub op: BinaryOperator,
    pub scalar: Scalar,
}

impl ScalarCompareOperator {
    pub fn new(child: Rc<dyn Operator>, op: BinaryOperator, scalar: Scalar) -> Self {
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

    fn children(&self) -> &[Rc<dyn Operator>] {
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

    fn with_children(&self, mut children: Vec<Rc<dyn Operator>>) -> Rc<dyn Operator> {
        Rc::new(ScalarCompareOperator::new(
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

        assert!(selected.true_count() <= lhs.len());
        assert!(selected.true_count() <= bools.len());
        for i in 0..selected.true_count() {
            unsafe { *bools.get_unchecked_mut(i) = Op::compare(lhs.get_unchecked(i), &self.rhs) };
        }

        Poll::Ready(Ok(()))
    }
}
