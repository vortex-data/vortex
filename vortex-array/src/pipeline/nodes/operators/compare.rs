// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::compute;
use crate::pipeline::bits::BitView;
use crate::pipeline::nodes::operators::{BindContext, Operator};
use crate::pipeline::types::{Element, VType};
use crate::pipeline::vector::VectorId;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, PipelineContext};
use std::marker::PhantomData;
use std::task::Poll;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};

#[macro_export]
macro_rules! match_each_compare_op {
    ($self:expr, | $enc:ident | $body:block) => {{
        match $self {
            $crate::compute::Operator::Eq => {
                type $enc = Eq;
                $body
            }
            $crate::compute::Operator::NotEq => {
                type $enc = NotEq;
                $body
            }
            $crate::compute::Operator::Gt => {
                type $enc = Gt;
                $body
            }
            $crate::compute::Operator::Gte => {
                type $enc = Gte;
                $body
            }
            $crate::compute::Operator::Lt => {
                type $enc = Lt;
                $body
            }
            $crate::compute::Operator::Lte => {
                type $enc = Lte;
                $body
            }
        }
    }};
}

#[derive(Debug, Hash)]
pub struct CompareOperator {
    children: [Box<dyn Operator>; 2],
    op: compute::Operator,
}

impl CompareOperator {
    pub fn new(lhs: Box<dyn Operator>, rhs: Box<dyn Operator>, op: compute::Operator) -> Self {
        assert_eq!(lhs.vtype(), rhs.vtype(), "Operands must have the same type");
        Self {
            children: [lhs, rhs],
            op,
        }
    }
}

impl Operator for CompareOperator {
    fn vtype(&self) -> VType {
        VType::Bool
    }

    fn children(&self) -> &[Box<dyn Operator>] {
        &self.children
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        assert_eq!(self.children[0].vtype(), self.children[1].vtype());

        match self.children[0].vtype() {
            VType::Primitive(ptype) => {
                match_each_native_ptype!(ptype, |T| {
                    match_each_compare_op!(self.op, |Op| {
                        Ok(Box::new(ComparePrimitiveKernel::<T, Op> {
                            lhs: ctx.children()[0],
                            rhs: ctx.children()[1],
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
}

/// A compare operator for primitive types that compares two vectors element-wise using a binary
/// operation.
pub struct ComparePrimitiveKernel<T, Op> {
    lhs: VectorId,
    rhs: VectorId,
    _phantom: PhantomData<(T, Op)>,
}

impl<T: Element + NativePType, Op: CompareOp<T>> Kernel for ComparePrimitiveKernel<T, Op> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        todo!()
    }

    fn step(
        &mut self,
        ctx: &dyn PipelineContext,
        _selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let lhs_vec = ctx.vector(self.lhs);
        let lhs = lhs_vec.elements::<T>();
        let rhs_vec = ctx.vector(self.rhs);
        let rhs = rhs_vec.elements::<T>();
        let bools = out.as_mut::<bool>();

        assert_eq!(
            lhs.len(),
            rhs.len(),
            "LHS and RHS must have the same length"
        );

        for i in 0..lhs_vec.len() {
            bools[i] = unsafe { Op::compare(lhs.get_unchecked(i), rhs.get_unchecked(i)) };
        }

        Poll::Ready(Ok(()))
    }
}

trait CompareOp<T> {
    fn compare(lhs: &T, rhs: &T) -> bool;
}

struct Eq;
impl<T: PartialEq> CompareOp<T> for Eq {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs == rhs
    }
}

struct NotEq;
impl<T: PartialEq> CompareOp<T> for NotEq {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs != rhs
    }
}

struct Gt;
impl<T: PartialOrd> CompareOp<T> for Gt {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs > rhs
    }
}

struct Gte;
impl<T: PartialOrd> CompareOp<T> for Gte {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs >= rhs
    }
}

struct Lt;
impl<T: PartialOrd> CompareOp<T> for Lt {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs < rhs
    }
}

struct Lte;
impl<T: PartialOrd> CompareOp<T> for Lte {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs <= rhs
    }
}
