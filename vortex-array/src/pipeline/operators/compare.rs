// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;
use std::task::Poll;

use itertools::Itertools;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::compute;
use crate::pipeline::bits::BitView;
use crate::pipeline::operators::constant::ConstantOperator;
use crate::pipeline::operators::scalar_compare::ScalarCompareOperator;
use crate::pipeline::operators::{BindContext, Operator};
use crate::pipeline::types::{Element, VType};
use crate::pipeline::vector::VectorId;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext};

#[macro_export]
macro_rules! match_each_compare_op {
    ($self:expr, | $enc:ident | $body:block) => {{
        match $self {
            $crate::compute::Operator::Eq => {
                type $enc = $crate::pipeline::operators::compare::Eq;
                $body
            }
            $crate::compute::Operator::NotEq => {
                type $enc = $crate::pipeline::operators::compare::NotEq;
                $body
            }
            $crate::compute::Operator::Gt => {
                type $enc = $crate::pipeline::operators::compare::Gt;
                $body
            }
            $crate::compute::Operator::Gte => {
                type $enc = $crate::pipeline::operators::compare::Gte;
                $body
            }
            $crate::compute::Operator::Lt => {
                type $enc = crate::pipeline::operators::compare::Lt;
                $body
            }
            $crate::compute::Operator::Lte => {
                type $enc = crate::pipeline::operators::compare::Lte;
                $body
            }
        }
    }};
}

#[derive(Debug, Hash)]
pub struct CompareOperator {
    children: [Arc<dyn Operator>; 2],
    op: compute::Operator,
}

impl CompareOperator {
    pub fn new(lhs: Arc<dyn Operator>, rhs: Arc<dyn Operator>, op: compute::Operator) -> Self {
        assert_eq!(lhs.vtype(), rhs.vtype(), "Operands must have the same type");
        Self {
            children: [lhs, rhs],
            op,
        }
    }
}

impl Operator for CompareOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn vtype(&self) -> VType {
        VType::Bool
    }

    fn children(&self) -> &[Arc<dyn Operator>] {
        &self.children
    }

    fn with_children(&self, children: Vec<Arc<dyn Operator>>) -> Arc<dyn Operator> {
        let [lhs, rhs] = children
            .try_into()
            .ok()
            .vortex_expect("Expected 2 children");
        Arc::new(CompareOperator::new(lhs, rhs, self.op))
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

    fn reduce_children(&self, children: &[Arc<dyn Operator>]) -> Option<Arc<dyn Operator>> {
        let constants = children
            .iter()
            .enumerate()
            .filter_map(|(idx, c)| {
                c.as_any()
                    .downcast_ref::<ConstantOperator>()
                    .map(|c| (idx, c))
            })
            .collect_vec();

        if constants.len() != 1 {
            return None;
        }
        let [(idx, lhs)] = constants
            .try_into()
            .ok()
            .vortex_expect("Expected 1 constant");

        if idx == 0 {
            Some(Arc::new(ScalarCompareOperator::new(
                children[1].clone(),
                self.op.inverse(),
                lhs.scalar.clone(),
            )))
        } else {
            Some(Arc::new(ScalarCompareOperator::new(
                children[0].clone(),
                self.op,
                lhs.scalar.clone(),
            )))
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
        ctx: &dyn KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let lhs_vec = ctx.vector(self.lhs);
        let lhs = lhs_vec.as_slice::<T>();
        let rhs_vec = ctx.vector(self.rhs);
        let rhs = rhs_vec.as_slice::<T>();
        let bools = out.as_slice_mut::<bool>();

        assert_eq!(
            lhs.len(),
            rhs.len(),
            "LHS and RHS must have the same length"
        );

        for i in 0..selected.true_count() {
            bools[i] = unsafe { Op::compare(lhs.get_unchecked(i), rhs.get_unchecked(i)) };
        }

        Poll::Ready(Ok(()))
    }
}

pub(crate) trait CompareOp<T> {
    fn compare(lhs: &T, rhs: &T) -> bool;
}

pub struct Eq;
impl<T: PartialEq> CompareOp<T> for Eq {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs == rhs
    }
}

pub struct NotEq;
impl<T: PartialEq> CompareOp<T> for NotEq {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs != rhs
    }
}

pub struct Gt;
impl<T: PartialOrd> CompareOp<T> for Gt {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs > rhs
    }
}

pub struct Gte;
impl<T: PartialOrd> CompareOp<T> for Gte {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs >= rhs
    }
}

pub struct Lt;
impl<T: PartialOrd> CompareOp<T> for Lt {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs < rhs
    }
}

pub struct Lte;
impl<T: PartialOrd> CompareOp<T> for Lte {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs <= rhs
    }
}
