// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::ConstantOperator;
use crate::compute::Operator as BinaryOperator;
use crate::pipeline::bits::BitView;
use crate::pipeline::operators::scalar_compare::ScalarCompareOperator;
use crate::pipeline::operators::{BindContext, Operator, OperatorRef};
use crate::pipeline::types::{Element, VType};
use crate::pipeline::vec::VectorId;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext};

#[macro_export]
macro_rules! match_each_compare_op {
    ($self:expr, | $enc:ident | $body:block) => {{
        match $self {
            BinaryOperator::Eq => {
                type $enc = $crate::pipeline::operators::compare::Eq;
                $body
            }
            BinaryOperator::NotEq => {
                type $enc = $crate::pipeline::operators::compare::NotEq;
                $body
            }
            BinaryOperator::Gt => {
                type $enc = $crate::pipeline::operators::compare::Gt;
                $body
            }
            BinaryOperator::Gte => {
                type $enc = $crate::pipeline::operators::compare::Gte;
                $body
            }
            BinaryOperator::Lt => {
                type $enc = $crate::pipeline::operators::compare::Lt;
                $body
            }
            BinaryOperator::Lte => {
                type $enc = $crate::pipeline::operators::compare::Lte;
                $body
            }
        }
    }};
}

/// Pipeline operator for comparing two arrays using various comparison operations.
#[derive(Debug, Hash)]
pub struct CompareOperator {
    children: [OperatorRef; 2],
    op: BinaryOperator,
}

impl CompareOperator {
    pub fn new(lhs: OperatorRef, rhs: OperatorRef, op: BinaryOperator) -> Self {
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

    fn children(&self) -> &[OperatorRef] {
        &self.children
    }

    fn with_children(&self, children: Vec<OperatorRef>) -> OperatorRef {
        let [lhs, rhs] = children
            .try_into()
            .ok()
            .vortex_expect("Expected 2 children");
        Arc::new(CompareOperator::new(lhs, rhs, self.op))
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        debug_assert_eq!(self.children[0].vtype(), self.children[1].vtype());

        let VType::Primitive(ptype) = self.children[0].vtype() else {
            vortex_bail!(
                "Unsupported type for comparison: {}",
                self.children[0].vtype()
            )
        };

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

    fn reduce_children(&self, children: &[OperatorRef]) -> Option<OperatorRef> {
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
/// Kernel that performs primitive type comparisons between two input vectors.
pub struct ComparePrimitiveKernel<T, Op> {
    lhs: VectorId,
    rhs: VectorId,
    _phantom: PhantomData<(T, Op)>,
}

impl<T: Element + NativePType, Op: CompareOp<T>> Kernel for ComparePrimitiveKernel<T, Op> {
    fn step(
        &mut self,
        ctx: &KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
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

        lhs.iter()
            .zip(rhs.iter())
            .zip(bools)
            .for_each(|((lhs, rhs), bool)| *bool = Op::compare(lhs, rhs));

        Ok(())
    }
}

pub(crate) trait CompareOp<T> {
    fn compare(lhs: &T, rhs: &T) -> bool;
}

/// Equality comparison operation.
pub struct Eq;
impl<T: PartialEq> CompareOp<T> for Eq {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs == rhs
    }
}

/// Not equal comparison operation.
pub struct NotEq;
impl<T: PartialEq> CompareOp<T> for NotEq {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs != rhs
    }
}

/// Greater than comparison operation.
pub struct Gt;
impl<T: PartialOrd> CompareOp<T> for Gt {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs > rhs
    }
}

/// Greater than or equal comparison operation.
pub struct Gte;
impl<T: PartialOrd> CompareOp<T> for Gte {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs >= rhs
    }
}

/// Less than comparison operation.
pub struct Lt;
impl<T: PartialOrd> CompareOp<T> for Lt {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs < rhs
    }
}

/// Less than or equal comparison operation.
pub struct Lte;
impl<T: PartialOrd> CompareOp<T> for Lte {
    #[inline(always)]
    fn compare(lhs: &T, rhs: &T) -> bool {
        lhs <= rhs
    }
}
