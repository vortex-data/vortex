// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ConstantArray;
use crate::compute::Operator as Op;
use crate::operator::{BindContext, Operator, OperatorId, OperatorRef, PipelinedOperator};
use crate::pipeline::bits::BitView;
use crate::pipeline::vec::VectorId;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Element, Kernel, KernelContext};
use itertools::Itertools;
use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

#[derive(Debug, Hash)]
pub struct CompareOperator {
    children: [OperatorRef; 2],
    op: Op,
    dtype: DType,
}

impl CompareOperator {
    pub fn try_new(
        lhs: OperatorRef,
        rhs: OperatorRef,
        op: Op,
    ) -> VortexResult<Arc<CompareOperator>> {
        if lhs.dtype() != rhs.dtype() {
            vortex_bail!(
                "Cannot compare arrays with different dtypes: {} and {}",
                lhs.dtype(),
                rhs.dtype()
            );
        }

        let lhs_const = lhs.as_any().downcast_ref::<ConstantArray>();
        let rhs_const = rhs.as_any().downcast_ref::<ConstantArray>();
        if lhs_const.is_some() && rhs_const.is_some() {
            // TODO(ngates): we should return the Constant result!
        }

        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let dtype = DType::Bool(nullability);

        Ok(Arc::new(CompareOperator {
            children: [lhs, rhs],
            op,
            dtype,
        }))
    }
}

impl Operator for CompareOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.compare")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.children[0].len(), self.children[1].len());
        self.children[0].len()
    }

    fn children(&self) -> &[OperatorRef] {
        &self.children
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        let (lhs, rhs) = children
            .into_iter()
            .tuples()
            .next()
            .vortex_expect("missing");
        Ok(Arc::new(CompareOperator {
            children: [lhs, rhs],
            op: self.op,
            dtype: self.dtype.clone(),
        }))
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        Some(self)
    }
}

macro_rules! match_each_compare_op {
    ($self:expr, | $enc:ident | $body:block) => {{
        match $self {
            Op::Eq => {
                type $enc = Eq;
                $body
            }
            Op::NotEq => {
                type $enc = NotEq;
                $body
            }
            Op::Gt => {
                type $enc = Gt;
                $body
            }
            Op::Gte => {
                type $enc = Gte;
                $body
            }
            Op::Lt => {
                type $enc = Lt;
                $body
            }
            Op::Lte => {
                type $enc = Lte;
                $body
            }
        }
    }};
}

impl PipelinedOperator for CompareOperator {
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        debug_assert_eq!(self.children[0].dtype(), self.children[1].dtype());

        let DType::Primitive(ptype, _) = self.children[0].dtype() else {
            vortex_bail!(
                "Unsupported type for comparison: {}",
                self.children[0].dtype()
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
        _selected: BitView,
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
