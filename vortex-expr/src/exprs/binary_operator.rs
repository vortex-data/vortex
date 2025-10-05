// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Operator;
use itertools::Itertools;
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;
use vortex_array::arrays::ConstantArray;
use vortex_array::operator::{OperatorEq, OperatorHash, OperatorId, OperatorRef};
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::vec::Selection;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{
    BindContext, Element, Kernel, KernelContext, PipelinedOperator, VectorHandle, N,
};
use vortex_dtype::{match_each_native_ptype, DType};
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};

#[derive(Debug)]
pub struct BinaryOperator {
    op: Operator,
    children: [OperatorRef; 2],
    dtype: DType,
}

impl BinaryOperator {
    pub fn try_new(lhs: OperatorRef, rhs: OperatorRef, op: Operator) -> VortexResult<Self> {
        if lhs.len() != rhs.len() {
            vortex_bail!(
                "Mismatched lengths for binary operator: lhs {}, rhs {}",
                lhs.len(),
                rhs.len()
            );
        }
        if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
            vortex_bail!(
                "Mismatched dtypes for binary operator: lhs {:?}, rhs {:?}",
                lhs.dtype(),
                rhs.dtype()
            );
        }

        let dtype = match op {
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => lhs.dtype().clone(),
            Operator::Eq
            | Operator::NotEq
            | Operator::Gt
            | Operator::Gte
            | Operator::Lt
            | Operator::Lte
            | Operator::And
            | Operator::Or => DType::Bool(lhs.dtype().nullability() | rhs.dtype().nullability()),
        };

        Ok(Self {
            op,
            children: [lhs, rhs],
            dtype,
        })
    }

    #[inline(always)]
    pub fn lhs(&self) -> &OperatorRef {
        &self.children[0]
    }

    #[inline(always)]
    pub fn rhs(&self) -> &OperatorRef {
        &self.children[1]
    }
}

impl OperatorHash for BinaryOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.op.hash(state);
        for child in &self.children {
            child.operator_hash(state);
        }
        self.dtype.hash(state);
    }
}

impl OperatorEq for BinaryOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.op == other.op
            && self.children.len() == other.children.len()
            && self
                .children
                .iter()
                .zip(other.children.iter())
                .all(|(a, b)| a.operator_eq(b))
            && self.dtype == other.dtype
    }
}

impl vortex_array::operator::Operator for BinaryOperator {
    fn id(&self) -> OperatorId {
        // TODO(ngates): distinguish different binary operators
        OperatorId::from("vortex.binary")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.lhs().len()
    }

    fn children(&self) -> &[OperatorRef] {
        &self.children
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        let (lhs, rhs) = children
            .into_iter()
            .collect_tuple()
            .ok_or_else(|| vortex_err!("expected two children for binary operator"))?;
        Ok(Arc::new(Self::try_new(lhs, rhs, self.op.clone())?))
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        Some(self)
    }
}

macro_rules! match_each_compare_op {
    ($self:expr, | $enc:ident | $body:block) => {{
        match $self {
            Operator::Eq => {
                type $enc = Eq;
                $body
            }
            Operator::NotEq => {
                type $enc = NotEq;
                $body
            }
            Operator::Gt => {
                type $enc = Gt;
                $body
            }
            Operator::Gte => {
                type $enc = Gte;
                $body
            }
            Operator::Lt => {
                type $enc = Lt;
                $body
            }
            Operator::Lte => {
                type $enc = Lte;
                $body
            }
            _ => vortex_bail!("Unsupported binary operator {}", $self),
        }
    }};
}

impl PipelinedOperator for BinaryOperator {
    #[allow(clippy::cognitive_complexity)]
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        debug_assert!(self.lhs().dtype().eq_ignore_nullability(self.rhs().dtype()));

        // NOTE(ngates): we really need to stop lumping all operations into one expression...
        if matches!(self.lhs().dtype(), DType::Bool(_)) {
            let lhs_const = self.lhs().as_any().downcast_ref::<ConstantArray>();
            if let Some(lhs_const) = lhs_const {
                if let Some(swapped_op) = self.op.swap() {
                    // LHS is constant, use ScalarComparePrimitiveKernel
                    match swapped_op {
                        Operator::And => {
                            return Ok(Box::new(ScalarBinaryPrimitiveKernel::<bool, And> {
                                lhs: ctx.vector_input(1),
                                rhs: lhs_const
                                    .scalar()
                                    .as_bool()
                                    .value()
                                    .vortex_expect("scalar value is null"),
                                _phantom: PhantomData,
                            }) as Box<dyn Kernel>);
                        }
                        _ => vortex_bail!("Unsupported binary operator {}", self.op),
                    }
                }
            }

            let rhs_const = self.rhs().as_any().downcast_ref::<ConstantArray>();
            if let Some(rhs_const) = rhs_const {
                // RHS is constant, use ScalarComparePrimitiveKernel
                match self.op {
                    Operator::And => {
                        return Ok(Box::new(ScalarBinaryPrimitiveKernel::<bool, And> {
                            lhs: ctx.vector_input(0),
                            rhs: rhs_const
                                .scalar()
                                .as_bool()
                                .value()
                                .vortex_expect("scalar value is null"),
                            _phantom: PhantomData,
                        }) as Box<dyn Kernel>);
                    }
                    _ => vortex_bail!("Unsupported binary operator {}", self.op),
                }
            }

            match self.op {
                Operator::And => {
                    return Ok(Box::new(BinaryPrimitiveKernel::<bool, And> {
                        lhs: ctx.vector_input(0),
                        rhs: ctx.vector_input(1),
                        _phantom: PhantomData,
                    }) as Box<dyn Kernel>);
                }
                _ => vortex_bail!("Unsupported binary operator {}", self.op),
            }
        }

        let DType::Primitive(ptype, _) = self.lhs().dtype() else {
            vortex_bail!("Unsupported binary operator {}", self.lhs().dtype());
        };
        let lhs_const = self.lhs().as_any().downcast_ref::<ConstantArray>();
        if let Some(lhs_const) = lhs_const {
            if let Some(swapped_op) = self.op.swap() {
                // LHS is constant, use ScalarComparePrimitiveKernel
                return match_each_native_ptype!(ptype, |T| {
                    match_each_compare_op!(swapped_op, |Op| {
                        Ok(Box::new(ScalarBinaryPrimitiveKernel::<T, Op> {
                            lhs: ctx.vector_input(1),
                            rhs: lhs_const
                                .scalar()
                                .as_primitive()
                                .typed_value::<T>()
                                .vortex_expect("scalar value not of type T"),
                            _phantom: PhantomData,
                        }) as Box<dyn Kernel>)
                    })
                });
            }
        }

        let rhs_const = self.rhs().as_any().downcast_ref::<ConstantArray>();
        if let Some(rhs_const) = rhs_const {
            // RHS is constant, use ScalarComparePrimitiveKernel
            return match_each_native_ptype!(ptype, |T| {
                match_each_compare_op!(self.op, |Op| {
                    Ok(Box::new(ScalarBinaryPrimitiveKernel::<T, Op> {
                        lhs: ctx.vector_input(0),
                        rhs: rhs_const
                            .scalar()
                            .as_primitive()
                            .typed_value::<T>()
                            .vortex_expect("scalar value not of type T"),
                        _phantom: PhantomData,
                    }) as Box<dyn Kernel>)
                })
            });
        }

        match_each_native_ptype!(ptype, |T| {
            match_each_compare_op!(self.op, |Op| {
                Ok(Box::new(BinaryPrimitiveKernel::<T, Op> {
                    lhs: ctx.vector_input(0),
                    rhs: ctx.vector_input(1),
                    _phantom: PhantomData,
                }) as Box<dyn Kernel>)
            })
        })
    }

    fn vector_children(&self) -> Vec<usize> {
        vec![0, 1]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![]
    }
}

/// A kernel that evaluates two vectors element-wise with a binary operator.
pub struct BinaryPrimitiveKernel<T, Op> {
    lhs: VectorHandle,
    rhs: VectorHandle,
    _phantom: PhantomData<(T, Op)>,
}

impl<T: Element, Op: BinaryOp<T> + Send> Kernel for BinaryPrimitiveKernel<T, Op> {
    fn step(
        &self,
        ctx: &KernelContext,
        _chunk_idx: usize,
        selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let lhs_vec = ctx.vector(self.lhs);
        let lhs = lhs_vec.as_array::<T>();
        let rhs_vec = ctx.vector(self.rhs);
        let rhs = rhs_vec.as_array::<T>();
        let results = out.as_array_mut::<Op::Output>();

        match (lhs_vec.selection(), rhs_vec.selection()) {
            (Selection::Prefix, Selection::Prefix) => {
                for i in 0..selection.true_count() {
                    results[i] = Op::evaluate(&lhs[i], &rhs[i]);
                }
                out.set_selection(Selection::Prefix)
            }
            (Selection::Mask, Selection::Mask) => {
                // TODO(ngates): check density to decide if we should iterate indices or do
                //  a full scan
                let mut pos = 0;
                selection.iter_ones(|idx| {
                    results[pos] = Op::evaluate(&lhs[idx], &rhs[idx]);
                    pos += 1;
                });
                out.set_selection(Selection::Prefix)
            }
            (Selection::Mask, Selection::Prefix) => {
                let mut pos = 0;
                selection.iter_ones(|idx| {
                    results[pos] = Op::evaluate(&lhs[idx], &rhs[pos]);
                    pos += 1;
                });
                out.set_selection(Selection::Prefix)
            }
            (Selection::Prefix, Selection::Mask) => {
                let mut pos = 0;
                selection.iter_ones(|idx| {
                    results[pos] = Op::evaluate(&lhs[pos], &rhs[idx]);
                    pos += 1;
                });
                out.set_selection(Selection::Prefix)
            }
        }

        Ok(())
    }
}

/// A kernel that evaluates a vector element-wise with a scalar and binary operator.
struct ScalarBinaryPrimitiveKernel<T: Element, Op: BinaryOp<T>> {
    lhs: VectorHandle,
    rhs: T,
    _phantom: PhantomData<Op>,
}

impl<T: Element, Op: BinaryOp<T> + Send> Kernel for ScalarBinaryPrimitiveKernel<T, Op> {
    fn step(
        &self,
        ctx: &KernelContext,
        _chunk_idx: usize,
        selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let lhs_vec = ctx.vector(self.lhs);
        let lhs = lhs_vec.as_array::<T>();
        let results = out.as_array_mut::<Op::Output>();

        match lhs_vec.selection() {
            Selection::Prefix => {
                for i in 0..selection.true_count() {
                    results[i] = Op::evaluate(&lhs[i], &self.rhs);
                }
                out.set_selection(Selection::Prefix)
            }
            Selection::Mask => {
                match selection.true_count() {
                    // This threshold probably depends on SIMD register width
                    // For now, we assume a conservative 16 bytes.
                    // We should just setup the benchmarks to find the threshold experimentally.
                    // In fact, Vortex should come with a command to run on a machine to find
                    // these numbers and persist them somewhere.
                    n if n > (N / (16 / size_of::<T>())) => {
                        for idx in 0..N {
                            results[idx] = Op::evaluate(&lhs[idx], &self.rhs);
                        }
                        out.set_selection(Selection::Mask)
                    }
                    _ => {
                        let mut pos = 0;
                        selection.iter_ones(|idx| {
                            results[pos] = Op::evaluate(&lhs[idx], &self.rhs);
                            pos += 1;
                        });
                        out.set_selection(Selection::Prefix)
                    }
                }
            }
        }

        Ok(())
    }
}

pub(crate) trait BinaryOp<T>: Send {
    type Output: Element;

    fn evaluate(lhs: &T, rhs: &T) -> Self::Output;
}

/// Equality comparison operation.
pub struct Eq;
impl<T: PartialEq> BinaryOp<T> for Eq {
    type Output = bool;

    #[inline(always)]
    fn evaluate(lhs: &T, rhs: &T) -> Self::Output {
        lhs == rhs
    }
}

/// Not equal comparison operation.
pub struct NotEq;
impl<T: PartialEq> BinaryOp<T> for NotEq {
    type Output = bool;
    #[inline(always)]
    fn evaluate(lhs: &T, rhs: &T) -> Self::Output {
        lhs != rhs
    }
}

/// Greater than comparison operation.
pub struct Gt;
impl<T: PartialOrd> BinaryOp<T> for Gt {
    type Output = bool;
    #[inline(always)]
    fn evaluate(lhs: &T, rhs: &T) -> Self::Output {
        lhs > rhs
    }
}

/// Greater than or equal comparison operation.
pub struct Gte;
impl<T: PartialOrd> BinaryOp<T> for Gte {
    type Output = bool;
    #[inline(always)]
    fn evaluate(lhs: &T, rhs: &T) -> Self::Output {
        lhs >= rhs
    }
}

/// Less than comparison operation.
pub struct Lt;
impl<T: PartialOrd> BinaryOp<T> for Lt {
    type Output = bool;
    #[inline(always)]
    fn evaluate(lhs: &T, rhs: &T) -> Self::Output {
        lhs < rhs
    }
}

/// Less than or equal comparison operation.
pub struct Lte;
impl<T: PartialOrd> BinaryOp<T> for Lte {
    type Output = bool;

    #[inline(always)]
    fn evaluate(lhs: &T, rhs: &T) -> Self::Output {
        lhs <= rhs
    }
}

/// Logical AND operation.
pub struct And;
impl BinaryOp<bool> for And {
    type Output = bool;
    #[inline(always)]
    fn evaluate(lhs: &bool, rhs: &bool) -> Self::Output {
        *lhs && *rhs
    }
}

/// Logical OR operation.
pub struct Or;
impl BinaryOp<bool> for Or {
    type Output = bool;
    #[inline(always)]
    fn evaluate(lhs: &bool, rhs: &bool) -> Self::Output {
        *lhs && *rhs
    }
}
