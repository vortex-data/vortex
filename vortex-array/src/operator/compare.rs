// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::{DType, NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::ConstantArray;
use crate::compute::Operator as Op;
use crate::operator::{LengthBounds, Operator, OperatorEq, OperatorHash, OperatorId, OperatorRef};
use crate::pipeline::bits::BitView;
use crate::pipeline::vec::Selection;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{
    BindContext, Element, Kernel, KernelContext, PipelinedOperator, RowSelection, VectorId,
};

#[derive(Debug)]
pub struct CompareOperator {
    children: [OperatorRef; 2],
    op: Op,
    dtype: DType,
}

impl CompareOperator {
    pub fn try_new(lhs: OperatorRef, rhs: OperatorRef, op: Op) -> VortexResult<CompareOperator> {
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

        Ok(CompareOperator {
            children: [lhs, rhs],
            op,
            dtype,
        })
    }

    pub fn op(&self) -> Op {
        self.op
    }
}

impl OperatorHash for CompareOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.op.hash(state);
        self.dtype.hash(state);
        self.children.iter().for_each(|c| c.operator_hash(state));
    }
}

impl OperatorEq for CompareOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.op == other.op
            && self.dtype == other.dtype
            && self
                .children
                .iter()
                .zip(other.children.iter())
                .all(|(a, b)| a.operator_eq(b))
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

    fn bounds(&self) -> LengthBounds {
        self.children[0].bounds() & self.children[1].bounds()
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
        // If both children support pipelining, but have different row selections, then we cannot
        // pipeline without an alignment step (which we currently do not support).
        if let Some((left, right)) = self.children[0]
            .as_pipelined()
            .zip(self.children[1].as_pipelined())
            && left.row_selection() != right.row_selection()
        {
            return None;
        }

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
    fn row_selection(&self) -> RowSelection {
        self.children[0]
            .as_pipelined()
            .map(|p| p.row_selection())
            .unwrap_or(RowSelection::All)
    }

    #[allow(clippy::cognitive_complexity)]
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        debug_assert_eq!(self.children[0].dtype(), self.children[1].dtype());

        let DType::Primitive(ptype, _) = self.children[0].dtype() else {
            vortex_bail!(
                "Unsupported type for comparison: {}",
                self.children[0].dtype()
            )
        };

        let lhs_const = self.children[0].as_any().downcast_ref::<ConstantArray>();
        if let Some(lhs_const) = lhs_const {
            // LHS is constant, use ScalarComparePrimitiveKernel
            return match_each_native_ptype!(ptype, |T| {
                match_each_compare_op!(self.op.swap(), |Op| {
                    Ok(Box::new(ScalarComparePrimitiveKernel::<T, Op> {
                        lhs: ctx.children()[1],
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

        let rhs_const = self.children[1].as_any().downcast_ref::<ConstantArray>();
        if let Some(rhs_const) = rhs_const {
            // RHS is constant, use ScalarComparePrimitiveKernel
            return match_each_native_ptype!(ptype, |T| {
                match_each_compare_op!(self.op, |Op| {
                    Ok(Box::new(ScalarComparePrimitiveKernel::<T, Op> {
                        lhs: ctx.children()[0],
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
                Ok(Box::new(ComparePrimitiveKernel::<T, Op> {
                    lhs: ctx.children()[0],
                    rhs: ctx.children()[1],
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

/// A compare operator for primitive types that compares two vectors element-wise using a binary
/// operation.
/// Kernel that performs primitive type comparisons between two input vectors.
pub struct ComparePrimitiveKernel<T, Op> {
    lhs: VectorId,
    rhs: VectorId,
    _phantom: PhantomData<(T, Op)>,
}

impl<T: Element + NativePType, Op: CompareOp<T> + Send> Kernel for ComparePrimitiveKernel<T, Op> {
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
        let bools = out.as_array_mut::<bool>();

        match (lhs_vec.selection(), rhs_vec.selection()) {
            (Selection::Prefix, Selection::Prefix) => {
                for i in 0..selection.true_count() {
                    bools[i] = Op::compare(&lhs[i], &rhs[i]);
                }
                out.set_selection(Selection::Prefix)
            }
            (Selection::Mask, Selection::Mask) => {
                // TODO(ngates): check density to decide if we should iterate indices or do
                //  a full scan
                let mut pos = 0;
                selection.iter_ones(|idx| {
                    bools[pos] = Op::compare(&lhs[idx], &rhs[idx]);
                    pos += 1;
                });
                out.set_selection(Selection::Prefix)
            }
            (Selection::Mask, Selection::Prefix) => {
                let mut pos = 0;
                selection.iter_ones(|idx| {
                    bools[pos] = Op::compare(&lhs[idx], &rhs[pos]);
                    pos += 1;
                });
                out.set_selection(Selection::Prefix)
            }
            (Selection::Prefix, Selection::Mask) => {
                let mut pos = 0;
                selection.iter_ones(|idx| {
                    bools[pos] = Op::compare(&lhs[pos], &rhs[idx]);
                    pos += 1;
                });
                out.set_selection(Selection::Prefix)
            }
        }

        Ok(())
    }
}

struct ScalarComparePrimitiveKernel<T: Element + NativePType, Op: CompareOp<T>> {
    lhs: VectorId,
    rhs: T,
    _phantom: PhantomData<Op>,
}

impl<T: Element + NativePType, Op: CompareOp<T> + Send> Kernel
    for ScalarComparePrimitiveKernel<T, Op>
{
    fn step(
        &self,
        ctx: &KernelContext,
        _chunk_idx: usize,
        selection: &BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let lhs_vec = ctx.vector(self.lhs);
        let lhs = lhs_vec.as_array::<T>();
        let bools = out.as_array_mut::<bool>();

        match lhs_vec.selection() {
            Selection::Prefix => {
                for i in 0..selection.true_count() {
                    bools[i] = Op::compare(&lhs[i], &self.rhs);
                }
                out.set_selection(Selection::Prefix)
            }
            Selection::Mask => {
                // TODO(ngates): decide at what true count we should iter indices...
                selection.iter_ones(|idx| {
                    bools[idx] = Op::compare(&lhs[idx], &self.rhs);
                });
                out.set_selection(Selection::Mask)
            }
        }

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

// TODO(ngates): bring these back!
// #[cfg(test)]
// mod tests {
//     use std::rc::Rc;
//
//     use vortex_buffer::BufferMut;
//     use vortex_dtype::Nullability;
//     use vortex_scalar::Scalar;
//
//     use crate::arrays::PrimitiveArray;
//     use crate::operator::bits::BitView;
//
//     #[test]
//     fn test_scalar_compare_stacked_on_primitive() {
//         // Create input data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
//         let size = 16;
//         let primitive_array = (0..i32::try_from(size).unwrap()).collect::<PrimitiveArray>();
//         let primitive_op = primitive_array.as_ref().to_operator().unwrap().unwrap();
//
//         // Create scalar compare operator: primitive_value > 10
//         let compare_value = Scalar::primitive(10i32, Nullability::NonNullable);
//         let scalar_compare_op = Rc::new(ScalarCompareOperator::new(
//             primitive_op,
//             BinaryOperator::Gt,
//             compare_value,
//         ));
//
//         // Create query plan from the stacked operators
//         let plan = QueryPlan::new(scalar_compare_op.as_ref()).unwrap();
//         let mut operator = plan.executable_plan().unwrap();
//
//         // Create all-true mask for simplicity
//         let mask_data = [usize::MAX; N_WORDS];
//         let mask_view = BitView::new(&mask_data);
//
//         // Create output buffer for boolean results
//         let mut output = BufferMut::<bool>::with_capacity(N);
//         unsafe { output.set_len(N) };
//         let mut output_view = ViewMut::new(&mut output[..], None);
//
//         // Execute the operator
//         let result = operator._step(mask_view, &mut output_view);
//         assert!(result.is_ok());
//
//         // Verify results: values 0-10 should be false, values 11-15 should be true
//         for i in 0..size {
//             let expected = i > 10;
//             assert_eq!(
//                 output[i], expected,
//                 "Position {}: expected {}, got {}",
//                 i, expected, output[i]
//             );
//         }
//     }
//
//     #[test]
//     fn test_scalar_compare_different_operators() {
//         // Test with different comparison operators
//         let size = 8;
//         let primitive_array = (0..i32::try_from(size).unwrap()).collect::<PrimitiveArray>();
//
//         let primitive_op = primitive_array.as_ref().to_operator().unwrap().unwrap();
//
//         // Test Eq: values == 3
//         let compare_value = Scalar::primitive(3i32, Nullability::NonNullable);
//         let eq_op = Rc::new(ScalarCompareOperator::new(
//             primitive_op,
//             BinaryOperator::Eq,
//             compare_value,
//         ));
//
//         let plan = QueryPlan::new(eq_op.as_ref()).unwrap();
//         let mut operator = plan.executable_plan().unwrap();
//
//         let mask_data = [usize::MAX; N_WORDS];
//         let mask_view = BitView::new(&mask_data);
//
//         let mut output = BufferMut::<bool>::with_capacity(N);
//         unsafe { output.set_len(N) };
//         let mut output_view = ViewMut::new(&mut output[..], None);
//
//         let result = operator._step(mask_view, &mut output_view);
//         assert!(result.is_ok());
//
//         // Only position 3 should be true
//         for i in 0..size {
//             let expected = i == 3;
//             assert_eq!(
//                 output[i], expected,
//                 "Eq test - Position {}: expected {}, got {}",
//                 i, expected, output[i]
//             );
//         }
//     }
//
//     #[test]
//     fn test_scalar_compare_with_f32() {
//         // Test with floating-point values
//         let size = 8;
//         let values: Vec<f32> = (0..size).map(|i| i as f32 + 0.5).collect();
//         let primitive_array = values.into_iter().collect::<PrimitiveArray>();
//
//         let primitive_op = primitive_array.as_ref().to_operator().unwrap().unwrap();
//
//         // Test Lt: values < 3.5
//         let compare_value = Scalar::primitive(3.5f32, Nullability::NonNullable);
//         let lt_op = Rc::new(ScalarCompareOperator::new(
//             primitive_op,
//             BinaryOperator::Lt,
//             compare_value,
//         ));
//
//         let plan = QueryPlan::new(lt_op.as_ref()).unwrap();
//         let mut operator = plan.executable_plan().unwrap();
//
//         let mask_data = [usize::MAX; N_WORDS];
//         let mask_view = BitView::new(&mask_data);
//
//         let mut output = BufferMut::<bool>::with_capacity(N);
//         unsafe { output.set_len(N) };
//         let mut output_view = ViewMut::new(&mut output[..], None);
//
//         let result = operator._step(mask_view, &mut output_view);
//         assert!(result.is_ok());
//
//         // Values 0.5, 1.5, 2.5 should be < 3.5 (true), 3.5+ should be false
//         for i in 0..size {
//             let value = i as f32 + 0.5;
//             let expected = value < 3.5;
//             assert_eq!(
//                 output[i], expected,
//                 "Lt test - Position {}: value {} should be {}, got {}",
//                 i, value, expected, output[i]
//             );
//         }
//     }
// }
