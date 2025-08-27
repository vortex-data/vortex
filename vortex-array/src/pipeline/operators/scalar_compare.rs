// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::compute::Operator as BinaryOperator;
use crate::match_each_compare_op;
use crate::pipeline::bits::BitView;
use crate::pipeline::operators::compare::CompareOp;
use crate::pipeline::operators::{BindContext, OperatorRef};
use crate::pipeline::types::{Element, VType};
use crate::pipeline::vec::VectorId;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext, Operator};

/// Pipeline operator for comparing an array against a scalar value.
#[derive(Debug, Hash)]
pub struct ScalarCompareOperator {
    children: [OperatorRef; 1],
    pub op: BinaryOperator,
    pub scalar: Scalar,
}

impl ScalarCompareOperator {
    pub fn new(child: OperatorRef, op: BinaryOperator, scalar: Scalar) -> Self {
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

    fn children(&self) -> &[OperatorRef] {
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

    fn with_children(&self, mut children: Vec<OperatorRef>) -> OperatorRef {
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
        ctx: &KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let lhs_vec = ctx.vector(self.lhs);
        let lhs = lhs_vec.as_slice::<T>();

        let bools = out.as_slice_mut::<bool>();

        debug_assert_eq!(selected.true_count(), lhs.len());
        lhs.iter().zip(bools).for_each(|(lhs, bool)| {
            *bool = Op::compare(lhs, &self.rhs);
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use vortex_buffer::BufferMut;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use super::*;
    use crate::arrays::PrimitiveArray;
    use crate::pipeline::bits::BitView;
    use crate::pipeline::query::QueryPlan;
    use crate::pipeline::view::ViewMut;
    use crate::pipeline::{N, N_WORDS};

    #[test]
    fn test_scalar_compare_stacked_on_primitive() {
        // Create input data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        let size = 16;
        let primitive_array = (0..i32::try_from(size).unwrap()).collect::<PrimitiveArray>();
        let primitive_op = primitive_array.as_ref().to_operator().unwrap().unwrap();

        // Create scalar compare operator: primitive_value > 10
        let compare_value = Scalar::primitive(10i32, Nullability::NonNullable);
        let scalar_compare_op = Rc::new(ScalarCompareOperator::new(
            primitive_op,
            BinaryOperator::Gt,
            compare_value,
        ));

        // Create query plan from the stacked operators
        let plan = QueryPlan::new(scalar_compare_op.as_ref()).unwrap();
        let mut pipeline = plan.executable_plan().unwrap();

        // Create all-true mask for simplicity
        let mask_data = [usize::MAX; N_WORDS];
        let mask_view = BitView::new(&mask_data);

        // Create output buffer for boolean results
        let mut output = BufferMut::<bool>::with_capacity(N);
        unsafe { output.set_len(N) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        // Execute the pipeline
        let result = pipeline._step(mask_view, &mut output_view);
        assert!(result.is_ok());

        // Verify results: values 0-10 should be false, values 11-15 should be true
        for i in 0..size {
            let expected = i > 10;
            assert_eq!(
                output[i], expected,
                "Position {}: expected {}, got {}",
                i, expected, output[i]
            );
        }
    }

    #[test]
    fn test_scalar_compare_different_operators() {
        // Test with different comparison operators
        let size = 8;
        let primitive_array = (0..i32::try_from(size).unwrap()).collect::<PrimitiveArray>();

        let primitive_op = primitive_array.as_ref().to_operator().unwrap().unwrap();

        // Test Eq: values == 3
        let compare_value = Scalar::primitive(3i32, Nullability::NonNullable);
        let eq_op = Rc::new(ScalarCompareOperator::new(
            primitive_op,
            BinaryOperator::Eq,
            compare_value,
        ));

        let plan = QueryPlan::new(eq_op.as_ref()).unwrap();
        let mut pipeline = plan.executable_plan().unwrap();

        let mask_data = [usize::MAX; N_WORDS];
        let mask_view = BitView::new(&mask_data);

        let mut output = BufferMut::<bool>::with_capacity(N);
        unsafe { output.set_len(N) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        let result = pipeline._step(mask_view, &mut output_view);
        assert!(result.is_ok());

        // Only position 3 should be true
        for i in 0..size {
            let expected = i == 3;
            assert_eq!(
                output[i], expected,
                "Eq test - Position {}: expected {}, got {}",
                i, expected, output[i]
            );
        }
    }

    #[test]
    fn test_scalar_compare_with_f32() {
        // Test with floating-point values
        let size = 8;
        let values: Vec<f32> = (0..size).map(|i| i as f32 + 0.5).collect();
        let primitive_array = values.into_iter().collect::<PrimitiveArray>();

        let primitive_op = primitive_array.as_ref().to_operator().unwrap().unwrap();

        // Test Lt: values < 3.5
        let compare_value = Scalar::primitive(3.5f32, Nullability::NonNullable);
        let lt_op = Rc::new(ScalarCompareOperator::new(
            primitive_op,
            BinaryOperator::Lt,
            compare_value,
        ));

        let plan = QueryPlan::new(lt_op.as_ref()).unwrap();
        let mut pipeline = plan.executable_plan().unwrap();

        let mask_data = [usize::MAX; N_WORDS];
        let mask_view = BitView::new(&mask_data);

        let mut output = BufferMut::<bool>::with_capacity(N);
        unsafe { output.set_len(N) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        let result = pipeline._step(mask_view, &mut output_view);
        assert!(result.is_ok());

        // Values 0.5, 1.5, 2.5 should be < 3.5 (true), 3.5+ should be false
        for i in 0..size {
            let value = i as f32 + 0.5;
            let expected = value < 3.5;
            assert_eq!(
                output[i], expected,
                "Lt test - Position {}: value {} should be {}, got {}",
                i, value, expected, output[i]
            );
        }
    }
}
