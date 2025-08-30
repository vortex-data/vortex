// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::sync::Arc;

use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};

use crate::pipeline::bits::BitView;
use crate::pipeline::operators::{BindContext, Operator, OperatorRef};
use crate::pipeline::types::VType;
use crate::pipeline::vec::VectorId;
use crate::pipeline::view::ViewMut;
use crate::pipeline::{Kernel, KernelContext};

/// Boolean operations supported by the binary boolean operator.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
}

/// Pipeline operator for binary boolean operations (AND, OR) on two boolean arrays.
#[derive(Debug, Hash)]
pub struct BinaryBoolOpOperator {
    children: [OperatorRef; 2],
    op: BoolOp,
}

impl BinaryBoolOpOperator {
    fn new(left: OperatorRef, right: OperatorRef, op: BoolOp) -> Self {
        // Verify both children are boolean type
        let VType::Bool = left.vtype() else {
            vortex_panic!("BinaryBoolOpOperator left child must be boolean type");
        };
        let VType::Bool = right.vtype() else {
            vortex_panic!("BinaryBoolOpOperator right child must be boolean type");
        };

        Self {
            children: [left, right],
            op,
        }
    }

    pub fn and(left: OperatorRef, right: OperatorRef) -> Self {
        Self::new(left, right, BoolOp::And)
    }

    pub fn or(left: OperatorRef, right: OperatorRef) -> Self {
        Self::new(left, right, BoolOp::Or)
    }
}

impl Operator for BinaryBoolOpOperator {
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
        let [left, right] = children
            .try_into()
            .ok()
            .vortex_expect("Expected 2 children");
        Arc::new(BinaryBoolOpOperator::new(left, right, self.op))
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        if self.vtype() != VType::Bool {
            vortex_bail!("BinaryBoolOpOperator only supports boolean types");
        }

        Ok(Box::new(BinaryBoolOpKernel {
            left: ctx.children()[0],
            right: ctx.children()[1],
            op: self.op,
        }))
    }
}

/// Kernel that performs binary boolean operations on two input vectors.
pub struct BinaryBoolOpKernel {
    left: VectorId,
    right: VectorId,
    op: BoolOp,
}

impl Kernel for BinaryBoolOpKernel {
    fn seek(&mut self, _chunk_idx: usize) -> VortexResult<()> {
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        let left_vec = ctx.vector(self.left);
        let right_vec = ctx.vector(self.right);
        let left_values = left_vec.as_slice::<bool>();
        let right_values = right_vec.as_slice::<bool>();
        let out_slice = out.as_slice_mut::<bool>();

        assert!(selected.true_count() <= left_values.len());
        assert!(selected.true_count() <= right_values.len());
        assert!(selected.true_count() <= out_slice.len());

        match self.op {
            BoolOp::And => {
                for i in 0..selected.true_count() {
                    unsafe {
                        let left_val = *left_values.get_unchecked(i);
                        let right_val = *right_values.get_unchecked(i);
                        *out_slice.get_unchecked_mut(i) = left_val && right_val;
                    }
                }
            }
            BoolOp::Or => {
                for i in 0..selected.true_count() {
                    unsafe {
                        let left_val = *left_values.get_unchecked(i);
                        let right_val = *right_values.get_unchecked(i);
                        *out_slice.get_unchecked_mut(i) = left_val || right_val;
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::BufferMut;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use super::*;
    use crate::arrays::{ConstantArray, PrimitiveArray};
    use crate::compute::Operator as BinaryOperator;
    use crate::pipeline::bits::BitView;
    use crate::pipeline::operators::scalar_compare::ScalarCompareOperator;
    use crate::pipeline::query::QueryPlan;
    use crate::pipeline::view::ViewMut;
    use crate::pipeline::{N, N_WORDS};

    #[test]
    fn test_binary_bool_and_basic() {
        // Create left data: [1, 0, 1, 0] to generate [true, false, true, false]
        let size = 4;
        let left_primitive_array = [1i32, 0, 1, 0].into_iter().collect::<PrimitiveArray>();
        let left_primitive_op = left_primitive_array
            .as_ref()
            .to_operator()
            .unwrap()
            .unwrap();

        // Create right data: [1, 1, 0, 0] to generate [true, true, false, false]
        let right_primitive_array = [1i32, 1, 0, 0].into_iter().collect::<PrimitiveArray>();
        let right_primitive_op = right_primitive_array.to_operator().unwrap().unwrap();

        let zero_scalar = Scalar::primitive(0i32, Nullability::NonNullable);
        let left_bool_op = Arc::new(ScalarCompareOperator::new(
            left_primitive_op,
            BinaryOperator::Gt,
            zero_scalar.clone(),
        ));
        let right_bool_op = Arc::new(ScalarCompareOperator::new(
            right_primitive_op,
            BinaryOperator::Gt,
            zero_scalar,
        ));

        // Create binary AND operator: left_bool AND right_bool
        let binary_and_op = Arc::new(BinaryBoolOpOperator::and(left_bool_op, right_bool_op));

        // Create query plan from the operator
        let plan = QueryPlan::new(binary_and_op.as_ref()).unwrap();
        let mut pipeline = plan.executable_plan().unwrap();

        // Create mask for first 4 elements
        let mut mask_data = [0usize; N_WORDS];
        mask_data[0] = 0b1111; // First 4 bits set
        let mask_view = BitView::new(&mask_data);

        // Create output buffer for results
        let mut output = BufferMut::<bool>::with_capacity(N);
        unsafe { output.set_len(N) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        // Execute the pipeline
        let result = pipeline._step(mask_view, &mut output_view);
        assert!(result.is_ok());

        // Verify results: left[i] AND right[i]
        // [true&true, false&true, true&false, false&false] = [true, false, false, false]
        let expected = [true, false, false, false];
        for i in 0..size {
            assert_eq!(
                output[i], expected[i],
                "Position {}: expected {}, got {}",
                i, expected[i], output[i]
            );
        }
    }

    #[test]
    fn test_binary_bool_or_basic() {
        // Create left data: [1, 0, 1, 0] to generate [true, false, true, false]
        let size = 4;
        let left_primitive_array = [1i32, 0, 1, 0].into_iter().collect::<PrimitiveArray>();
        let left_primitive_op = left_primitive_array.to_operator().unwrap().unwrap();

        // Create right data: [1, 1, 0, 0] to generate [true, true, false, false]
        let right_primitive_array = [1i32, 1, 0, 0].into_iter().collect::<PrimitiveArray>();
        let right_primitive_op = right_primitive_array.to_operator().unwrap().unwrap();

        // Create comparisons to generate boolean values: value > 0
        let zero_scalar = Scalar::primitive(0i32, Nullability::NonNullable);
        let left_bool_op = Arc::new(ScalarCompareOperator::new(
            left_primitive_op,
            BinaryOperator::Gt,
            zero_scalar.clone(),
        ));
        let right_bool_op = Arc::new(ScalarCompareOperator::new(
            right_primitive_op,
            BinaryOperator::Gt,
            zero_scalar,
        ));

        // Create binary OR operator: left_bool OR right_bool
        let binary_or_op = Arc::new(BinaryBoolOpOperator::or(left_bool_op, right_bool_op));

        // Create query plan from the operator
        let plan = QueryPlan::new(binary_or_op.as_ref()).unwrap();
        let mut pipeline = plan.executable_plan().unwrap();

        // Create mask for first 4 elements
        let mut mask_data = [0usize; N_WORDS];
        mask_data[0] = 0b1111; // First 4 bits set
        let mask_view = BitView::new(&mask_data);

        // Create output buffer for results
        let mut output = BufferMut::<bool>::with_capacity(N);
        unsafe { output.set_len(N) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        // Execute the pipeline
        let result = pipeline._step(mask_view, &mut output_view);
        assert!(result.is_ok());

        // Verify results: left[i] OR right[i]
        // [true|true, false|true, true|false, false|false] = [true, true, true, false]
        let expected = [true, true, true, false];
        for i in 0..size {
            assert_eq!(
                output[i], expected[i],
                "Position {}: expected {}, got {}",
                i, expected[i], output[i]
            );
        }
    }

    #[test]
    fn test_binary_bool_with_constant() {
        // Test combining boolean operators with constants: (x AND y) OR true
        let size = 4;

        // Create x data: [1, 0, 1, 0] to generate [true, false, true, false]
        let x_primitive_array = [1i32, 0, 1, 0].into_iter().collect::<PrimitiveArray>();
        let x_primitive_op = x_primitive_array.to_operator().unwrap().unwrap();

        // Create y data: [0, 0, 1, 1] to generate [false, false, true, true]
        let y_primitive_array = [0i32, 0, 1, 1].into_iter().collect::<PrimitiveArray>();
        let y_primitive_op = y_primitive_array.to_operator().unwrap().unwrap();

        // Create comparisons to generate boolean values: value > 0
        use vortex_dtype::Nullability;
        use vortex_scalar::Scalar;

        use crate::compute::Operator as BinaryOperator;
        use crate::pipeline::operators::scalar_compare::ScalarCompareOperator;

        let zero_scalar = Scalar::primitive(0i32, Nullability::NonNullable);
        let x_bool_op = Arc::new(ScalarCompareOperator::new(
            x_primitive_op,
            BinaryOperator::Gt,
            zero_scalar.clone(),
        ));
        let y_bool_op = Arc::new(ScalarCompareOperator::new(
            y_primitive_op,
            BinaryOperator::Gt,
            zero_scalar,
        ));

        // Create constant true
        let constant_true = ConstantArray::new(Scalar::from(true), size)
            .to_operator()
            .unwrap()
            .unwrap();

        // Build pipeline: (x AND y) OR true
        let x_and_y = Arc::new(BinaryBoolOpOperator::and(x_bool_op, y_bool_op));
        let final_op = Arc::new(BinaryBoolOpOperator::or(x_and_y, constant_true));

        // Create query plan
        let plan = QueryPlan::new(final_op.as_ref()).unwrap();
        let mut pipeline = plan.executable_plan().unwrap();

        // Create mask for first 4 elements
        let mut mask_data = [0usize; N_WORDS];
        mask_data[0] = 0b1111; // First 4 bits set
        let mask_view = BitView::new(&mask_data);

        // Create output buffer
        let mut output = BufferMut::<bool>::with_capacity(N);
        unsafe { output.set_len(N) };
        let mut output_view = ViewMut::new(&mut output[..], None);

        // Execute the pipeline
        let result = pipeline._step(mask_view, &mut output_view);
        assert!(result.is_ok());

        // Verify results: (x[i] AND y[i]) OR true
        // [(true&false)|true, (false&false)|true, (true&true)|true, (false&true)|true] = [true, true, true, true]
        let expected = [true, true, true, true];
        for i in 0..size {
            assert_eq!(
                output[i], expected[i],
                "Position {}: expected {}, got {}",
                i, expected[i], output[i]
            );
        }
    }
}
