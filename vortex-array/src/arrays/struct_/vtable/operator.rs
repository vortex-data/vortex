// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_vector::Vector;
use vortex_vector::struct_::StructVector;

use crate::ArrayRef;
use crate::array::transform::{ArrayParentReduceRule, ArrayRuleContext};
use crate::arrays::expr::{ExprArray, ExprVTable};
use crate::arrays::struct_::vtable::reduce::{apply_partitioned_expr, partition_struct_expr};
use crate::arrays::{StructArray, StructVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::expr::session::ExprSession;
use crate::vtable::{OperatorVTable, ValidityHelper};

impl OperatorVTable<StructVTable> for StructVTable {
    fn bind(
        array: &StructArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        // Bind all child field arrays with the selection.
        let field_kernels: Vec<_> = array
            .fields()
            .iter()
            .map(|field| ctx.bind(field, selection))
            .collect::<VortexResult<_>>()?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        Ok(kernel(move || {
            // Execute all child field kernels.
            let fields: Vec<Vector> = field_kernels
                .into_iter()
                .map(|k| k.execute())
                .collect::<VortexResult<_>>()?;
            let validity_mask = validity.execute()?;

            Ok(StructVector::try_new(Arc::new(fields.into_boxed_slice()), validity_mask)?.into())
        }))
    }
}

/// Rule to partition expressions over struct fields when a StructArray is wrapped by an ExprArray.
///
/// This optimization pushes expression evaluation down to individual struct fields, enabling
/// better field-level optimizations and potentially avoiding materialization of unused fields.
pub struct StructExprPartitionRule;

impl ArrayParentReduceRule<StructVTable, ExprVTable> for StructExprPartitionRule {
    fn reduce_parent(
        &self,
        array: &StructArray,
        parent: &ExprArray,
        _child_idx: usize,
        _ctx: &ArrayRuleContext,
    ) -> VortexResult<Option<ArrayRef>> {
        if array.dtype().is_nullable() {
            // TODO(joe): cannot handle nullable struct pushdown yet.
            return Ok(None);
        }

        let session = ExprSession::default();

        // Partition the expression over the struct fields
        let partitioned = partition_struct_expr(array, parent.expr().clone(), &session)?;

        // Apply the partitioned expression to create a new struct with ExprArrays
        let result = apply_partitioned_expr(array, partitioned)?;

        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {

    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{FieldNames, PTypeDowncast};
    use vortex_error::VortexExpect;
    use vortex_mask::Mask;
    use vortex_vector::VectorOps;

    use super::*;
    use crate::arrays::expr::ExprVTable;
    use crate::arrays::{BoolArray, ExprArray, PrimitiveArray, StructArray};
    use crate::expr::transform::ExprOptimizer;
    use crate::expr::{and, col, eq, get_item, gt, lit, lt, pack, root};
    use crate::validity::Validity;
    use crate::{Array, IntoArray, assert_arrays_eq};

    #[test]
    fn test_struct_operator_basic() {
        // Create a struct array with two fields: integers and booleans.
        let int_field = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let bool_field = BoolArray::from_iter([true, false, true, false, true]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["ints", "bools"]),
            vec![int_field.into_array(), bool_field.into_array()],
            5,
            Validity::AllValid,
        )
        .unwrap();

        // Execute without selection.
        let result = struct_array.execute().unwrap();
        assert_eq!(result.len(), 5);

        // Verify the struct vector fields.
        let struct_vector = result.as_struct();
        let fields = struct_vector.fields();
        assert_eq!(fields.len(), 2);

        // Verify the integer field values match the original.
        let int_vector = fields[0].as_primitive().clone().into_i32();
        assert_eq!(int_vector.elements().as_slice(), &[1, 2, 3, 4, 5]);

        // Verify the boolean field values match the original.
        let bool_vector = fields[1].as_bool();
        let bool_values: Vec<bool> = (0..5).map(|i| bool_vector.bits().value(i)).collect();
        assert_eq!(bool_values, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_struct_operator_with_mask() {
        // Create a struct array with two fields.
        let int_field = PrimitiveArray::from_iter([10i32, 20, 30, 40, 50, 60]);
        let bool_field = BoolArray::from_iter([true, false, true, false, true, false]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["numbers", "flags"]),
            vec![int_field.into_array(), bool_field.into_array()],
            6,
            Validity::AllValid,
        )
        .unwrap();

        // Create a selection mask that selects indices 0, 2, 4 (alternating pattern).
        let selection = Mask::from_iter([true, false, true, false, true, false]);

        // Execute with selection mask.
        let result = struct_array.execute_with_selection(&selection).unwrap();

        // Verify the result has the filtered length.
        assert_eq!(result.len(), 3);

        // Verify the struct vector fields.
        let struct_vector = result.as_struct();
        let fields = struct_vector.fields();
        assert_eq!(fields.len(), 2);

        // Verify the integer field has the correct filtered values (indices 0, 2, 4).
        let int_vector = fields[0].as_primitive().clone().into_i32();
        assert_eq!(int_vector.elements().as_slice(), &[10, 30, 50]);

        // Verify the boolean field has the correct filtered values (indices 0, 2, 4).
        let bool_vector = fields[1].as_bool();
        let bool_values: Vec<bool> = (0..3).map(|i| bool_vector.bits().value(i)).collect();
        assert_eq!(bool_values, vec![true, true, true]);
    }

    #[test]
    fn test_struct_operator_null_handling() {
        // Create fields with nulls.
        let int_field = PrimitiveArray::from_option_iter([
            Some(100i32),
            None,
            Some(200),
            Some(300),
            None,
            Some(400),
        ]);

        // Create bool field with its own validity.
        let bool_array = BoolArray::from_iter([true, false, true, false, true, false]);
        let bool_validity = Validity::from_iter([true, true, false, true, true, false]);
        let bool_field = BoolArray::from_bit_buffer(bool_array.bit_buffer().clone(), bool_validity);

        // Create struct with its own validity mask (rows 1 and 4 are null).
        let struct_validity = Validity::from_iter([true, false, true, true, false, true]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["values", "flags"]),
            vec![int_field.into_array(), bool_field.into_array()],
            6,
            struct_validity,
        )
        .unwrap();

        // Create a selection mask that selects indices 0, 1, 2, 4, 5.
        let selection = Mask::from_iter([true, true, true, false, true, true]);

        // Execute with selection mask.
        let result = struct_array.execute_with_selection(&selection).unwrap();

        assert_eq!(result.len(), 5);

        // Verify the struct vector fields.
        let struct_vector = result.as_struct();
        let fields = struct_vector.fields();
        assert_eq!(fields.len(), 2);

        // Verify integer field has the correct filtered values with nulls.
        // Selected indices: 0, 1, 2, 4, 5 from [Some(100), None, Some(200), Some(300), None, Some(400)].
        let int_vector = fields[0].as_primitive().clone().into_i32();
        let int_values: Vec<Option<i32>> = (0..5).map(|i| int_vector.get(i).copied()).collect();
        assert_eq!(
            int_values,
            vec![Some(100), None, Some(200), None, Some(400)]
        );

        // Verify boolean field values.
        // Selected indices: 0, 1, 2, 4, 5 from [T, F, T, F, T, F].
        let bool_vector = fields[1].as_bool();
        let bool_values: Vec<bool> = (0..5).map(|i| bool_vector.bits().value(i)).collect();
        assert_eq!(bool_values, vec![true, false, true, true, false]);

        // Verify the struct-level validity is correctly propagated.
        // Original struct validity: [T, F, T, T, F, T]
        // Selected indices: 0, 1, 2, 4, 5 -> validity: [T, F, T, F, T].
        let validity_mask = struct_vector.validity();
        let struct_validity_values: Vec<bool> = (0..5).map(|i| validity_mask.value(i)).collect();
        assert_eq!(struct_validity_values, vec![true, false, true, false, true]);
    }

    fn test_struct_array() -> ArrayRef {
        let a_field = PrimitiveArray::from_iter([1i32, 3, 5, 7, 9]);
        let b_field = PrimitiveArray::from_iter([2i32, 4, 6, 8, 10]);

        StructArray::new(
            FieldNames::from(["a", "b"]),
            vec![a_field.into_array(), b_field.into_array()],
            5,
            Validity::NonNullable,
        )
        .into_array()
    }

    #[test]
    fn test_struct_reduce_parent_single_field_simple() -> VortexResult<()> {
        let struct_array = test_struct_array();

        let expr = gt(get_item("a", root()), lit(5));
        let expr_array = ExprArray::new_infer_dtype(struct_array.clone().into_array(), expr)?;

        let actual = expr_array.to_canonical().into_array();
        let expected = (0..5)
            .map(|i| (i * 2 + 1) > 5)
            .collect::<BoolArray>()
            .into_array();

        assert_arrays_eq!(expected, actual);

        // Use the optimizer to apply parent rules
        let array_session = crate::ArraySession::default();
        let expr_session = ExprSession::default();
        let expr_optimizer = ExprOptimizer::new(&expr_session);
        let optimizer = array_session.optimizer(expr_optimizer);

        let result = optimizer.optimize_array(expr_array.into_array())?;

        let result = result.as_::<ExprVTable>();
        assert_eq!(&gt(root(), lit(5i32)), result.expr());

        let actual = result.to_canonical().into_array();
        assert_arrays_eq!(expected, actual);

        Ok(())
    }

    #[test]
    fn test_struct_reduce_parent_single_field_compound() -> VortexResult<()> {
        let struct_array = test_struct_array();

        let expr = and(
            gt(get_item("a", root()), lit(5)),
            lt(get_item("a", root()), lit(10)),
        );
        let expr_array = ExprArray::new_infer_dtype(struct_array.clone().into_array(), expr)?;

        let actual = expr_array.to_canonical().into_array();
        let expected = (0..5)
            .map(|i| (i * 2 + 1) > 5 && (i * 2 + 1) < 10)
            .collect::<BoolArray>()
            .into_array();
        assert_arrays_eq!(expected, actual);

        // Use the optimizer to apply parent rules
        let array_session = crate::ArraySession::default();
        let expr_session = ExprSession::default();
        let expr_optimizer = ExprOptimizer::new(&expr_session);
        let optimizer = array_session.optimizer(expr_optimizer);

        let result = optimizer.optimize_array(expr_array.into_array())?;

        let result = result.as_::<ExprVTable>();
        assert_eq!(
            &and(gt(root(), lit(5i32)), lt(root(), lit(10i32))),
            result.expr()
        );

        let actual = result.to_canonical().into_array();
        assert_arrays_eq!(expected, actual);

        Ok(())
    }

    #[test]
    fn test_struct_reduce_parent_multi_field() -> VortexResult<()> {
        let struct_array = test_struct_array();

        let expr = and(
            and(gt(col("a"), lit(5)), lt(col("b"), lit(4))),
            gt(col("a"), lit(6)),
        );
        let expr_array = ExprArray::new_infer_dtype(struct_array.clone().into_array(), expr)?;

        // Use the optimizer to apply parent rules
        let array_session = crate::ArraySession::default();
        let expr_session = ExprSession::default();
        let expr_optimizer = ExprOptimizer::new(&expr_session);
        let optimizer = array_session.optimizer(expr_optimizer);

        let result = optimizer.optimize_array(expr_array.into_array())?;

        // Assert the result is an ExprArray wrapping a StructArray
        let result_expr = result
            .as_opt::<ExprVTable>()
            .vortex_expect("should be an ExprArray");

        // The field name can change.
        assert_eq!(
            result_expr.expr(),
            &and(
                and(get_item("a_0", col("a")), get_item("b_0", col("b"))),
                get_item("a_1", col("a")),
            ),
        );

        let result_struct = result_expr
            .child()
            .as_opt::<StructVTable>()
            .vortex_expect("child should be a struct");
        assert_eq!(
            result_struct.fields().len(),
            2,
            "Should have 2 fields (a and b)"
        );
        assert_eq!(result_struct.names()[0], "a");
        assert_eq!(result_struct.names()[1], "b");

        // Assert field 'a' is an ExprArray with a pack expression
        let field_a = &result_struct.fields()[0];
        let field_a_expr = field_a
            .as_opt::<ExprVTable>()
            .vortex_expect("field 'a' should be ExprArray");

        assert_eq!(
            &pack(
                [
                    ("a_0", gt(root(), lit(5i32))),
                    ("a_1", gt(root(), lit(6i32)))
                ],
                NonNullable
            ),
            field_a_expr.expr()
        );

        assert!(Arc::ptr_eq(
            &struct_array.as_::<StructVTable>().fields()[0],
            field_a_expr.child()
        ));

        let field_b = &result_struct.fields()[1];
        let field_b_expr = field_b
            .as_opt::<ExprVTable>()
            .vortex_expect("field 'b' should be ExprArray");

        assert_eq!(
            &pack([("b_0", lt(root(), lit(4i32)))], NonNullable),
            field_b_expr.expr()
        );
        assert!(Arc::ptr_eq(
            &struct_array.as_::<StructVTable>().fields()[1],
            field_b_expr.child()
        ));

        Ok(())
    }

    #[test]
    fn test_struct_reduce_parent_constant_expr() -> VortexResult<()> {
        let struct_array = test_struct_array();

        let expr = eq(lit(1), lit(0));
        let expr_array =
            ExprArray::new_infer_dtype(struct_array.clone().into_array(), expr.clone())?;

        let actual = expr_array.to_canonical().into_array();
        let expected = (0..5).map(|_| false).collect::<BoolArray>().into_array();
        assert_arrays_eq!(expected, actual);

        // Use the optimizer to apply parent rules
        let array_session = crate::ArraySession::default();
        let expr_session = ExprSession::default();
        let expr_optimizer = ExprOptimizer::new(&expr_session);
        let optimizer = array_session.optimizer(expr_optimizer);

        let result = optimizer.optimize_array(expr_array.into_array())?;
        let actual = result.to_canonical().into_array();
        assert_arrays_eq!(expected, actual);

        let result_struct = result.as_::<ExprVTable>();

        assert_eq!(result_struct.expr(), &expr);
        assert_arrays_eq!(
            result_struct.child(),
            StructArray::new(FieldNames::empty(), vec![], 5, Validity::NonNullable)
        );

        Ok(())
    }
}
