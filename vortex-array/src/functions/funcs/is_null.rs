// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::Expression;
use crate::functions::execution::ExecutionCtx;
use crate::functions::signature::{Signature, UnarySignature};
use crate::functions::{EmptyOptions, FunctionId, VTable};
use std::ops::Not;
use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::bool::BoolVector;
use vortex_vector::{Vector, VectorOps};

/// Function that returns `true` for null values, and `false` otherwise.
pub struct IsNull;

impl VTable for IsNull {
    type Options = EmptyOptions;

    fn id(&self) -> FunctionId {
        FunctionId::from("is_null")
    }

    fn signature(&self, _options: &Self::Options) -> impl Signature {
        // Unary signature over any input type.
        UnarySignature
    }

    fn return_dtype(&self, _options: &Self::Options, _arg_types: &[DType]) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn execute(&self, _options: &Self::Options, ctx: &dyn ExecutionCtx) -> VortexResult<Vector> {
        let input = ctx.input_vector(0)?;
        let is_null = input.validity().not();
        Ok(BoolVector::new(is_null.to_bit_buffer(), Mask::new_true(is_null.len())).into())
    }
}

/// Creates an expression that checks for null values.
///
/// Returns a boolean array indicating which positions contain null values.
///
/// ```rust
/// # use vortex_array::expr::{is_null, root};
/// let expr = is_null(root());
/// ```
pub fn is_null(child: Expression) -> Expression {
    IsNull.new_expr((), vec![child])
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Field;
    use vortex_dtype::FieldPath;
    use vortex_dtype::FieldPathSet;
    use vortex_dtype::Nullability;
    use vortex_error::VortexUnwrap as _;
    use vortex_scalar::Scalar;
    use vortex_utils::aliases::hash_map::HashMap;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::is_null;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::expr::exprs::binary::eq;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::root::root;
    use crate::expr::pruning::checked_pruning_expr;
    use crate::expr::test_harness;
    use crate::stats::Stat;
    use crate::IntoArray;

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            is_null(root()).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = is_null(root());
        expr.with_children([root()]).vortex_unwrap();
    }

    #[test]
    fn evaluate_mask() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)])
                .into_array();
        let expected = [false, true, false, true, false];

        let result = is_null(root()).evaluate(&test_array.clone()).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result.scalar_at(i),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn evaluate_all_false() {
        let test_array = buffer![1, 2, 3, 4, 5].into_array();

        let result = is_null(root()).evaluate(&test_array.clone()).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(
            result.as_constant().unwrap(),
            Scalar::bool(false, Nullability::NonNullable)
        );
    }

    #[test]
    fn evaluate_all_true() {
        let test_array =
            PrimitiveArray::from_option_iter(vec![None::<i32>, None, None, None, None])
                .into_array();

        let result = is_null(root()).evaluate(&test_array.clone()).unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(
            result.as_constant().unwrap(),
            Scalar::bool(true, Nullability::NonNullable)
        );
    }

    #[test]
    fn evaluate_struct() {
        let test_array = StructArray::from_fields(&[(
            "a",
            PrimitiveArray::from_option_iter(vec![Some(1), None, Some(2), None, Some(3)])
                .into_array(),
        )])
        .unwrap()
        .into_array();
        let expected = [false, true, false, true, false];

        let result = is_null(get_item("a", root()))
            .evaluate(&test_array.clone())
            .unwrap();

        assert_eq!(result.len(), test_array.len());
        assert_eq!(result.dtype(), &DType::Bool(Nullability::NonNullable));

        for (i, expected_value) in expected.iter().enumerate() {
            assert_eq!(
                result.scalar_at(i),
                Scalar::bool(*expected_value, Nullability::NonNullable)
            );
        }
    }

    #[test]
    fn test_display() {
        let expr = is_null(get_item("name", root()));
        assert_eq!(expr.to_string(), "is_null($.name)");

        let expr2 = is_null(root());
        assert_eq!(expr2.to_string(), "is_null($)");
    }

    #[test]
    fn test_is_null_falsification() {
        let expr = is_null(col("a"));

        let (pruning_expr, st) = checked_pruning_expr(
            &expr,
            &FieldPathSet::from_iter([FieldPath::from_iter([
                Field::Name("a".into()),
                Field::Name("null_count".into()),
            ])]),
        )
        .unwrap();

        assert_eq!(&pruning_expr, &eq(col("a_null_count"), lit(0u64)));
        assert_eq!(
            st.map(),
            &HashMap::from_iter([(FieldPath::from_name("a"), HashSet::from([Stat::NullCount]))])
        );
    }
}
