// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::ops::Not;

use vortex_array::arrays::{BoolArray, ConstantArray};
use vortex_array::stats::Stat;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::exprs::binary::eq;
use crate::exprs::literal::lit;
use crate::{ChildName, ExprId, Expression, ExpressionView, StatsCatalog, VTable, VTableExt};

/// Expression that checks for null values.
pub struct IsNull;

impl VTable for IsNull {
    type Instance = ();

    fn id(&self) -> ExprId {
        ExprId::new_ref("is_null")
    }

    fn serialize(&self, _instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        Ok(Some(()))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if expr.children().len() != 1 {
            vortex_bail!(
                "IsNull expression expects exactly one child, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for IsNull expression", child_idx),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "is_null(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _expr: &ExpressionView<Self>, _scope: &DType) -> VortexResult<DType> {
        Ok(DType::Bool(Nullability::NonNullable))
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let array = expr.child(0).evaluate(scope)?;
        match array.validity_mask() {
            Mask::AllTrue(len) => Ok(ConstantArray::new(false, len).into_array()),
            Mask::AllFalse(len) => Ok(ConstantArray::new(true, len).into_array()),
            Mask::Values(mask) => Ok(BoolArray::from(mask.bit_buffer().not()).into_array()),
        }
    }

    fn stat_falsification(
        &self,
        expr: &ExpressionView<Self>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        let field_path = expr.children()[0].stat_field_path()?;
        let null_count_expr = catalog.stats_ref(&field_path, Stat::NullCount)?;
        Some(eq(null_count_expr, lit(0u64)))
    }
}

/// Creates an expression that checks for null values.
///
/// Returns a boolean array indicating which positions contain null values.
///
/// ```rust
/// # use vortex_expr::{is_null, root};
/// let expr = is_null(root());
/// ```
pub fn is_null(child: Expression) -> Expression {
    IsNull.new_expr((), vec![child])
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::{PrimitiveArray, StructArray};
    use vortex_array::stats::Stat;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Field, FieldPath, FieldPathSet, Nullability};
    use vortex_error::VortexUnwrap as _;
    use vortex_scalar::Scalar;
    use vortex_utils::aliases::hash_map::HashMap;

    use super::is_null;
    use crate::exprs::binary::eq;
    use crate::exprs::get_item::{col, get_item};
    use crate::exprs::literal::lit;
    use crate::exprs::root::root;
    use crate::pruning::checked_pruning_expr;
    use crate::{HashSet, test_harness};

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
