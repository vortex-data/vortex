// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::BoundExpr;
use crate::expr::analysis::BooleanLabels;
use crate::expr::label_tree;

pub fn label_is_fallible(expr: &BoundExpr) -> BooleanLabels<'_> {
    label_tree(expr, |expr| expr.is_fallible(), |acc, &child| acc | child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::is_null;
    use crate::expr::lit;
    use crate::expr::merge_opts;
    use crate::expr::not;
    use crate::scalar_fn::fns::merge::DuplicateHandling;

    fn numeric_scope() -> DType {
        let i32_dtype = DType::Primitive(PType::I32, NonNullable);
        DType::Struct(
            StructFields::from_iter([
                ("a", i32_dtype.clone()),
                ("b", i32_dtype),
                ("x", DType::Bool(NonNullable)),
            ]),
            NonNullable,
        )
    }

    fn struct_scope() -> DType {
        let field_dtype = DType::Struct(
            StructFields::from_iter([("value", DType::Primitive(PType::I32, NonNullable))]),
            NonNullable,
        );
        DType::Struct(
            StructFields::from_iter([("a", field_dtype.clone()), ("b", field_dtype)]),
            NonNullable,
        )
    }

    #[test]
    fn not_is_not_fallible() {
        let dtype = numeric_scope();
        let expr = not(col("x", &dtype));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn checked_add_defaults_to_fallible() {
        let dtype = numeric_scope();
        let expr = checked_add(col("a", &dtype), col("b", &dtype));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn eq_not_fallible() {
        let dtype = numeric_scope();
        let expr = eq(col("a", &dtype), lit(5));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn merge_with_error_handling_is_fallible() {
        let field_dtype = DType::Struct(
            StructFields::from_iter([("left_value", DType::Primitive(PType::I32, NonNullable))]),
            NonNullable,
        );
        let other_field_dtype = DType::Struct(
            StructFields::from_iter([("right_value", DType::Primitive(PType::I32, NonNullable))]),
            NonNullable,
        );
        let dtype = DType::Struct(
            StructFields::from_iter([("a", field_dtype), ("b", other_field_dtype)]),
            NonNullable,
        );
        let expr = merge_opts(
            [col("a", &dtype), col("b", &dtype)],
            DuplicateHandling::Error,
        );
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn merge_with_rightmost_handling_is_not_fallible() {
        let dtype = struct_scope();
        let expr = merge_opts(
            [col("a", &dtype), col("b", &dtype)],
            DuplicateHandling::RightMost,
        );
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn nested_with_fallible_child() {
        let dtype = numeric_scope();
        let child = checked_add(col("a", &dtype), col("b", &dtype));
        let expr = eq(child.clone(), lit(0));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&child), Some(&true));
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn nested_without_fallible_child() {
        let dtype = numeric_scope();
        let child = is_null(col("x", &dtype));
        let expr = not(child.clone());
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&child), Some(&false));
        assert_eq!(labels.get(&expr), Some(&false));
    }
}
