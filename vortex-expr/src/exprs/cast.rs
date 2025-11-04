// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use vortex_array::compute::cast as compute_cast;
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::v2::Expression;
use crate::{ChildName, ExprId, ExprInstance, NotSupported, VTable, VTableExt};

/// A cast expression that converts values to a target data type.
pub struct Cast;

impl VTable for Cast {
    type Instance = DType;
    type AnalysisVTable = NotSupported;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.cast")
    }

    fn validate(&self, expr: &ExprInstance<Self>) -> VortexResult<()> {
        if expr.children().len() != 1 {
            vortex_bail!(
                "Cast expression requires exactly 1 child, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Cast expression", child_idx),
        }
    }

    fn return_dtype(&self, expr: &ExprInstance<Self>, scope: &DType) -> VortexResult<DType> {
        Ok(expr.deref().clone())
    }

    fn evaluate(&self, expr: &ExprInstance<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let array = expr.children()[0].evaluate(scope)?;
        compute_cast(&array, expr.deref()).map_err(|e| {
            e.with_context(format!(
                "Failed to cast array of dtype {} to {}",
                array.dtype(),
                expr.deref()
            ))
        })
    }
}

/// Creates an expression that casts values to a target data type.
///
/// Converts the input expression's values to the specified target type.
///
/// ```rust
/// # use vortex_dtype::{DType, Nullability, PType};
/// # use vortex_expr::{cast, root};
/// let expr = cast(root(), DType::Primitive(PType::I64, Nullability::NonNullable));
/// ```
pub fn cast(child: Expression, target: DType) -> Expression {
    Cast.try_new(target, [child])
        .vortex_expect("Failed to create Cast expression")
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::StructArray;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{cast, get_item, root, test_harness, Expression, Scope};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            cast(root(), DType::Bool(Nullability::NonNullable))
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
    }

    #[test]
    fn replace_children() {
        let expr = cast(root(), DType::Bool(Nullability::Nullable));
        let _ = expr.with_children(vec![root()]);
    }

    #[test]
    fn evaluate() {
        let test_array = StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array();

        let expr: Expression = cast(
            get_item("a", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        let result = expr.evaluate(&Scope::new(test_array)).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_display() {
        let expr = cast(
            get_item("value", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        assert_eq!(expr.to_string(), "cast($.value, i64)");

        let expr2 = cast(root(), DType::Bool(Nullability::Nullable));
        assert_eq!(expr2.to_string(), "cast($, bool?)");
    }
}
