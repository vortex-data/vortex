// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::ops::Deref;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::compute::cast as compute_cast;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_proto::expr as pb;

use crate::expression::Expression;
use crate::{ChildName, ExprId, ExpressionView, StatsCatalog, VTable, VTableExt};

/// A cast expression that converts values to a target data type.
pub struct Cast;

impl VTable for Cast {
    type Instance = DType;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.cast")
    }

    fn serialize(&self, instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::CastOpts {
                target: Some(instance.into()),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        Ok(Some(
            pb::CastOpts::decode(metadata)?
                .target
                .as_ref()
                .ok_or_else(|| vortex_err!("Missing target dtype in Cast expression"))?
                .try_into()?,
        ))
    }

    fn validate(&self, expr: &ExpressionView<Self>) -> VortexResult<()> {
        if expr.children().len() != 1 {
            vortex_bail!(
                "Cast expression requires exactly 1 child, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Cast expression", child_idx),
        }
    }

    fn fmt_sql(&self, expr: &ExpressionView<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "cast(")?;
        expr.children()[0].fmt_sql(f)?;
        write!(f, " as {}", expr.data())?;
        write!(f, ")")
    }

    fn fmt_data(&self, instance: &Self::Instance, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", instance)
    }

    fn return_dtype(&self, expr: &ExpressionView<Self>, _scope: &DType) -> VortexResult<DType> {
        Ok(expr.data().clone())
    }

    fn evaluate(&self, expr: &ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let array = expr.children()[0].evaluate(scope)?;
        compute_cast(&array, expr.data()).map_err(|e| {
            e.with_context(format!(
                "Failed to cast array of dtype {} to {}",
                array.dtype(),
                expr.deref()
            ))
        })
    }

    fn stat_max(
        &self,
        expr: &ExpressionView<Self>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        expr.children()[0].stat_max(catalog)
    }

    fn stat_min(
        &self,
        expr: &ExpressionView<Self>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        expr.children()[0].stat_min(catalog)
    }

    fn stat_nan_count(
        &self,
        expr: &ExpressionView<Self>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        expr.children()[0].stat_nan_count(catalog)
    }

    fn stat_field_path(&self, expr: &ExpressionView<Self>) -> Option<FieldPath> {
        expr.children()[0].stat_field_path()
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
    Cast.try_new_expr(target, [child])
        .vortex_expect("Failed to create Cast expression")
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::VortexUnwrap as _;

    use super::cast;
    use crate::exprs::get_item::get_item;
    use crate::exprs::root::root;
    use crate::{Expression, test_harness};

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
        expr.with_children(vec![root()]).vortex_unwrap();
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
        let result = expr.evaluate(&test_array).unwrap();

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
        assert_eq!(expr.to_string(), "cast($.value as i64)");

        let expr2 = cast(root(), DType::Bool(Nullability::Nullable));
        assert_eq!(expr2.to_string(), "cast($ as bool?)");
    }
}
