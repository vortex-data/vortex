// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{
    root, AnalysisExpr, ChildName, ExprId, ExprInstance, Expression, Scope, StatsCatalog,
    VTable, VTableExt,
};
use prost::Message;
use std::fmt::Formatter;
use std::ops::Not;
use vortex_array::compute::mask;
use vortex_array::stats::Stat;
use vortex_array::{ArrayRef, ToCanonical};
use vortex_dtype::{DType, FieldName, FieldPath, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_proto::expr as pb;

pub struct GetItem;

impl VTable for GetItem {
    type Instance = FieldName;

    fn id(&self) -> ExprId {
        ExprId::from("vortex.get_item")
    }

    fn serialize(&self, instance: &Self::Instance) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::GetItemOpts {
                path: instance.to_string(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8]) -> VortexResult<Option<Self::Instance>> {
        let opts = pb::GetItemOpts::decode(metadata)?;
        Ok(Some(FieldName::from(opts.path)))
    }

    fn validate(&self, expr: &ExprInstance<Self>) -> VortexResult<()> {
        if expr.children().len() != 1 {
            vortex_bail!(
                "GetItem expression requires exactly 1 child, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn child_name(&self, _instance: &Self::Instance, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for GetItem expression", child_idx),
        }
    }

    fn fmt_compact(&self, expr: &ExprInstance<Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        expr.children()[0].fmt_compact(f)?;
        write!(f, ".{}", &*expr)
    }

    fn return_dtype(&self, expr: &ExprInstance<Self>, scope: &DType) -> VortexResult<DType> {
        let input = expr.children()[0].return_dtype(scope)?;
        input
            .as_struct_fields_opt()
            .and_then(|st| st.field(**expr))
            .map(|f| f.union_nullability(input.nullability()))
            .ok_or_else(|| vortex_err!("Couldn't find the {} field in the input scope", &*expr))
    }

    fn evaluate(&self, expr: &ExprInstance<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let input = expr.children()[0].evaluate(scope)?.to_struct();
        let field = input.field_by_name(&**expr).cloned()?;

        match input.dtype().nullability() {
            Nullability::NonNullable => Ok(field),
            Nullability::Nullable => mask(&field, &input.validity_mask().not()),
        }
    }

    fn max(&self, expr: &ExprInstance<Self>, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        catalog.stats_ref(&FieldPath::from_name(&*expr), Stat::Max)
    }

    fn min(&self, expr: &ExprInstance<Self>, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        catalog.stats_ref(&FieldPath::from_name(&*expr), Stat::Min)
    }

    fn nan_count(
        &self,
        expr: &ExprInstance<Self>,
        catalog: &mut dyn StatsCatalog,
    ) -> Option<Expression> {
        catalog.stats_ref(&FieldPath::from_name(&*expr), Stat::NaNCount)
    }

    fn field_path(&self, expr: &ExprInstance<Self>) -> Option<FieldPath> {
        expr.children()[0].field_path().map(|fp| fp.push(&*expr))
    }
}

/// Creates an expression that accesses a field from the root array.
///
/// Equivalent to `get_item(field, root())` - extracts a named field from the input array.
///
/// ```rust
/// # use vortex_expr::col;
/// let expr = col("name");
/// ```
pub fn col(field: impl Into<FieldName>) -> Expression {
    GetItem.new(field.into(), vec![root()])
}

/// Creates an expression that extracts a named field from a struct expression.
///
/// Accesses the specified field from the result of the child expression.
///
/// ```rust
/// # use vortex_expr::{get_item, root};
/// let expr = get_item("user_id", root());
/// ```
pub fn get_item(field: impl Into<FieldName>, child: Expression) -> Expression {
    GetItem.new(field.into(), vec![child])
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::StructArray;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, FieldNames, Nullability};
    use vortex_scalar::Scalar;

    use crate::get_item::get_item;
    use crate::{root, Scope};

    fn test_array() -> StructArray {
        StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
    }

    #[test]
    fn get_item_by_name() {
        let st = test_array();
        let get_item = get_item("a", root());
        let item = get_item.evaluate(&Scope::new(st.to_array())).unwrap();
        assert_eq!(item.dtype(), &DType::from(I32))
    }

    #[test]
    fn get_item_by_name_none() {
        let st = test_array();
        let get_item = get_item("c", root());
        assert!(get_item.evaluate(&Scope::new(st.to_array())).is_err());
    }

    #[test]
    fn get_nullable_field() {
        let st = StructArray::try_new(
            FieldNames::from(["a"]),
            vec![buffer![1i32].into_array()],
            1,
            Validity::AllInvalid,
        )
        .unwrap()
        .to_array();

        let get_item = get_item("a", root());
        let item = get_item.evaluate(&Scope::new(st)).unwrap();
        assert_eq!(
            item.scalar_at(0),
            Scalar::null(DType::Primitive(I32, Nullability::Nullable))
        );
    }
}
