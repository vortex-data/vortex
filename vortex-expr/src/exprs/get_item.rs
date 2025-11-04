// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::ops::Not;

use vortex_array::compute::mask;
use vortex_array::stats::Stat;
use vortex_array::{ArrayRef, DeserializeMetadata, ProstMetadata, ToCanonical};
use vortex_dtype::{DType, FieldName, FieldPath, Nullability};
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_proto::expr as pb;

use crate::display::{DisplayAs, DisplayFormat};
use crate::{root, vtable, AnalysisExpr, ExprEncodingRef, ExprId, Expression, IntoExpr, Scope, StatsCatalog, VTable};

vtable!(GetItem);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, Hash, Eq)]
pub struct GetItemExpr {
    field: FieldName,
    child: Expression,
}

impl PartialEq for GetItemExpr {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field && self.child.eq(&other.child)
    }
}

pub struct GetItemExprEncoding;

pub struct GetItemMetadata {
    pub path: String,
}

impl VTable for GetItemVTable {
    type Encoding = GetItemExprEncoding;
    type Metadata = GetItemMetadata;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("get_item")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(GetItemExprEncoding.as_ref())
    }

    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(ProstMetadata(pb::GetItemOpts {
            path: expr.field.to_string(),
        }))
    }

    fn children(expr: &Self::Expr) -> Vec<&Expression> {
        vec![&expr.child]
    }

    fn with_children(expr: &Self::Expr, children: Vec<Expression>) -> VortexResult<Self::Expr> {
        Ok(GetItemExpr {
            field: expr.field.clone(),
            child: children[0].clone(),
        })
    }

    fn build(
        _encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<Expression>,
    ) -> VortexResult<Self::Expr> {
        if children.len() != 1 {
            vortex_bail!(
                "GetItem expression must have exactly 1 child, got {}",
                children.len()
            );
        }

        let field = FieldName::from(metadata.path.clone());
        Ok(GetItemExpr {
            field,
            child: children[0].clone(),
        })
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let input = expr.child.unchecked_evaluate(scope)?.to_struct();
        let field = input.field_by_name(expr.field()).cloned()?;

        match input.dtype().nullability() {
            Nullability::NonNullable => Ok(field),
            Nullability::Nullable => mask(&field, &input.validity_mask().not()),
        }
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let input = expr.child.return_dtype(scope)?;
        input
            .as_struct_fields_opt()
            .and_then(|st| st.field(expr.field()))
            .map(|f| f.union_nullability(input.nullability()))
            .ok_or_else(|| {
                vortex_err!(
                    "Couldn't find the {} field in the input scope",
                    expr.field()
                )
            })
    }

    type AnalysisVTable = ;

    fn validate(expr: &ExpressionView<Self>) -> VortexResult<()> {
        todo!()
    }

    fn child_name(expr: ExpressionView<Self>, _n: usize) -> ChildName {
        todo!()
    }
}

impl GetItemExpr {
    pub fn new(field: impl Into<FieldName>, child: Expression) -> Self {
        Self {
            field: field.into(),
            child,
        }
    }

    pub fn new_expr(field: impl Into<FieldName>, child: Expression) -> Expression {
        Self::new(field, child).into_expr()
    }

    pub fn field(&self) -> &FieldName {
        &self.field
    }

    pub fn child(&self) -> &Expression {
        &self.child
    }

    pub fn is(expr: &Expression) -> bool {
        expr.is::<GetItemVTable>()
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
    GetItemExpr::new(field, root()).into_expr()
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
    GetItemExpr::new(field, child).into_expr()
}

impl DisplayAs for GetItemExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
                write!(f, "{}.{}", self.child, &self.field)
            }
            DisplayFormat::Tree => {
                write!(f, "GetItem({})", self.field)
            }
        }
    }
}
impl AnalysisExpr for GetItemExpr {
    fn max(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        catalog.stats_ref(&self.field_path()?, Stat::Max)
    }

    fn min(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        catalog.stats_ref(&self.field_path()?, Stat::Min)
    }

    fn nan_count(&self, catalog: &mut dyn StatsCatalog) -> Option<Expression> {
        catalog.stats_ref(&self.field_path()?, Stat::NaNCount)
    }

    fn field_path(&self) -> Option<FieldPath> {
        self.child()
            .field_path()
            .map(|fp| fp.push(self.field.clone()))
    }
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
