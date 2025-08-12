// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use vortex_array::stats::Stat;
use vortex_array::{ArrayRef, DeserializeMetadata, ProstMetadata, ToCanonical};
use vortex_dtype::{DType, FieldName, FieldPath};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_proto::expr as pb;

use crate::{
    AnalysisExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, StatsCatalog, VTable, root,
    vtable,
};

vtable!(GetItem);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Debug, Clone, Hash, Eq)]
pub struct GetItemExpr {
    field: FieldName,
    child: ExprRef,
}

impl PartialEq for GetItemExpr {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field && self.child.eq(&other.child)
    }
}

pub struct GetItemExprEncoding;

impl VTable for GetItemVTable {
    type Expr = GetItemExpr;
    type Encoding = GetItemExprEncoding;
    type Metadata = ProstMetadata<pb::GetItemOpts>;

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

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.child]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(GetItemExpr {
            field: expr.field.clone(),
            child: children[0].clone(),
        })
    }

    fn build(
        _encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
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
        expr.child
            .unchecked_evaluate(scope)?
            .to_struct()?
            .field_by_name(expr.field())
            .cloned()
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let input = expr.child.return_dtype(scope)?;
        input
            .as_struct_opt()
            .and_then(|st| st.field(expr.field()))
            .ok_or_else(|| {
                vortex_err!(
                    "Couldn't find the {} field in the input scope",
                    expr.field()
                )
            })
    }
}

impl GetItemExpr {
    pub fn new(field: impl Into<FieldName>, child: ExprRef) -> Self {
        Self {
            field: field.into(),
            child,
        }
    }

    pub fn new_expr(field: impl Into<FieldName>, child: ExprRef) -> ExprRef {
        Self::new(field, child).into_expr()
    }

    pub fn field(&self) -> &FieldName {
        &self.field
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }

    pub fn is(expr: &ExprRef) -> bool {
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
pub fn col(field: impl Into<FieldName>) -> ExprRef {
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
pub fn get_item(field: impl Into<FieldName>, child: ExprRef) -> ExprRef {
    GetItemExpr::new(field, child).into_expr()
}

impl Display for GetItemExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.child, &self.field)
    }
}

impl AnalysisExpr for GetItemExpr {
    fn max(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Max)
    }

    fn min(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Min)
    }

    fn nan_count(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
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
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::PType::I32;

    use crate::get_item::get_item;
    use crate::{Scope, root};

    fn test_array() -> StructArray {
        StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
    }

    #[test]
    pub fn get_item_by_name() {
        let st = test_array();
        let get_item = get_item("a", root());
        let item = get_item.evaluate(&Scope::new(st.to_array())).unwrap();
        assert_eq!(item.dtype(), &DType::from(I32))
    }

    #[test]
    pub fn get_item_by_name_none() {
        let st = test_array();
        let get_item = get_item("c", root());
        assert!(get_item.evaluate(&Scope::new(st.to_array())).is_err());
    }
}
