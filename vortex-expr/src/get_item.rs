use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::stats::Stat;
use vortex_array::{ArrayRef, ToCanonical};
use vortex_dtype::{DType, FieldName};
use vortex_error::{VortexResult, vortex_err};

use crate::{AccessPath, AnalysisExpr, ExprRef, Scope, ScopeDType, StatsCatalog, VortexExpr, root};

#[derive(Debug, Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct GetItem {
    field: FieldName,
    child: ExprRef,
}

impl GetItem {
    pub fn new_expr(field: impl Into<FieldName>, child: ExprRef) -> ExprRef {
        Arc::new(Self {
            field: field.into(),
            child,
        })
    }

    pub fn field(&self) -> &FieldName {
        &self.field
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }

    pub fn is(expr: &ExprRef) -> bool {
        expr.as_any().is::<Self>()
    }
}

pub fn col(field: impl Into<FieldName>) -> ExprRef {
    GetItem::new_expr(field, root())
}

pub fn get_item(field: impl Into<FieldName>, child: ExprRef) -> ExprRef {
    GetItem::new_expr(field, child)
}

pub fn get_item_scope(field: impl Into<FieldName>) -> ExprRef {
    GetItem::new_expr(field, root())
}

impl Display for GetItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.child, &self.field)
    }
}

#[cfg(feature = "proto")]
pub(crate) mod proto {
    use vortex_error::{VortexResult, vortex_bail};
    use vortex_proto::expr::kind;
    use vortex_proto::expr::kind::Kind;

    use crate::{ExprDeserialize, ExprRef, ExprSerializable, GetItem, Id};

    pub(crate) struct GetItemSerde;

    impl Id for GetItemSerde {
        fn id(&self) -> &'static str {
            "get_item"
        }
    }

    impl ExprDeserialize for GetItemSerde {
        fn deserialize(&self, kind: &Kind, children: Vec<ExprRef>) -> VortexResult<ExprRef> {
            let Kind::GetItem(kind::GetItem { path }) = kind else {
                vortex_bail!("wrong kind {:?}, want get_item", kind)
            };

            Ok(GetItem::new_expr(path.to_string(), children[0].clone()))
        }
    }

    impl ExprSerializable for GetItem {
        fn id(&self) -> &'static str {
            GetItemSerde.id()
        }

        fn serialize_kind(&self) -> VortexResult<Kind> {
            Ok(Kind::GetItem(kind::GetItem {
                path: self.field.to_string(),
            }))
        }
    }
}

impl AnalysisExpr for GetItem {
    fn max(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Max)
    }

    fn min(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        catalog.stats_ref(&self.field_path()?, Stat::Min)
    }

    fn field_path(&self) -> Option<AccessPath> {
        self.child()
            .field_path()
            .map(|fp| AccessPath::new(fp.field_path.push(self.field.clone()), fp.identifier))
    }
}

impl VortexExpr for GetItem {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, scope: &Scope) -> VortexResult<ArrayRef> {
        self.child
            .unchecked_evaluate(scope)?
            .to_struct()?
            .field_by_name(self.field())
            .cloned()
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![self.child()]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 1);
        Self::new_expr(self.field().clone(), children[0].clone())
    }

    fn return_dtype(&self, scope: &ScopeDType) -> VortexResult<DType> {
        let input = self.child.return_dtype(scope)?;
        input
            .as_struct()
            .ok_or_else(|| vortex_err!("GetItem: child dtype is not a struct"))?
            .field(self.field())
    }
}

impl PartialEq for GetItem {
    fn eq(&self, other: &GetItem) -> bool {
        self.field == other.field && self.child.eq(&other.child)
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
