use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::ArrayData;
use vortex_dtype::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone, Eq, Hash)]
#[allow(clippy::derived_hash_with_manual_eq)]
pub struct GetItem {
    field: Field,
    child: ExprRef,
}

impl GetItem {
    pub fn new_expr(field: impl Into<Field>, child: ExprRef) -> ExprRef {
        Arc::new(Self {
            field: field.into(),
            child,
        })
    }

    pub fn field(&self) -> &Field {
        &self.field
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }
}

pub fn get_item(field: impl Into<Field>, child: ExprRef) -> ExprRef {
    GetItem::new_expr(field, child)
}

impl Display for GetItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.child, self.field)
    }
}

impl VortexExpr for GetItem {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let child = self.child.evaluate(batch)?;
        child
            .as_struct_array()
            .ok_or_else(|| vortex_err!("GetItem: child array into struct"))?
            // TODO(joe): apply struct validity
            .maybe_null_field(self.field())
            .ok_or_else(|| vortex_err!("Field {} not found", self.field))
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![self.child()]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 1);
        Self::new_expr(self.field().clone(), children[0].clone())
    }
}

impl PartialEq for GetItem {
    fn eq(&self, other: &GetItem) -> bool {
        self.field == other.field && self.child.eq(&other.child)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::StructArray;
    use vortex_array::{ArrayDType, IntoArrayData};
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::PType::{I32, I64};

    use crate::get_item::get_item;
    use crate::ident;

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
        let get_item = get_item("a", ident());
        let item = get_item.evaluate(st.as_ref()).unwrap();
        assert_eq!(item.dtype(), &DType::from(I32))
    }

    #[test]
    pub fn get_item_by_name_none() {
        let st = test_array();
        let get_item = get_item("c", ident());
        assert!(get_item.evaluate(st.as_ref()).is_err());
    }

    #[test]
    pub fn get_item_by_idx() {
        let st = test_array();
        let get_item = get_item(1, ident());
        let item = get_item.evaluate(st.as_ref()).unwrap();
        assert_eq!(item.dtype(), &DType::from(I64))
    }
}
