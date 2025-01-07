use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use vortex_array::variants::StructArrayTrait;
use vortex_array::{ArrayData, IntoArrayVariant};
use vortex_dtype::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone)]
struct GetItem {
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

impl PartialEq<dyn Any> for GetItem {
    fn eq(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<GetItem>()
            .map(|item| self.field == item.field && self.child.eq(&item.child))
            .unwrap_or(false)
    }
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
        let st = child
            .into_struct()
            .map_err(|e| e.with_context("GetItem: child array into struct"))?;

        match &self.field {
            Field::Name(name) => st.field_by_name(name.as_ref()),
            Field::Index(idx) => st.field(*idx),
        }
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
