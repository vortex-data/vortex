use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::array::StructArray;
use vortex_array::variants::StructArrayTrait;
use vortex_array::ArrayData;
use vortex_dtype::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, PartialEq, Hash, Clone, Eq)]
pub struct Column {
    field: Field,
}

impl Column {
    pub fn new_expr(field: impl Into<Field>) -> ExprRef {
        Arc::new(Self {
            field: field.into(),
        })
    }

    pub fn field(&self) -> &Field {
        &self.field
    }
}

pub fn col(field: impl Into<Field>) -> ExprRef {
    Arc::new(Column {
        field: field.into(),
    })
}

impl From<String> for Column {
    fn from(value: String) -> Self {
        Column {
            field: value.into(),
        }
    }
}

impl From<usize> for Column {
    fn from(value: usize) -> Self {
        Column {
            field: value.into(),
        }
    }
}

impl Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.field)
    }
}

impl VortexExpr for Column {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let s = StructArray::try_from(batch.clone())?;

        match &self.field {
            Field::Name(n) => s.field_by_name(n),
            Field::Index(i) => s.field(*i),
        }
        .ok_or_else(|| vortex_err!("Array doesn't contain child array {}", self.field))
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 0);
        self
    }
}
