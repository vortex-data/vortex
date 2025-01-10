use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::{ArrayDType, ArrayData};
use vortex_dtype::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
        batch
            .as_struct_array()
            .ok_or_else(|| {
                vortex_err!(
                    "Array must be a struct array, however it was a {}",
                    batch.dtype()
                )
            })?
            .maybe_null_field(&self.field)
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
