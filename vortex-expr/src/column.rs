use std::any::Any;
use std::fmt::Display;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::{ArrayDType, ArrayData};
use vortex_dtype::FieldName;
use vortex_error::{vortex_err, VortexResult};

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Column {
    field: FieldName,
}

impl Column {
    pub fn new_expr(field: impl Into<FieldName>) -> ExprRef {
        Arc::new(Self {
            field: field.into(),
        })
    }

    pub fn field(&self) -> &FieldName {
        &self.field
    }
}

pub fn col(field: impl Into<FieldName>) -> ExprRef {
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

impl Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.field)
    }
}

impl VortexExpr for Column {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn unchecked_evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        batch
            .clone()
            .as_struct_array()
            .ok_or_else(|| {
                vortex_err!(
                    "Array must be a struct array, however it was a {}",
                    batch.dtype()
                )
            })?
            .maybe_null_field_by_name(&self.field)
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

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability, PType};

    use crate::{col, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(
            col("a").return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(
            col("col1").return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::U16, Nullability::Nullable)
        );
    }
}
