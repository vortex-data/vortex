use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;
use vortex_array::array::StructArray;
use vortex_array::variants::StructArrayTrait;
use vortex_array::ArrayData;
use vortex_dtype::field::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{unbox_any, ExprRef, VortexExpr};

#[derive(Debug, PartialEq, Hash, Clone, Eq)]
pub struct Column {
    field: Field,
}

impl Column {
    pub fn new_expr(field: Field) -> ExprRef {
        Arc::new(Self { field })
    }

    pub fn field(&self) -> &Field {
        &self.field
    }
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

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        references.insert(self.field());
    }
}

impl PartialEq<dyn Any> for Column {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x == self)
            .unwrap_or(false)
    }
}
