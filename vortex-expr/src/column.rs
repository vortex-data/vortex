use std::any::Any;

use vortex::aliases::hash_set::HashSet;
use vortex::array::StructArray;
use vortex::variants::StructArrayTrait;
use vortex::Array;
use vortex_dtype::field::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{unbox_any, VortexExpr};

#[derive(Debug, PartialEq, Hash, Clone, Eq)]
pub struct Column {
    field: Field,
}

impl Column {
    pub fn new(field: Field) -> Self {
        Self { field }
    }

    pub fn field(&self) -> &Field {
        &self.field
    }
}

impl From<String> for Column {
    fn from(value: String) -> Self {
        Column::new(value.into())
    }
}

impl From<usize> for Column {
    fn from(value: usize) -> Self {
        Column::new(value.into())
    }
}

impl VortexExpr for Column {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &Array) -> VortexResult<Array> {
        let s = StructArray::try_from(batch)?;

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
