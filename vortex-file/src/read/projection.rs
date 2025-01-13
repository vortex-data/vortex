use vortex_dtype::{FieldName, FieldNames};
use vortex_error::{vortex_bail, VortexResult};

// TODO(robert): Add ability to project nested columns.
//  Until datafusion supports nested column pruning we should create a separate variant to implement it
#[derive(Debug, Clone, Default)]
pub enum Projection {
    #[default]
    All,
    Flat(Vec<FieldName>),
}

impl Projection {
    pub fn new(names: impl Into<FieldNames>) -> Self {
        Self::Flat(names.into().to_vec())
    }

    pub fn project(&self, fields: &[FieldName]) -> VortexResult<Self> {
        Ok(match self {
            Projection::All => Projection::Flat(fields.to_vec()),
            Projection::Flat(own_projection) => {
                if !fields.iter().all(|f| own_projection.contains(f)) {
                    vortex_bail!("Can't project {own_projection:?} into {fields:?}")
                }
                Projection::Flat(fields.to_vec())
            }
        })
    }
}

impl From<Vec<FieldName>> for Projection {
    fn from(fields: Vec<FieldName>) -> Self {
        Self::Flat(fields)
    }
}
