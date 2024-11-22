use vortex_dtype::field::Field;
use vortex_error::{vortex_bail, VortexResult};

// TODO(robert): Add ability to project nested columns.
//  Until datafusion supports nested column pruning we should create a separate variant to implement it
#[derive(Debug, Clone, Default)]
pub enum Projection {
    #[default]
    All,
    Flat(Vec<Field>),
}

impl Projection {
    pub fn new(indices: impl AsRef<[usize]>) -> Self {
        Self::Flat(indices.as_ref().iter().copied().map(Field::from).collect())
    }

    pub fn project(&self, fields: &[Field]) -> VortexResult<Self> {
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

impl From<Vec<Field>> for Projection {
    fn from(indices: Vec<Field>) -> Self {
        Self::Flat(indices)
    }
}

impl From<Vec<usize>> for Projection {
    fn from(indices: Vec<usize>) -> Self {
        Self::Flat(indices.into_iter().map(Field::from).collect())
    }
}
