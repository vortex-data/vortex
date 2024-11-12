use vortex_dtype::field::Field;
use vortex_error::{vortex_bail, VortexResult};

// TODO(robert): Add ability to project nested columns.
//  Until datafusion supports nested column pruning we should create a separate variant to implement it
#[derive(Debug, Clone, Default)]
pub enum Projection {
    #[default]
    All,
    Flat(Vec<Field>),
    SingleField(Field),
}

impl Projection {
    pub fn new(indices: impl AsRef<[usize]>) -> Self {
        Self::Flat(indices.as_ref().iter().copied().map(Field::from).collect())
    }

    pub fn project(&self, projection: &Projection) -> VortexResult<Self> {
        Ok(match self {
            Projection::All => projection.clone(),
            Projection::Flat(own_projection) => match projection {
                Projection::All => Projection::Flat(own_projection.clone()),
                Projection::Flat(fields) => {
                    if !fields.iter().all(|f| own_projection.contains(f)) {
                        vortex_bail!("Can't project {own_projection:?} into {fields:?}")
                    }
                    Projection::Flat(fields.clone())
                }
                Projection::SingleField(f) => {
                    if !own_projection.contains(f) {
                        vortex_bail!("{own_projection:?} doesn't contain field {f}")
                    }
                    Projection::SingleField(f.clone())
                }
            },
            Projection::SingleField(sf) => match projection {
                Projection::All => Projection::SingleField(sf.clone()),
                Projection::Flat(fields) => {
                    if !fields.iter().all(|f| sf == f) {
                        vortex_bail!("Can't project {sf:?} into {fields:?}")
                    }
                    Projection::SingleField(sf.clone())
                }
                Projection::SingleField(pf) => {
                    if pf != sf {
                        vortex_bail!("Can't project single field {sf} to {pf}");
                    }
                    Projection::SingleField(sf.clone())
                }
            },
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
