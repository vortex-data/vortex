use vortex_dtype::field::Field;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use self::projection::Projection;

pub mod projection;

#[derive(Clone, Debug)]
pub struct Schema(pub(crate) DType);

impl Schema {
    pub fn new(schema_dtype: DType) -> Self {
        Self(schema_dtype)
    }

    pub fn project(&self, projection: &Projection) -> VortexResult<Self> {
        match projection {
            Projection::All => Ok(self.clone()),
            Projection::Flat(fields) => {
                let DType::Struct(s, n) = &self.0 else {
                    vortex_bail!("Can't project non struct types")
                };
                s.project(fields.as_ref())
                    .map(|p| Self(DType::Struct(p, *n)))
            }
            Projection::SingleField(f) => {
                let DType::Struct(s, _) = &self.0 else {
                    vortex_bail!("Can't project non struct types")
                };
                Ok(Self(s.field_info(f)?.dtype.clone()))
            }
        }
    }

    pub fn dtype(&self) -> &DType {
        &self.0
    }

    pub fn field_type(&self, field: &Field) -> VortexResult<DType> {
        let DType::Struct(s, _) = &self.0 else {
            vortex_bail!("Can't project non struct types")
        };

        let idx = match field {
            Field::Name(name) => s.find_name(name),
            Field::Index(i) => Some(*i),
        };

        idx.and_then(|idx| s.dtypes().get(idx).cloned())
            .ok_or_else(|| vortex_err!("Couldn't find field {field}"))
    }
}

impl From<Schema> for DType {
    fn from(value: Schema) -> Self {
        value.0
    }
}
