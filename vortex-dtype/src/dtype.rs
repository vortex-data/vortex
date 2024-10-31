use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use DType::*;

use crate::field::Field;
use crate::nullability::Nullability;
use crate::{ExtDType, PType};

pub type FieldName = Arc<str>;
pub type FieldNames = Arc<[FieldName]>;

pub type Metadata = Vec<u8>;

/// Array logical types.
///
/// Vortex arrays preserve a single logical type, while the encodings allow for multiple
/// physical types to encode that type.
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DType {
    Null,
    Bool(Nullability),
    Primitive(PType, Nullability),
    Utf8(Nullability),
    Binary(Nullability),
    Struct(StructDType, Nullability),
    List(Arc<DType>, Nullability),
    Extension(ExtDType, Nullability),
}

impl DType {
    pub const BYTES: Self = Primitive(PType::U8, Nullability::NonNullable);

    /// The default DType for indices
    pub const IDX: Self = Primitive(PType::U64, Nullability::NonNullable);

    pub fn nullability(&self) -> Nullability {
        self.is_nullable().into()
    }

    pub fn is_nullable(&self) -> bool {
        use crate::nullability::Nullability::*;

        match self {
            Null => true,
            Bool(n) => matches!(n, Nullable),
            Primitive(_, n) => matches!(n, Nullable),
            Utf8(n) => matches!(n, Nullable),
            Binary(n) => matches!(n, Nullable),
            Struct(_, n) => matches!(n, Nullable),
            List(_, n) => matches!(n, Nullable),
            Extension(_, n) => matches!(n, Nullable),
        }
    }

    pub fn as_nonnullable(&self) -> Self {
        self.with_nullability(Nullability::NonNullable)
    }

    pub fn as_nullable(&self) -> Self {
        self.with_nullability(Nullability::Nullable)
    }

    pub fn with_nullability(&self, nullability: Nullability) -> Self {
        match self {
            Null => Null,
            Bool(_) => Bool(nullability),
            Primitive(p, _) => Primitive(*p, nullability),
            Utf8(_) => Utf8(nullability),
            Binary(_) => Binary(nullability),
            Struct(st, _) => Struct(st.clone(), nullability),
            List(c, _) => List(c.clone(), nullability),
            Extension(ext, _) => Extension(ext.clone(), nullability),
        }
    }

    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        self.as_nullable().eq(&other.as_nullable())
    }

    pub fn is_struct(&self) -> bool {
        matches!(self, Struct(_, _))
    }

    pub fn is_unsigned_int(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_unsigned_int)
    }

    pub fn is_signed_int(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_signed_int)
    }

    pub fn is_int(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_int)
    }

    pub fn is_float(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_float)
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, Bool(_))
    }

    pub fn as_struct(&self) -> Option<&StructDType> {
        match self {
            Struct(s, _) => Some(s),
            _ => None,
        }
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Null => write!(f, "null"),
            Bool(n) => write!(f, "bool{}", n),
            Primitive(pt, n) => write!(f, "{}{}", pt, n),
            Utf8(n) => write!(f, "utf8{}", n),
            Binary(n) => write!(f, "binary{}", n),
            Struct(sdt, n) => write!(
                f,
                "{{{}}}{}",
                sdt.names()
                    .iter()
                    .zip(sdt.dtypes().iter())
                    .map(|(n, dt)| format!("{}={}", n, dt))
                    .join(", "),
                n
            ),
            List(edt, n) => write!(f, "list({}){}", edt, n),
            Extension(ext, n) => write!(
                f,
                "ext({}{}){}",
                ext.id(),
                ext.metadata()
                    .map(|m| format!(", {:?}", m))
                    .unwrap_or_else(|| "".to_string()),
                n
            ),
        }
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructDType {
    names: FieldNames,
    dtypes: Arc<[DType]>,
}

pub struct FieldInfo<'a> {
    pub index: usize,
    pub name: Arc<str>,
    pub dtype: &'a DType,
}

impl StructDType {
    pub fn new(names: FieldNames, dtypes: Vec<DType>) -> Self {
        Self {
            names,
            dtypes: dtypes.into(),
        }
    }

    pub fn names(&self) -> &FieldNames {
        &self.names
    }

    pub fn find_name(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n.as_ref() == name)
    }

    pub fn field_info(&self, field: &Field) -> VortexResult<FieldInfo> {
        let index = match field {
            Field::Name(name) => self
                .find_name(name)
                .ok_or_else(|| vortex_err!("Unknown field: {}", name))?,
            Field::Index(index) => *index,
        };
        if index > self.names.len() {
            vortex_bail!("field index out of bounds: {}", index)
        }
        Ok(FieldInfo {
            index,
            name: self.names[index].clone(),
            dtype: &self.dtypes[index],
        })
    }

    pub fn dtypes(&self) -> &Arc<[DType]> {
        &self.dtypes
    }

    pub fn project(&self, projection: &[Field]) -> VortexResult<Self> {
        let mut names = Vec::with_capacity(projection.len());
        let mut dtypes = Vec::with_capacity(projection.len());

        for field in projection.iter() {
            let FieldInfo { name, dtype, .. } = self.field_info(field)?;

            names.push(name.clone());
            dtypes.push(dtype.clone());
        }

        Ok(StructDType::new(names.into(), dtypes))
    }
}

#[cfg(test)]
mod test {
    use std::mem;

    use crate::dtype::DType;
    use crate::{Nullability, StructDType};

    #[test]
    fn size_of() {
        assert_eq!(mem::size_of::<DType>(), 40);
    }

    #[test]
    fn is_nullable() {
        assert!(!DType::Struct(
            StructDType::new(vec![].into(), Vec::new()),
            Nullability::NonNullable
        )
        .is_nullable());
    }
}
