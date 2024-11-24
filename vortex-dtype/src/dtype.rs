use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexResult};
use DType::*;

use crate::field::Field;
use crate::nullability::Nullability;
use crate::{ExtDType, PType};

/// A name for a field in a struct
pub type FieldName = Arc<str>;
/// An ordered list of field names in a struct
pub type FieldNames = Arc<[FieldName]>;

/// The logical types of elements in Vortex arrays.
///
/// Vortex arrays preserve a single logical type, while the encodings allow for multiple
/// physical ways to encode that type.
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DType {
    /// The logical null type (only has a single value, `null`)
    Null,
    /// The logical boolean type (`true` or `false` if non-nullable; `true`, `false`, or `null` if nullable)
    Bool(Nullability),
    /// Primitive, fixed-width numeric types (e.g., `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `f32`, `f64`)
    Primitive(PType, Nullability),
    /// UTF-8 strings
    Utf8(Nullability),
    /// Binary data
    Binary(Nullability),
    /// A struct is composed of an ordered list of fields, each with a corresponding name and DType
    /// TODO(ngates): we may want StructDType to be Arc<[Field]> instead so it's only a single Arc.
    Struct(StructDType, Nullability),
    /// A variable-length list type, parameterized by a single element DType
    List(Arc<DType>, Nullability),
    /// User-defined extension types
    Extension(Arc<ExtDType>),
}

impl DType {
    /// The default DType for bytes
    pub const BYTES: Self = Primitive(PType::U8, Nullability::NonNullable);

    /// The default DType for indices
    pub const IDX: Self = Primitive(PType::U64, Nullability::NonNullable);

    /// The DType for small indices (primarily created from bitmaps)
    pub const IDX_32: Self = Primitive(PType::U32, Nullability::NonNullable);

    /// Get the nullability of the DType
    pub fn nullability(&self) -> Nullability {
        self.is_nullable().into()
    }

    /// Check if the DType is nullable
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
            Extension(ext_dtype) => ext_dtype.storage_dtype().is_nullable(),
        }
    }

    /// Get a new DType with `Nullability::NonNullable` (but otherwise the same as `self`)
    pub fn as_nonnullable(&self) -> Self {
        self.with_nullability(Nullability::NonNullable)
    }

    /// Get a new DType with `Nullability::Nullable` (but otherwise the same as `self`)
    pub fn as_nullable(&self) -> Self {
        self.with_nullability(Nullability::Nullable)
    }

    /// Get a new DType with the given nullability (but otherwise the same as `self`)
    pub fn with_nullability(&self, nullability: Nullability) -> Self {
        match self {
            Null => Null,
            Bool(_) => Bool(nullability),
            Primitive(p, _) => Primitive(*p, nullability),
            Utf8(_) => Utf8(nullability),
            Binary(_) => Binary(nullability),
            Struct(st, _) => Struct(st.clone(), nullability),
            List(c, _) => List(c.clone(), nullability),
            Extension(ext) => Extension(Arc::new(ext.with_nullability(nullability))),
        }
    }

    /// Check if `self` and `other` are equal, ignoring nullability
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        self.as_nullable().eq(&other.as_nullable())
    }

    /// Check if `self` is a `StructDType`
    pub fn is_struct(&self) -> bool {
        matches!(self, Struct(_, _))
    }

    /// Check if `self` is an unsigned integer
    pub fn is_unsigned_int(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_unsigned_int)
    }

    /// Check if `self` is a signed integer
    pub fn is_signed_int(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_signed_int)
    }

    /// Check if `self` is an integer (signed or unsigned)
    pub fn is_int(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_int)
    }

    /// Check if `self` is a floating point number
    pub fn is_float(&self) -> bool {
        PType::try_from(self).is_ok_and(PType::is_float)
    }

    /// Check if `self` is a boolean
    pub fn is_boolean(&self) -> bool {
        matches!(self, Bool(_))
    }

    /// Get the `StructDType` if `self` is a `StructDType`, otherwise `None`
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
            Extension(ext) => write!(
                f,
                "ext({}, {}{}){}",
                ext.id(),
                ext.storage_dtype()
                    .with_nullability(Nullability::NonNullable),
                ext.metadata()
                    .map(|m| format!(", {:?}", m))
                    .unwrap_or_else(|| "".to_string()),
                ext.storage_dtype().nullability(),
            ),
        }
    }
}

/// A struct dtype is a list of names and corresponding dtypes
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StructDType {
    names: FieldNames,
    dtypes: Arc<[DType]>,
}

/// Information about a field in a struct dtype
pub struct FieldInfo<'a> {
    /// The position index of the field within the enclosing struct
    pub index: usize,
    /// The name of the field
    pub name: Arc<str>,
    /// The dtype of the field
    pub dtype: &'a DType,
}

impl StructDType {
    /// Create a new `StructDType` from a list of names and dtypes
    pub fn new(names: FieldNames, dtypes: Vec<DType>) -> Self {
        if names.len() != dtypes.len() {
            vortex_panic!(
                "length mismatch between names ({}) and dtypes ({})",
                names.len(),
                dtypes.len()
            );
        }
        Self {
            names,
            dtypes: dtypes.into(),
        }
    }

    /// Get the names of the fields in the struct
    pub fn names(&self) -> &FieldNames {
        &self.names
    }

    /// Find the index of a field by name
    /// Returns `None` if the field is not found
    pub fn find_name(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n.as_ref() == name)
    }

    /// Get information about the referenced field, either by name or index
    /// Returns an error if the field is not found
    pub fn field_info(&self, field: &Field) -> VortexResult<FieldInfo> {
        let index = match field {
            Field::Name(name) => self
                .find_name(name)
                .ok_or_else(|| vortex_err!("Unknown field: {}", name))?,
            Field::Index(index) => *index,
        };
        if index >= self.names.len() {
            vortex_bail!("field index out of bounds: {}", index)
        }
        Ok(FieldInfo {
            index,
            name: self.names[index].clone(),
            dtype: &self.dtypes[index],
        })
    }

    /// Get the dtypes of the fields in the struct
    pub fn dtypes(&self) -> &Arc<[DType]> {
        &self.dtypes
    }

    /// Project a subset of fields from the struct
    /// Returns an error if any of the referenced fields are not found
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
    use crate::field::Field;
    use crate::{Nullability, PType, StructDType};

    #[test]
    fn size_of() {
        assert_eq!(mem::size_of::<DType>(), 40);
    }

    #[test]
    fn nullability() {
        assert!(!DType::Struct(
            StructDType::new(vec![].into(), Vec::new()),
            Nullability::NonNullable
        )
        .is_nullable());

        let primitive = DType::Primitive(PType::U8, Nullability::Nullable);
        assert!(primitive.is_nullable());
        assert!(!primitive.as_nonnullable().is_nullable());
        assert!(primitive.as_nonnullable().as_nullable().is_nullable());
    }

    #[test]
    fn test_struct() {
        let a_type = DType::Primitive(PType::I32, Nullability::Nullable);
        let b_type = DType::Bool(Nullability::NonNullable);

        let dtype = DType::Struct(
            StructDType::new(
                vec!["A".into(), "B".into()].into(),
                vec![a_type.clone(), b_type.clone()],
            ),
            Nullability::Nullable,
        );
        assert!(dtype.is_nullable());
        assert!(dtype.as_struct().is_some());
        assert!(a_type.as_struct().is_none());

        let sdt = dtype.as_struct().unwrap();
        assert_eq!(sdt.names().len(), 2);
        assert_eq!(sdt.dtypes().len(), 2);
        assert_eq!(sdt.names()[0], "A".into());
        assert_eq!(sdt.names()[1], "B".into());
        assert_eq!(sdt.dtypes()[0], a_type);
        assert_eq!(sdt.dtypes()[1], b_type);

        let proj = sdt
            .project(&[Field::Index(1), Field::Name("A".into())])
            .unwrap();
        assert_eq!(proj.names()[0], "B".into());
        assert_eq!(proj.dtypes()[0], b_type);
        assert_eq!(proj.names()[1], "A".into());
        assert_eq!(proj.dtypes()[1], a_type);

        let field_info = sdt.field_info(&Field::Name("B".into())).unwrap();
        assert_eq!(field_info.index, 1);
        assert_eq!(field_info.name, "B".into());
        assert_eq!(field_info.dtype, &b_type);

        let field_info = sdt.field_info(&Field::Index(0)).unwrap();
        assert_eq!(field_info.index, 0);
        assert_eq!(field_info.name, "A".into());
        assert_eq!(field_info.dtype, &a_type);

        assert!(sdt.field_info(&Field::Index(2)).is_err());

        assert_eq!(sdt.find_name("A"), Some(0));
        assert_eq!(sdt.find_name("B"), Some(1));
        assert_eq!(sdt.find_name("C"), None);
    }
}
