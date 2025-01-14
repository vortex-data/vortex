use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use itertools::Itertools;
use DType::*;

use crate::nullability::Nullability;
use crate::{ExtDType, PType, StructDType};

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
    Struct(StructDType, Nullability),
    /// A variable-length list type, parameterized by a single element DType
    List(Arc<DType>, Nullability),
    /// User-defined extension types
    Extension(Arc<ExtDType>),
}

impl DType {
    /// The default DType for bytes
    pub const BYTES: Self = Primitive(PType::U8, Nullability::NonNullable);

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
        match (self, other) {
            (Null, Null) => true,
            (Null, _) => false,
            (Bool(_), Bool(_)) => true,
            (Bool(_), _) => false,
            (Primitive(lhs_ptype, _), Primitive(rhs_ptype, _)) => lhs_ptype == rhs_ptype,
            (Primitive(..), _) => false,
            (Utf8(_), Utf8(_)) => true,
            (Utf8(_), _) => false,
            (Binary(_), Binary(_)) => true,
            (Binary(_), _) => false,
            (List(lhs_dtype, _), List(rhs_dtype, _)) => lhs_dtype.eq_ignore_nullability(rhs_dtype),
            (List(..), _) => false,
            (Struct(lhs_dtype, _), Struct(rhs_dtype, _)) => {
                (lhs_dtype.names() == rhs_dtype.names())
                    && (lhs_dtype
                        .dtypes()
                        .zip_eq(rhs_dtype.dtypes())
                        .all(|(l, r)| l.eq_ignore_nullability(&r)))
            }
            (Struct(..), _) => false,
            (Extension(lhs_extdtype), Extension(rhs_extdtype)) => lhs_extdtype == rhs_extdtype,
            (Extension(_), _) => false,
        }
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

    /// Get the inner dtype if `self` is a `ListDType`, otherwise `None`
    pub fn as_list_element(&self) -> Option<&DType> {
        match self {
            List(s, _) => Some(s.as_ref()),
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
                    .zip(sdt.dtypes())
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

#[cfg(test)]
mod test {
    use std::mem;

    use crate::dtype::DType;

    #[test]
    fn size_of() {
        assert_eq!(mem::size_of::<DType>(), 40);
    }
}
