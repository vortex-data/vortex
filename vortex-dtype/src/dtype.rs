use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::Index;
use std::sync::Arc;

use DType::*;
use itertools::Itertools;
use static_assertions::const_assert_eq;
use vortex_error::vortex_panic;

use crate::decimal::DecimalDType;
use crate::nullability::Nullability;
use crate::{ExtDType, FieldDType, PType, StructFields};

/// A name for a field in a struct
pub type FieldName = Arc<str>;

/// An ordered list of field names in a struct
#[derive(Clone, PartialEq, Eq, Debug, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FieldNames(Arc<[FieldName]>);

impl FieldNames {
    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if the number of elements is 0.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a borrowed iterator over the field names.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &FieldName> {
        FieldNamesIter {
            inner: self,
            idx: 0,
        }
    }

    /// Returns a reference to a field name, or None if `index` is out of bounds.
    pub fn get(&self, index: usize) -> Option<&FieldName> {
        self.0.get(index)
    }
}

impl AsRef<[FieldName]> for FieldNames {
    fn as_ref(&self) -> &[FieldName] {
        &self.0
    }
}

impl Index<usize> for FieldNames {
    type Output = FieldName;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// Iterator of references to field names
pub struct FieldNamesIter<'a> {
    inner: &'a FieldNames,
    idx: usize,
}

impl<'a> Iterator for FieldNamesIter<'a> {
    type Item = &'a FieldName;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.inner.len() {
            return None;
        }

        let i = &self.inner.0[self.idx];
        self.idx += 1;
        Some(i)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len() - self.idx;
        (len, Some(len))
    }
}

impl ExactSizeIterator for FieldNamesIter<'_> {}

/// Owned iterator of field names.
pub struct FieldNamesIntoIter {
    inner: FieldNames,
    idx: usize,
}

impl Iterator for FieldNamesIntoIter {
    type Item = FieldName;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.inner.len() {
            return None;
        }

        let i = self.inner.0[self.idx].clone();
        self.idx += 1;
        Some(i)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len() - self.idx;
        (len, Some(len))
    }
}

impl ExactSizeIterator for FieldNamesIntoIter {}

impl IntoIterator for FieldNames {
    type Item = FieldName;

    type IntoIter = FieldNamesIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        FieldNamesIntoIter {
            inner: self,
            idx: 0,
        }
    }
}

impl From<Vec<FieldName>> for FieldNames {
    fn from(value: Vec<FieldName>) -> Self {
        Self(value.into())
    }
}

impl From<&[&'static str]> for FieldNames {
    fn from(value: &[&'static str]) -> Self {
        Self(value.iter().cloned().map(Arc::from).collect())
    }
}

impl From<&[FieldName]> for FieldNames {
    fn from(value: &[FieldName]) -> Self {
        Self(Arc::from(value))
    }
}

impl<const N: usize> From<[&'static str; N]> for FieldNames {
    fn from(value: [&'static str; N]) -> Self {
        Self(value.into_iter().map(Arc::from).collect())
    }
}

impl<const N: usize> From<[FieldName; N]> for FieldNames {
    fn from(value: [FieldName; N]) -> Self {
        Self(value.into())
    }
}

impl<F: Into<FieldName>> FromIterator<F> for FieldNames {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        Self(iter.into_iter().map(|v| v.into()).collect())
    }
}

/// The logical types of elements in Vortex arrays.
///
/// Vortex arrays preserve a single logical type, while the encodings allow for multiple
/// physical ways to encode that type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DType {
    /// The logical null type (only has a single value, `null`)
    Null,
    /// The logical boolean type (`true` or `false` if non-nullable; `true`, `false`, or `null` if nullable)
    Bool(Nullability),
    /// Primitive, fixed-width numeric types (e.g., `u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `u64`, `i64`, `f32`, `f64`)
    Primitive(PType, Nullability),
    /// Real numbers with fixed exact precision and scale.
    Decimal(DecimalDType, Nullability),
    /// UTF-8 strings
    Utf8(Nullability),
    /// Binary data
    Binary(Nullability),
    /// A struct is composed of an ordered list of fields, each with a corresponding name and DType
    Struct(StructFields, Nullability),
    /// A variable-length list type, parameterized by a single element DType
    List(Arc<DType>, Nullability),
    /// User-defined extension types
    Extension(Arc<ExtDType>),
}

#[cfg(not(target_arch = "wasm32"))]
const_assert_eq!(size_of::<DType>(), 16);

#[cfg(target_arch = "wasm32")]
const_assert_eq!(size_of::<DType>(), 8);

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
            Extension(ext_dtype) => ext_dtype.storage_dtype().is_nullable(),
            Bool(n)
            | Primitive(_, n)
            | Decimal(_, n)
            | Utf8(n)
            | Binary(n)
            | Struct(_, n)
            | List(_, n) => matches!(n, Nullable),
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
            Decimal(d, _) => Decimal(*d, nullability),
            Utf8(_) => Utf8(nullability),
            Binary(_) => Binary(nullability),
            Struct(st, _) => Struct(st.clone(), nullability),
            List(c, _) => List(c.clone(), nullability),
            Extension(ext) => Extension(Arc::new(ext.with_nullability(nullability))),
        }
    }

    /// Union the nullability of this dtype with the other nullability, returning a new dtype.
    pub fn union_nullability(&self, other: Nullability) -> Self {
        let nullability = self.nullability() | other;
        self.with_nullability(nullability)
    }

    /// Check if `self` and `other` are equal, ignoring nullability
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        match (self, other) {
            (Null, Null) => true,
            (Bool(_), Bool(_)) => true,
            (Primitive(lhs_ptype, _), Primitive(rhs_ptype, _)) => lhs_ptype == rhs_ptype,
            (Decimal(lhs, _), Decimal(rhs, _)) => lhs == rhs,
            (Utf8(_), Utf8(_)) => true,
            (Binary(_), Binary(_)) => true,
            (List(lhs_dtype, _), List(rhs_dtype, _)) => lhs_dtype.eq_ignore_nullability(rhs_dtype),
            (Struct(lhs_dtype, _), Struct(rhs_dtype, _)) => {
                (lhs_dtype.names() == rhs_dtype.names())
                    && (lhs_dtype
                        .fields()
                        .zip_eq(rhs_dtype.fields())
                        .all(|(l, r)| l.eq_ignore_nullability(&r)))
            }
            (Extension(lhs_extdtype), Extension(rhs_extdtype)) => {
                lhs_extdtype.as_ref().eq_ignore_nullability(rhs_extdtype)
            }
            _ => false,
        }
    }

    /// Check if `self` is a `StructDType`
    pub fn is_struct(&self) -> bool {
        matches!(self, Struct(_, _))
    }

    /// Check if `self` is a primitive tpye
    pub fn is_primitive(&self) -> bool {
        matches!(self, Primitive(_, _))
    }

    /// Returns this DType's `PType` if it is a primitive type, otherwise panics.
    pub fn as_ptype(&self) -> PType {
        match self {
            Primitive(ptype, _) => *ptype,
            _ => vortex_panic!("DType is not a primitive type"),
        }
    }

    /// Check if `self` is an unsigned integer
    pub fn is_unsigned_int(&self) -> bool {
        if let Primitive(ptype, _) = self {
            return ptype.is_unsigned_int();
        }
        false
    }

    /// Check if `self` is a signed integer
    pub fn is_signed_int(&self) -> bool {
        if let Primitive(ptype, _) = self {
            return ptype.is_signed_int();
        }
        false
    }

    /// Check if `self` is an integer (signed or unsigned)
    pub fn is_int(&self) -> bool {
        if let Primitive(ptype, _) = self {
            return ptype.is_int();
        }
        false
    }

    /// Check if `self` is a floating point number
    pub fn is_float(&self) -> bool {
        if let Primitive(ptype, _) = self {
            return ptype.is_float();
        }
        false
    }

    /// Check if `self` is a boolean
    pub fn is_boolean(&self) -> bool {
        matches!(self, Bool(_))
    }

    /// Check if `self` is a binary
    pub fn is_binary(&self) -> bool {
        matches!(self, Binary(_))
    }

    /// Check if `self` is a utf8
    pub fn is_utf8(&self) -> bool {
        matches!(self, Utf8(_))
    }

    /// Check if `self` is an extension type
    pub fn is_extension(&self) -> bool {
        matches!(self, Extension(_))
    }

    /// Check if `self` is a decimal type
    pub fn is_decimal(&self) -> bool {
        matches!(self, Decimal(..))
    }

    /// Check returns the inner decimal type if the dtype is a decimal
    pub fn as_decimal(&self) -> Option<&DecimalDType> {
        match self {
            Decimal(decimal, _) => Some(decimal),
            _ => None,
        }
    }

    /// Get the `StructDType` if `self` is a `StructDType`, otherwise `None`
    pub fn as_struct(&self) -> Option<&StructFields> {
        match self {
            Struct(s, _) => Some(s),
            _ => None,
        }
    }

    /// Get the inner dtype if `self` is a `ListDType`, otherwise `None`
    pub fn as_list_element(&self) -> Option<&Arc<DType>> {
        match self {
            List(s, _) => Some(s),
            _ => None,
        }
    }

    /// Convenience method for creating a struct dtype
    pub fn struct_<I: IntoIterator<Item = (impl Into<FieldName>, impl Into<FieldDType>)>>(
        iter: I,
        nullability: Nullability,
    ) -> Self {
        Struct(StructFields::from_iter(iter), nullability)
    }

    /// Convenience method for creating a list dtype
    pub fn list(dtype: impl Into<DType>, nullability: Nullability) -> Self {
        List(Arc::new(dtype.into()), nullability)
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Null => write!(f, "null"),
            Bool(n) => write!(f, "bool{n}"),
            Primitive(pt, n) => write!(f, "{pt}{n}"),
            Decimal(dt, n) => write!(f, "{dt}{n}"),
            Utf8(n) => write!(f, "utf8{n}"),
            Binary(n) => write!(f, "binary{n}"),
            Struct(sdt, n) => write!(
                f,
                "{{{}}}{}",
                sdt.names()
                    .iter()
                    .zip(sdt.fields())
                    .map(|(n, dt)| format!("{n}={dt}"))
                    .join(", "),
                n
            ),
            List(edt, n) => write!(f, "list({edt}){n}"),
            Extension(ext) => write!(
                f,
                "ext({}, {}{}){}",
                ext.id(),
                ext.storage_dtype()
                    .with_nullability(Nullability::NonNullable),
                ext.metadata()
                    .map(|m| format!(", {m:?}"))
                    .unwrap_or_else(|| "".to_string()),
                ext.storage_dtype().nullability(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_names_iter() {
        let names = ["a", "b"];
        let field_names = FieldNames::from(names);
        assert_eq!(field_names.iter().len(), names.len());
        let mut iter = field_names.iter();
        assert_eq!(iter.next(), Some(&"a".into()));
        assert_eq!(iter.next(), Some(&"b".into()));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_field_names_owned_iter() {
        let names = ["a", "b"];
        let field_names = FieldNames::from(names);
        assert_eq!(field_names.clone().into_iter().len(), names.len());
        let mut iter = field_names.into_iter();
        assert_eq!(iter.next(), Some("a".into()));
        assert_eq!(iter.next(), Some("b".into()));
        assert_eq!(iter.next(), None);
    }
}
