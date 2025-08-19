// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use DType::*;
use itertools::Itertools;
use static_assertions::const_assert_eq;
use vortex_error::vortex_panic;

use crate::decimal::DecimalDType;
use crate::nullability::Nullability;
use crate::{ExtDType, FieldDType, FieldName, PType, StructFields};

/// The logical types of elements in Vortex arrays.
///
/// `DType` represents the different logical data types that can be represented in a Vortex array.
///
/// This is different from physical types, which represent the actual layout of data (compressed or
/// uncompressed). The set of physical types/formats (or data layout) is surjective into the set of
/// logical types (or in other words, all physical types map to a single logical type).
///
/// Note that a `DType` represents the logical type of the elements in the `Array`s, **not** the
/// logical type of the `Array` itself.
///
/// For example, an array with [`DType::Primitive`]([`I32`], [`NonNullable`]) could be physically
/// encoded as any of the following:
///
/// - A flat array of `i32` values.
/// - A run-length encoded sequence.
/// - Dictionary encoded values with bitpacked codes.
///
/// All of these physical encodings preserve the same logical [`I32`] type, even if the physical
/// data is different.
///
/// [`I32`]: PType::I32
/// [`NonNullable`]: Nullability::NonNullable
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DType {
    /// A logical null type.
    ///
    /// `Null` only has a single value, `null`.
    Null,

    /// A logical boolean type.
    ///
    /// `Bool` can be `true` or `false` if non-nullable. It can be `true`, `false`, or `null` if
    /// nullable.
    Bool(Nullability),

    /// A logical fixed-width numeric type.
    ///
    /// This can be unsigned, signed, or floating point. See [`PType`] for more information.
    Primitive(PType, Nullability),

    /// Logical real numbers with fixed precision and scale.
    ///
    /// See [`DecimalDType`] for more information.
    Decimal(DecimalDType, Nullability),

    /// Logical UTF-8 strings.
    Utf8(Nullability),

    /// Logical binary data.
    Binary(Nullability),

    /// A logical variable-length list type.
    ///
    /// This is parameterized by a single `DType` that represents the element type of the inner
    /// lists.
    List(Arc<DType>, Nullability),

    /// A logical struct type.
    ///
    /// A `Struct` type is composed of an ordered list of fields, each with a corresponding name and
    /// `DType`. See [`StructFields`] for more information.
    Struct(StructFields, Nullability),

    /// A user-defined extension type.
    ///
    /// See [`ExtDType`] for more information.
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

    /// Check if `self` is a `ListDType`
    pub fn is_list(&self) -> bool {
        matches!(self, List(_, _))
    }

    /// Check if `self` is a primitive type
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
    pub fn as_decimal_opt(&self) -> Option<&DecimalDType> {
        match self {
            Decimal(decimal, _) => Some(decimal),
            _ => None,
        }
    }

    /// Get the `StructDType` if `self` is a `StructDType`, otherwise `None`
    pub fn as_struct_opt(&self) -> Option<&StructFields> {
        match self {
            Struct(s, _) => Some(s),
            _ => None,
        }
    }

    /// Get the inner dtype if `self` is a `ListDType`, otherwise `None`
    pub fn as_list_element_opt(&self) -> Option<&Arc<DType>> {
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
    use crate::field_names::FieldNames;

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

    #[test]
    fn test_field_names_equality() {
        let field_names = FieldNames::from(["field1", "field2", "field3"]);

        // FieldNames == &FieldNames
        let field_names_ref = &field_names;
        assert_eq!(field_names, field_names_ref);

        // FieldNames == &[&str]
        let str_slice = &["field1", "field2", "field3"][..];
        assert_eq!(field_names, str_slice);

        // &FieldNames == &[&str]
        assert_eq!(&field_names, str_slice);

        // FieldNames == [&str; N] (array)
        assert_eq!(field_names, ["field1", "field2", "field3"]);

        // &FieldNames == [&str; N] (array)
        assert_eq!(&field_names, ["field1", "field2", "field3"]);

        // FieldNames == &[FieldName]
        let field_name_vec: Vec<FieldName> =
            vec!["field1".into(), "field2".into(), "field3".into()];
        let field_name_slice = field_name_vec.as_slice();
        assert_eq!(field_names, field_name_slice);

        // &FieldNames == &[FieldName]
        assert_eq!(&field_names, field_name_slice);

        // Test inequality cases
        assert_ne!(field_names, &["field1", "field2"][..]);
        assert_ne!(field_names, ["different", "fields", "here"]);
        assert_ne!(field_names, &["field1", "field2", "field3", "extra"][..]);
    }

    #[test]
    fn test_field_names_display() {
        let names = FieldNames::from(["a", "b", "c"]);
        let f = format!("{names}");

        assert_eq!(f, r#"["a", "b", "c"]"#);
    }
}
