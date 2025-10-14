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

    /// A logical fixed-size list type.
    ///
    /// This is parameterized by a `DType` that represents the element type of the inner lists, as
    /// well as a `u32` size that determines the fixed length of each `FixedSizeList` scalar.
    FixedSizeList(Arc<DType>, u32, Nullability),

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

/// This trait is implemented by native Rust types that can be converted
/// to and from Vortex scalar values.
/// e.g. `&str` -> `DType::Utf8`
///      `bool` -> `DType::Bool`
///
/// The dtype is the one closet matching the domain of the rust type
/// e.g. `Option<T>` -> Nullable DType.
pub trait NativeDType {
    /// Returns the Vortex data type for this scalar type.
    fn dtype() -> DType;
}

#[cfg(not(target_arch = "wasm32"))]
const_assert_eq!(size_of::<DType>(), 16);

#[cfg(target_arch = "wasm32")]
const_assert_eq!(size_of::<DType>(), 12);

impl DType {
    /// The default `DType` for bytes.
    pub const BYTES: Self = Primitive(PType::U8, Nullability::NonNullable);

    /// Get the nullability of the `DType`.
    #[inline]
    pub fn nullability(&self) -> Nullability {
        self.is_nullable().into()
    }

    /// Check if the `DType` is [`Nullability::Nullable`].
    #[inline]
    pub fn is_nullable(&self) -> bool {
        match self {
            Null => true,
            Extension(ext_dtype) => ext_dtype.storage_dtype().is_nullable(),
            Bool(null)
            | Primitive(_, null)
            | Decimal(_, null)
            | Utf8(null)
            | Binary(null)
            | Struct(_, null)
            | List(_, null)
            | FixedSizeList(_, _, null) => matches!(null, Nullability::Nullable),
        }
    }

    /// Get a new `DType` with [`Nullability::NonNullable`] (but otherwise the same as `self`)
    pub fn as_nonnullable(&self) -> Self {
        self.with_nullability(Nullability::NonNullable)
    }

    /// Get a new `DType` with [`Nullability::Nullable`] (but otherwise the same as `self`)
    pub fn as_nullable(&self) -> Self {
        self.with_nullability(Nullability::Nullable)
    }

    /// Get a new DType with the given nullability (but otherwise the same as `self`)
    pub fn with_nullability(&self, nullability: Nullability) -> Self {
        match self {
            Null => Null,
            Bool(_) => Bool(nullability),
            Primitive(pdt, _) => Primitive(*pdt, nullability),
            Decimal(ddt, _) => Decimal(*ddt, nullability),
            Utf8(_) => Utf8(nullability),
            Binary(_) => Binary(nullability),
            Struct(sf, _) => Struct(sf.clone(), nullability),
            List(edt, _) => List(edt.clone(), nullability),
            FixedSizeList(edt, size, _) => FixedSizeList(edt.clone(), *size, nullability),
            Extension(ext) => Extension(Arc::new(ext.with_nullability(nullability))),
        }
    }

    /// Union the nullability of this `DType` with the other nullability, returning a new `DType`.
    pub fn union_nullability(&self, other: Nullability) -> Self {
        let nullability = self.nullability() | other;
        self.with_nullability(nullability)
    }

    /// Check if `self` and `other` are equal, ignoring nullability.
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        match (self, other) {
            (Null, Null) => true,
            (Bool(_), Bool(_)) => true,
            (Primitive(lhs_ptype, _), Primitive(rhs_ptype, _)) => lhs_ptype == rhs_ptype,
            (Decimal(lhs, _), Decimal(rhs, _)) => lhs == rhs,
            (Utf8(_), Utf8(_)) => true,
            (Binary(_), Binary(_)) => true,
            (List(lhs_dtype, _), List(rhs_dtype, _)) => lhs_dtype.eq_ignore_nullability(rhs_dtype),
            (FixedSizeList(lhs_dtype, lhs_size, _), FixedSizeList(rhs_dtype, rhs_size, _)) => {
                lhs_size == rhs_size && lhs_dtype.eq_ignore_nullability(rhs_dtype)
            }
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

    /// Returns `true` if `self` is a subset type of `other, otherwise `false`.
    ///
    /// If `self` is nullable, this means that the other `DType` must also be nullable (since a
    /// nullable type represents more values than a non-nullable type) and equal.
    ///
    /// If `self` is non-nullable, then the other `DType` must be equal ignoring nullabillity.
    ///
    /// We implement this functionality as a complement to `is_superset_of`.
    pub fn eq_with_nullability_subset(&self, other: &Self) -> bool {
        if self.is_nullable() {
            self == other
        } else {
            self.eq_ignore_nullability(other)
        }
    }

    /// Returns `true` if `self` is a superset type of `other, otherwise `false`.
    ///
    /// If `self` is non-nullable, this means that the other `DType` must also be non-nullable
    /// (since a non-nullable type represents less values than a nullable type) and equal.
    ///
    /// If `self` is nullable, then the other `DType` must be equal ignoring nullabillity.
    ///
    /// This function is useful (in the `vortex-array` crate) for determining if an `Array` can
    /// extend a given `ArrayBuilder`: it can only extend it if the `DType` of the builder is a
    /// superset of the `Array`.
    pub fn eq_with_nullability_superset(&self, other: &Self) -> bool {
        if self.is_nullable() {
            self.eq_ignore_nullability(other)
        } else {
            self == other
        }
    }

    /// Check if `self` is a boolean
    pub fn is_boolean(&self) -> bool {
        matches!(self, Bool(_))
    }

    /// Check if `self` is a primitive type
    pub fn is_primitive(&self) -> bool {
        matches!(self, Primitive(_, _))
    }

    /// Returns this [`DType`]'s [`PType`] if it is a primitive type, otherwise panics.
    pub fn as_ptype(&self) -> PType {
        if let Primitive(ptype, _) = self {
            *ptype
        } else {
            vortex_panic!("DType is not a primitive type")
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

    /// Check if `self` is a [`DType::Decimal`].
    pub fn is_decimal(&self) -> bool {
        matches!(self, Decimal(..))
    }

    /// Check if `self` is a [`DType::Utf8`]
    pub fn is_utf8(&self) -> bool {
        matches!(self, Utf8(_))
    }

    /// Check if `self` is a [`DType::Binary`]
    pub fn is_binary(&self) -> bool {
        matches!(self, Binary(_))
    }

    /// Check if `self` is a [`DType::List`].
    pub fn is_list(&self) -> bool {
        matches!(self, List(_, _))
    }

    /// Check if `self` is a [`DType::FixedSizeList`],
    pub fn is_fixed_size_list(&self) -> bool {
        matches!(self, FixedSizeList(..))
    }

    /// Check if `self` is a [`DType::Struct`]
    pub fn is_struct(&self) -> bool {
        matches!(self, Struct(_, _))
    }

    /// Check if `self` is a [`DType::Extension`] type
    pub fn is_extension(&self) -> bool {
        matches!(self, Extension(_))
    }

    /// Check if `self` is a nested type, i.e. list, fixed size list, struct, or extension of a
    /// recursive type.
    pub fn is_nested(&self) -> bool {
        match self {
            List(..) | FixedSizeList(..) | Struct(..) => true,
            Extension(ext) => ext.storage_dtype().is_nested(),
            _ => false,
        }
    }

    /// Check returns the inner decimal type if the dtype is a [`DType::Decimal`].
    pub fn as_decimal_opt(&self) -> Option<&DecimalDType> {
        if let Decimal(decimal, _) = self {
            Some(decimal)
        } else {
            None
        }
    }

    /// Get the inner element dtype if `self` is a [`DType::List`], otherwise returns `None`.
    ///
    /// Note that this does _not_ return `Some` if `self` is a [`DType::FixedSizeList`].
    pub fn as_list_element_opt(&self) -> Option<&Arc<DType>> {
        if let List(edt, _) = self {
            Some(edt)
        } else {
            None
        }
    }

    /// Get the inner element dtype if `self` is a [`DType::FixedSizeList`], otherwise returns
    /// `None`.
    ///
    /// Note that this does _not_ return `Some` if `self` is a [`DType::List`].
    pub fn as_fixed_size_list_element_opt(&self) -> Option<&Arc<DType>> {
        if let FixedSizeList(edt, ..) = self {
            Some(edt)
        } else {
            None
        }
    }

    /// Get the inner element dtype if `self` is **either** a [`DType::List`] or a
    /// [`DType::FixedSizeList`], otherwise returns `None`
    pub fn as_any_size_list_element_opt(&self) -> Option<&Arc<DType>> {
        if let FixedSizeList(edt, ..) = self {
            Some(edt)
        } else if let List(edt, ..) = self {
            Some(edt)
        } else {
            None
        }
    }

    /// Get the `StructDType` if `self` is a `StructDType`, otherwise `None`
    pub fn as_struct_fields_opt(&self) -> Option<&StructFields> {
        if let Struct(f, _) = self {
            Some(f)
        } else {
            None
        }
    }

    /// Convenience method for creating a [`DType::List`].
    pub fn list(dtype: impl Into<DType>, nullability: Nullability) -> Self {
        List(Arc::new(dtype.into()), nullability)
    }

    /// Convenience method for creating a [`DType::Struct`].
    pub fn struct_<I: IntoIterator<Item = (impl Into<FieldName>, impl Into<FieldDType>)>>(
        iter: I,
        nullability: Nullability,
    ) -> Self {
        Struct(StructFields::from_iter(iter), nullability)
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Null => write!(f, "null"),
            Bool(null) => write!(f, "bool{null}"),
            Primitive(pdt, null) => write!(f, "{pdt}{null}"),
            Decimal(ddt, null) => write!(f, "{ddt}{null}"),
            Utf8(null) => write!(f, "utf8{null}"),
            Binary(null) => write!(f, "binary{null}"),
            Struct(sf, null) => write!(
                f,
                "{{{}}}{null}",
                sf.names()
                    .iter()
                    .zip(sf.fields())
                    .map(|(field_null, dt)| format!("{field_null}={dt}"))
                    .join(", "),
            ),
            List(edt, null) => write!(f, "list({edt}){null}"),
            FixedSizeList(edt, size, null) => write!(f, "fixed_size_list({edt})[{size}]{null}"),
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
