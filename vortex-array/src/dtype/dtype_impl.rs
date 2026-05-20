// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use DType::*;
use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use super::DType;
use crate::dtype::FieldDType;
use crate::dtype::FieldName;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::decimal::DecimalDType;
use crate::dtype::decimal::DecimalType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::nullability::Nullability;

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

/// Assert that the size of DType is 16 bytes.
#[cfg(not(target_arch = "wasm32"))]
const _: [(); size_of::<DType>()] = [(); 24]; // FIXME(ngates): should we keep this at 16?

/// Assert that the size of DType is 12 bytes on wasm32.
#[cfg(target_arch = "wasm32")]
const _: [(); size_of::<DType>()] = [(); 12];

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
            Bool(null)
            | Primitive(_, null)
            | Decimal(_, null)
            | Utf8(null)
            | Binary(null)
            | List(_, null)
            | FixedSizeList(_, _, null)
            | Struct(_, null)
            | Union(null)
            | Variant(null) => matches!(null, Nullability::Nullable),
            Extension(ext_dtype) => ext_dtype.storage_dtype().is_nullable(),
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
            List(edt, _) => List(Arc::clone(edt), nullability),
            FixedSizeList(edt, size, _) => FixedSizeList(Arc::clone(edt), *size, nullability),
            Struct(sf, _) => Struct(sf.clone(), nullability),
            Union(_) => Union(nullability),
            Variant(_) => Variant(nullability),
            Extension(ext) => Extension(ext.with_nullability(nullability)),
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
            (Union(_), Union(_)) => true,
            (Variant(_), Variant(_)) => true,
            (Extension(lhs_extdtype), Extension(rhs_extdtype)) => {
                lhs_extdtype.eq_ignore_nullability(rhs_extdtype)
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
            vortex_panic!("DType {self} is not a primitive type")
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

    /// Check if `self` is numeric.
    pub fn is_numeric(&self) -> bool {
        matches!(self, Primitive(..) | Decimal(..))
    }

    /// Check if `self` is a temporal extension type.
    pub fn is_temporal(&self) -> bool {
        match self {
            Extension(ext) => {
                use crate::dtype::extension::Matcher;
                use crate::extension::datetime::AnyTemporal;
                AnyTemporal::matches(ext)
            }
            _ => false,
        }
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

    /// Check if `self` is a [`DType::Union`] type.
    pub fn is_union(&self) -> bool {
        matches!(self, Union(..))
    }

    /// Check if `self` is a [`DType::Variant`] type
    pub fn is_variant(&self) -> bool {
        matches!(self, Variant(_))
    }

    /// Check if `self` is a [`DType::Extension`] type
    pub fn is_extension(&self) -> bool {
        matches!(self, Extension(_))
    }

    /// Check if `self` is a nested type, i.e. list, fixed size list, struct, or extension of a
    /// recursive type.
    pub fn is_nested(&self) -> bool {
        match self {
            List(..) | FixedSizeList(..) | Struct(..) | Union(..) | Variant(..) => true,
            Extension(ext) => ext.storage_dtype().is_nested(),
            _ => false,
        }
    }

    /// Returns the number of bytes occupied by a single scalar of this fixed-width type.
    ///
    /// For non-fixed-width types, return None.
    ///
    /// [`Bool`] is defined as 1 even though a Vortex array may pack Booleans to one bit per element.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            Null => Some(0),
            Bool(_) => Some(1),
            Primitive(ptype, _) => Some(ptype.byte_width()),
            Decimal(decimal, _) => {
                Some(DecimalType::smallest_decimal_value_type(decimal).byte_width())
            }
            Utf8(_) | Binary(_) | List(..) => None,
            FixedSizeList(elem_dtype, list_size, _) => {
                elem_dtype.element_size().map(|s| s * *list_size as usize)
            }
            Struct(fields, ..) => {
                let mut sum = 0_usize;
                for f in fields.fields() {
                    let element_size = f.element_size()?;
                    sum = sum
                        .checked_add(element_size)
                        .vortex_expect("sum of field sizes is bigger than usize");
                }
                Some(sum)
            }
            Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            Variant(_) => None,
            Extension(ext) => ext.storage_dtype().element_size(),
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

    /// Owned version of [Self::as_decimal_opt].
    pub fn into_decimal_opt(self) -> Option<DecimalDType> {
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

    /// Owned version of [Self::as_list_element_opt].
    pub fn into_list_element_opt(self) -> Option<Arc<DType>> {
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

    /// Owned version of [Self::as_fixed_size_list_element_opt].
    pub fn into_fixed_size_list_element_opt(self) -> Option<Arc<DType>> {
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

    /// Owned version of [Self::as_any_size_list_element_opt].
    pub fn into_any_size_list_element_opt(self) -> Option<Arc<DType>> {
        if let FixedSizeList(edt, ..) = self {
            Some(edt)
        } else if let List(edt, ..) = self {
            Some(edt)
        } else {
            None
        }
    }

    /// Returns the [`StructFields`] from a struct [`DType`].
    ///
    /// # Panics
    ///
    /// If the [`DType`] is not a struct.
    pub fn as_struct_fields(&self) -> &StructFields {
        if let Struct(f, _) = self {
            return f;
        }
        vortex_panic!("DType is not a Struct")
    }

    /// Owned version of [Self::as_struct_fields].
    pub fn into_struct_fields(self) -> StructFields {
        if let Struct(f, _) = self {
            return f;
        }
        vortex_panic!("DType is not a Struct")
    }

    /// Get the `StructDType` if `self` is a `StructDType`, otherwise `None`
    pub fn as_struct_fields_opt(&self) -> Option<&StructFields> {
        if let Struct(f, _) = self {
            Some(f)
        } else {
            None
        }
    }

    /// Owned version of [Self::as_struct_fields_opt].
    pub fn into_struct_fields_opt(self) -> Option<StructFields> {
        if let Struct(f, _) = self {
            Some(f)
        } else {
            None
        }
    }

    /// Downcast a `DType` to an `ExtDType`
    pub fn as_extension(&self) -> &ExtDTypeRef {
        let Extension(ext) = self else {
            vortex_panic!("DType is not an Extension")
        };
        ext
    }

    /// Get the `ExtDTypeRef` if `self` is an `Extension` type, otherwise `None`
    pub fn as_extension_opt(&self) -> Option<&ExtDTypeRef> {
        if let Extension(ext) = self {
            Some(ext)
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
            List(edt, null) => write!(f, "list({edt}){null}"),
            FixedSizeList(edt, size, null) => write!(f, "fixed_size_list({edt})[{size}]{null}"),
            Struct(sf, null) => write!(
                f,
                "{{{}}}{null}",
                sf.names()
                    .iter()
                    .zip(sf.fields())
                    .map(|(field_null, dt)| format!("{field_null}={dt}"))
                    .join(", "),
            ),
            Union(null) => write!(f, "union(){null}"),
            Variant(null) => write!(f, "variant{null}"),
            Extension(ext) => write!(f, "{}", ext),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::decimal::DecimalDType;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;

    #[test]
    fn test_ext_dtype_eq_ignore_nullability() {
        let d1 = DType::Extension(Time::new(TimeUnit::Seconds, Nullable).erased());
        let d2 = DType::Extension(Time::new(TimeUnit::Seconds, NonNullable).erased());
        assert!(d1.eq_ignore_nullability(&d2));

        let t1 = DType::Extension(
            Timestamp::new_with_tz(TimeUnit::Seconds, Some("UTC".into()), Nullable).erased(),
        );
        let t2 = DType::Extension(
            Timestamp::new_with_tz(TimeUnit::Seconds, Some("ET".into()), Nullable).erased(),
        );
        assert!(!t1.eq_ignore_nullability(&t2));
    }

    #[test]
    fn is_numeric() {
        assert!(DType::Primitive(PType::I32, NonNullable).is_numeric());
        assert!(DType::Primitive(PType::F64, NonNullable).is_numeric());
        assert!(DType::Decimal(DecimalDType::new(10, 2), NonNullable).is_numeric());
        assert!(!DType::Bool(NonNullable).is_numeric());
        assert!(!DType::Utf8(NonNullable).is_numeric());
        assert!(!DType::Null.is_numeric());
    }

    #[test]
    fn element_size_null() {
        assert_eq!(DType::Null.element_size(), Some(0));
    }

    #[test]
    fn element_size_bool() {
        assert_eq!(DType::Bool(NonNullable).element_size(), Some(1));
    }

    #[test]
    fn element_size_primitives() {
        assert_eq!(
            DType::Primitive(PType::U8, NonNullable).element_size(),
            Some(1)
        );
        assert_eq!(
            DType::Primitive(PType::I32, NonNullable).element_size(),
            Some(4)
        );
        assert_eq!(
            DType::Primitive(PType::F64, NonNullable).element_size(),
            Some(8)
        );
    }

    #[test]
    fn element_size_decimal() {
        let decimal = DecimalDType::new(10, 2);
        // precision 10 -> DecimalType::I64 -> 8 bytes
        assert_eq!(DType::Decimal(decimal, NonNullable).element_size(), Some(8));
    }

    #[test]
    fn element_size_fixed_size_list() {
        let elem = Arc::new(DType::Primitive(PType::F64, NonNullable));
        assert_eq!(
            DType::FixedSizeList(Arc::clone(&elem), 1000, NonNullable).element_size(),
            Some(8000)
        );

        assert_eq!(
            DType::FixedSizeList(
                Arc::new(DType::FixedSizeList(elem, 20, NonNullable)),
                1000,
                NonNullable
            )
            .element_size(),
            Some(160_000)
        );
    }

    #[test]
    fn element_size_nested_fixed_size_list() {
        let inner = Arc::new(DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F64, NonNullable)),
            10,
            NonNullable,
        ));
        assert_eq!(
            DType::FixedSizeList(inner, 100, NonNullable).element_size(),
            Some(8000)
        );
    }

    #[test]
    fn element_size_extension() {
        assert_eq!(
            DType::Extension(Date::new(TimeUnit::Days, NonNullable).erased()).element_size(),
            Some(4)
        );
    }

    #[test]
    fn element_size_variable_width() {
        assert_eq!(DType::Utf8(NonNullable).element_size(), None);
        assert_eq!(DType::Binary(NonNullable).element_size(), None);
        assert_eq!(
            DType::List(
                Arc::new(DType::Primitive(PType::I32, NonNullable)),
                NonNullable
            )
            .element_size(),
            None
        );
    }
}
