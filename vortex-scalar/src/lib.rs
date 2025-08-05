// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar values and types for the Vortex system.
//!
//! This crate provides scalar types and values that can be used to represent individual
//! data elements in the Vortex array system. Scalars are composed of a logical data type
//! ([`DType`]) and a value ([`ScalarValue`]).

#![deny(missing_docs)]

use std::cmp::Ordering;
use std::hash::Hash;
use std::sync::Arc;

pub use scalar_type::ScalarType;
use vortex_buffer::{Buffer, BufferString, ByteBuffer};
use vortex_dtype::half::f16;
use vortex_dtype::{DECIMAL128_MAX_PRECISION, DType, Nullability, PType};
#[cfg(feature = "arbitrary")]
pub mod arbitrary;
mod arrow;
mod bigint;
mod binary;
mod bool;
mod decimal;
mod display;
mod extension;
mod list;
mod null;
mod primitive;
mod proto;
mod pvalue;
mod scalar_type;
mod scalar_value;
mod struct_;
mod utf8;

pub use bigint::*;
pub use binary::*;
pub use bool::*;
pub use decimal::*;
pub use extension::*;
pub use list::*;
pub use primitive::*;
pub use pvalue::*;
pub use scalar_value::*;
pub use struct_::*;
pub use utf8::*;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

/// A single logical item, composed of both a [`ScalarValue`] and a logical [`DType`].
///
/// A [`ScalarValue`] is opaque, and should be accessed via one of the type-specific scalar wrappers
/// for example [`BoolScalar`], [`PrimitiveScalar`], etc.
///
/// Note that [`PartialOrd`] is implemented only for an exact match of the scalar's dtype,
/// including nullability. When the DType does match, ordering is nulls first (lowest), then the
/// natural ordering of the scalar value.
#[derive(Debug, Clone)]
pub struct Scalar {
    dtype: DType,
    value: ScalarValue,
}

impl Scalar {
    /// Creates a new scalar with the given data type and value.
    ///
    /// This function performs type coercion when necessary to ensure the value matches
    /// the expected data type. This is particularly important for backwards compatibility
    /// with serialized scalars where floating point values may have been stored as integers.
    ///
    /// # Panics
    ///
    /// Panics if the value cannot be coerced to the expected data type.
    pub fn new(dtype: DType, value: ScalarValue) -> Self {
        let value = Self::coerce_value(&dtype, value).vortex_expect("Failed to coerce value");
        Self { dtype, value }
    }

    /// Coerces a scalar value to match the expected data type.
    ///
    /// This handles cases where:
    /// - Floating point values were serialized as their bit representation
    /// - Struct fields need recursive coercion
    /// - List elements need recursive coercion
    fn coerce_value(dtype: &DType, value: ScalarValue) -> VortexResult<ScalarValue> {
        match (dtype, &value.0) {
            // Handle primitive type coercion
            (DType::Primitive(ptype, _), InnerScalarValue::Primitive(pvalue)) => {
                match (ptype, pvalue) {
                    // F16 coercion from integer types (backwards compatibility)
                    (PType::F16, PValue::U64(v)) if *v <= u16::MAX as u64 => {
                        Ok(ScalarValue(InnerScalarValue::Primitive(PValue::F16(
                            f16::from_bits(u16::try_from(*v).map_err(|_| {
                                vortex_err!(
                                    "bit representation of f16 has more than 16 bits: {}",
                                    v
                                )
                            })?),
                        ))))
                    }
                    // No coercion needed
                    _ => Ok(value),
                }
            }
            // Handle struct coercion - recursively coerce fields
            (DType::Struct(struct_fields, _), InnerScalarValue::List(field_values)) => {
                let coerced_fields: Result<Vec<ScalarValue>, _> = struct_fields
                    .fields()
                    .zip(field_values.iter())
                    .map(|(field_dtype, field_value)| {
                        Self::coerce_value(&field_dtype, field_value.clone())
                    })
                    .collect();
                Ok(ScalarValue(InnerScalarValue::List(coerced_fields?.into())))
            }
            // Handle list coercion - recursively coerce elements
            (DType::List(elem_dtype, _), InnerScalarValue::List(elements)) => {
                let coerced_elements: Result<Vec<ScalarValue>, _> = elements
                    .iter()
                    .map(|elem| Self::coerce_value(elem_dtype, elem.clone()))
                    .collect();
                Ok(ScalarValue(InnerScalarValue::List(
                    coerced_elements?.into(),
                )))
            }
            // Handle extension type coercion - recursively coerce the storage scalar
            (DType::Extension(ext_dtype), _) => {
                Self::coerce_value(ext_dtype.storage_dtype(), value)
            }
            // No coercion needed for other types
            _ => Ok(value),
        }
    }

    /// Returns a reference to the scalar's data type.
    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns a reference to the scalar's underlying value.
    #[inline]
    pub fn value(&self) -> &ScalarValue {
        &self.value
    }

    /// Consumes the scalar and returns its data type and value as a tuple.
    #[inline]
    pub fn into_parts(self) -> (DType, ScalarValue) {
        (self.dtype, self.value)
    }

    /// Consumes the scalar and returns its underlying value.
    #[inline]
    pub fn into_value(self) -> ScalarValue {
        self.value
    }

    /// Returns true if the scalar is not null.
    pub fn is_valid(&self) -> bool {
        !self.value.is_null()
    }

    /// Returns true if the scalar is null.
    pub fn is_null(&self) -> bool {
        self.value.is_null()
    }

    /// Creates a null scalar with the given nullable data type.
    ///
    /// # Panics
    ///
    /// Panics if the data type is not nullable.
    pub fn null(dtype: DType) -> Self {
        assert!(
            dtype.is_nullable(),
            "Creating null scalar for non-nullable DType {dtype}"
        );
        Self {
            dtype,
            value: ScalarValue(InnerScalarValue::Null),
        }
    }

    /// Creates a null scalar for the given scalar type.
    ///
    /// The resulting scalar will have a nullable version of the type's data type.
    pub fn null_typed<T: ScalarType>() -> Self {
        Self {
            dtype: T::dtype().as_nullable(),
            value: ScalarValue(InnerScalarValue::Null),
        }
    }

    /// Casts the scalar to the target data type.
    ///
    /// Returns an error if the cast is not supported or if the value cannot be represented
    /// in the target type.
    pub fn cast(&self, target: &DType) -> VortexResult<Self> {
        if let DType::Extension(ext_dtype) = target {
            let storage_scalar = self.cast_to_non_extension(ext_dtype.storage_dtype())?;
            Ok(Scalar::extension(ext_dtype.clone(), storage_scalar))
        } else {
            self.cast_to_non_extension(target)
        }
    }

    fn cast_to_non_extension(&self, target: &DType) -> VortexResult<Self> {
        assert!(!matches!(target, DType::Extension(..)));
        if self.is_null() {
            if target.is_nullable() {
                return Ok(Scalar::new(target.clone(), self.value.clone()));
            } else {
                vortex_bail!("Can't cast null scalar to non-nullable type {}", target)
            }
        }

        if self.dtype().eq_ignore_nullability(target) {
            return Ok(Scalar::new(target.clone(), self.value.clone()));
        }

        match &self.dtype {
            DType::Null => unreachable!(), // handled by if is_null case
            DType::Bool(_) => self.as_bool().cast(target),
            DType::Primitive(..) => self.as_primitive().cast(target),
            DType::Decimal(..) => todo!("(aduffy): implement DecimalScalar casting"),
            DType::Utf8(_) => self.as_utf8().cast(target),
            DType::Binary(_) => self.as_binary().cast(target),
            DType::Struct(..) => self.as_struct().cast(target),
            DType::List(..) => self.as_list().cast(target),
            DType::Extension(..) => self.as_extension().cast(target),
        }
    }

    /// Converts the scalar to have a nullable version of its data type.
    pub fn into_nullable(self) -> Self {
        Self {
            dtype: self.dtype.as_nullable(),
            value: self.value,
        }
    }

    /// Returns the size of the scalar in bytes, uncompressed.
    pub fn nbytes(&self) -> usize {
        match self.dtype() {
            DType::Null => 0,
            DType::Bool(_) => 1,
            DType::Primitive(ptype, _) => ptype.byte_width(),
            DType::Decimal(dt, _) => {
                if dt.precision() >= DECIMAL128_MAX_PRECISION {
                    size_of::<i128>()
                } else {
                    size_of::<i256>()
                }
            }
            DType::Binary(_) | DType::Utf8(_) => self
                .value()
                .as_buffer()
                .ok()
                .flatten()
                .map_or(0, |s| s.len()),
            DType::Struct(_dtype, _) => self
                .as_struct()
                .fields()
                .map(|fields| fields.into_iter().map(|f| f.nbytes()).sum::<usize>())
                .unwrap_or_default(),
            DType::List(_dtype, _) => self
                .as_list()
                .elements()
                .map(|fields| fields.into_iter().map(|f| f.nbytes()).sum::<usize>())
                .unwrap_or_default(),
            DType::Extension(_ext_dtype) => self.as_extension().storage().nbytes(),
        }
    }

    /// Creates a "default" scalar value for the given data type.
    ///
    /// For nullable types, returns null. For non-nullable types, returns
    /// an appropriate zero/empty value.
    pub fn default_value(dtype: DType) -> Self {
        if dtype.is_nullable() {
            return Self::null(dtype);
        }

        match dtype {
            DType::Null => Self::null(dtype),
            DType::Bool(nullability) => Self::bool(false, nullability),
            DType::Primitive(pt, nullability) => {
                Self::primitive_value(PValue::zero(pt), pt, nullability)
            }
            DType::Decimal(dt, nullability) => {
                Self::decimal(DecimalValue::from(0), dt, nullability)
            }
            DType::Utf8(nullability) => Self::utf8("", nullability),
            DType::Binary(nullability) => Self::binary(Buffer::empty(), nullability),
            DType::Struct(sf, nullability) => {
                let fields: Vec<_> = sf.fields().map(Scalar::default_value).collect();
                Self::struct_(DType::Struct(sf, nullability), fields)
            }
            DType::List(dt, nullability) => Self::list(dt, vec![], nullability),
            DType::Extension(dt) => {
                let scalar = Self::default_value(dt.storage_dtype().clone());
                Self::extension(dt, scalar)
            }
        }
    }
}

impl Scalar {
    /// Returns a view of the scalar as a boolean scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not a boolean type.
    pub fn as_bool(&self) -> BoolScalar<'_> {
        BoolScalar::try_from(self).vortex_expect("Failed to convert scalar to bool")
    }

    /// Returns a view of the scalar as a boolean scalar if it has a boolean type.
    pub fn as_bool_opt(&self) -> Option<BoolScalar<'_>> {
        matches!(self.dtype, DType::Bool(..)).then(|| self.as_bool())
    }

    /// Returns a view of the scalar as a primitive scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not a primitive type.
    pub fn as_primitive(&self) -> PrimitiveScalar<'_> {
        PrimitiveScalar::try_from(self).vortex_expect("Failed to convert scalar to primitive")
    }

    /// Returns a view of the scalar as a primitive scalar if it has a primitive type.
    pub fn as_primitive_opt(&self) -> Option<PrimitiveScalar<'_>> {
        matches!(self.dtype, DType::Primitive(..)).then(|| self.as_primitive())
    }

    /// Returns a view of the scalar as a decimal scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not a decimal type.
    pub fn as_decimal(&self) -> DecimalScalar<'_> {
        DecimalScalar::try_from(self).vortex_expect("Failed to convert scalar to decimal")
    }

    /// Returns a view of the scalar as a decimal scalar if it has a decimal type.
    pub fn as_decimal_opt(&self) -> Option<DecimalScalar<'_>> {
        matches!(self.dtype, DType::Decimal(..)).then(|| self.as_decimal())
    }

    /// Returns a view of the scalar as a UTF-8 string scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not a UTF-8 type.
    pub fn as_utf8(&self) -> Utf8Scalar<'_> {
        Utf8Scalar::try_from(self).vortex_expect("Failed to convert scalar to utf8")
    }

    /// Returns a view of the scalar as a UTF-8 string scalar if it has a UTF-8 type.
    pub fn as_utf8_opt(&self) -> Option<Utf8Scalar<'_>> {
        matches!(self.dtype, DType::Utf8(..)).then(|| self.as_utf8())
    }

    /// Returns a view of the scalar as a binary scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not a binary type.
    pub fn as_binary(&self) -> BinaryScalar<'_> {
        BinaryScalar::try_from(self).vortex_expect("Failed to convert scalar to binary")
    }

    /// Returns a view of the scalar as a binary scalar if it has a binary type.
    pub fn as_binary_opt(&self) -> Option<BinaryScalar<'_>> {
        matches!(self.dtype, DType::Binary(..)).then(|| self.as_binary())
    }

    /// Returns a view of the scalar as a struct scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not a struct type.
    pub fn as_struct(&self) -> StructScalar<'_> {
        StructScalar::try_from(self).vortex_expect("Failed to convert scalar to struct")
    }

    /// Returns a view of the scalar as a struct scalar if it has a struct type.
    pub fn as_struct_opt(&self) -> Option<StructScalar<'_>> {
        matches!(self.dtype, DType::Struct(..)).then(|| self.as_struct())
    }

    /// Returns a view of the scalar as a list scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not a list type.
    pub fn as_list(&self) -> ListScalar<'_> {
        ListScalar::try_from(self).vortex_expect("Failed to convert scalar to list")
    }

    /// Returns a view of the scalar as a list scalar if it has a list type.
    pub fn as_list_opt(&self) -> Option<ListScalar<'_>> {
        matches!(self.dtype, DType::List(..)).then(|| self.as_list())
    }

    /// Returns a view of the scalar as an extension scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not an extension type.
    pub fn as_extension(&self) -> ExtScalar<'_> {
        ExtScalar::try_from(self).vortex_expect("Failed to convert scalar to extension")
    }

    /// Returns a view of the scalar as an extension scalar if it has an extension type.
    pub fn as_extension_opt(&self) -> Option<ExtScalar<'_>> {
        matches!(self.dtype, DType::Extension(..)).then(|| self.as_extension())
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        if !self.dtype.eq_ignore_nullability(&other.dtype) {
            return false;
        }

        match self.dtype() {
            DType::Null => true,
            DType::Bool(_) => self.as_bool() == other.as_bool(),
            DType::Primitive(..) => self.as_primitive() == other.as_primitive(),
            DType::Decimal(..) => self.as_decimal() == other.as_decimal(),
            DType::Utf8(_) => self.as_utf8() == other.as_utf8(),
            DType::Binary(_) => self.as_binary() == other.as_binary(),
            DType::Struct(..) => self.as_struct() == other.as_struct(),
            DType::List(..) => self.as_list() == other.as_list(),
            DType::Extension(_) => self.as_extension() == other.as_extension(),
        }
    }
}

impl Eq for Scalar {}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self.dtype().eq_ignore_nullability(other.dtype()) {
            return None;
        }
        match self.dtype() {
            DType::Null => Some(Ordering::Equal),
            DType::Bool(_) => self.as_bool().partial_cmp(&other.as_bool()),
            DType::Primitive(..) => self.as_primitive().partial_cmp(&other.as_primitive()),
            DType::Decimal(..) => self.as_decimal().partial_cmp(&other.as_decimal()),
            DType::Utf8(_) => self.as_utf8().partial_cmp(&other.as_utf8()),
            DType::Binary(_) => self.as_binary().partial_cmp(&other.as_binary()),
            DType::Struct(..) => self.as_struct().partial_cmp(&other.as_struct()),
            DType::List(..) => self.as_list().partial_cmp(&other.as_list()),
            DType::Extension(_) => self.as_extension().partial_cmp(&other.as_extension()),
        }
    }
}

impl Hash for Scalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.dtype() {
            DType::Null => self.dtype().hash(state), // Hash the dtype instead of the value
            DType::Bool(_) => self.as_bool().hash(state),
            DType::Primitive(..) => self.as_primitive().hash(state),
            DType::Decimal(..) => self.as_decimal().hash(state),
            DType::Utf8(_) => self.as_utf8().hash(state),
            DType::Binary(_) => self.as_binary().hash(state),
            DType::Struct(..) => self.as_struct().hash(state),
            DType::List(..) => self.as_list().hash(state),
            DType::Extension(_) => self.as_extension().hash(state),
        }
    }
}

impl AsRef<Self> for Scalar {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<T> From<Option<T>> for Scalar
where
    T: ScalarType,
    Scalar: From<T>,
{
    fn from(value: Option<T>) -> Self {
        value
            .map(Scalar::from)
            .map(|x| x.into_nullable())
            .unwrap_or_else(|| Scalar {
                dtype: T::dtype().as_nullable(),
                value: ScalarValue(InnerScalarValue::Null),
            })
    }
}

impl From<PrimitiveScalar<'_>> for Scalar {
    fn from(pscalar: PrimitiveScalar<'_>) -> Self {
        let dtype = pscalar.dtype().clone();
        let value = pscalar
            .pvalue()
            .map(|pvalue| ScalarValue(InnerScalarValue::Primitive(pvalue)))
            .unwrap_or_else(|| ScalarValue(InnerScalarValue::Null));
        Self::new(dtype, value)
    }
}

impl From<DecimalScalar<'_>> for Scalar {
    fn from(decimal_scalar: DecimalScalar<'_>) -> Self {
        let dtype = decimal_scalar.dtype().clone();
        let value = decimal_scalar
            .decimal_value()
            .map(|value| ScalarValue(InnerScalarValue::Decimal(value)))
            .unwrap_or_else(|| ScalarValue(InnerScalarValue::Null));
        Self::new(dtype, value)
    }
}

macro_rules! from_vec_for_scalar {
    ($T:ty) => {
        impl From<Vec<$T>> for Scalar {
            fn from(value: Vec<$T>) -> Self {
                Scalar {
                    dtype: DType::List(Arc::from(<$T>::dtype()), Nullability::NonNullable),
                    value: ScalarValue(InnerScalarValue::List(
                        value
                            .into_iter()
                            .map(Scalar::from)
                            .map(|s| s.into_value())
                            .collect::<Arc<[_]>>(),
                    )),
                }
            }
        }
    };
}

// no From<Vec<u8>> because it could either be a List or a Buffer
from_vec_for_scalar!(u16);
from_vec_for_scalar!(u32);
from_vec_for_scalar!(u64);
from_vec_for_scalar!(usize); // For usize only, we implicitly cast for better ergonomics.
from_vec_for_scalar!(i8);
from_vec_for_scalar!(i16);
from_vec_for_scalar!(i32);
from_vec_for_scalar!(i64);
from_vec_for_scalar!(f16);
from_vec_for_scalar!(f32);
from_vec_for_scalar!(f64);
from_vec_for_scalar!(String);
from_vec_for_scalar!(BufferString);
from_vec_for_scalar!(ByteBuffer);

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_dtype::half::f16;
    use vortex_dtype::{DType, ExtDType, ExtID, FieldDType, Nullability, PType, StructFields};

    use crate::{InnerScalarValue, PValue, Scalar, ScalarValue};

    #[rstest]
    fn null_can_cast_to_anything_nullable(
        #[values(
            DType::Null,
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("a"),
                Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
                None,
            ))),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("b"),
                Arc::from(DType::Utf8(Nullability::Nullable)),
                None,
            )))
        )]
        source_dtype: DType,
        #[values(
            DType::Null,
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("a"),
                Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
                None,
            ))),
            DType::Extension(Arc::from(ExtDType::new(
                ExtID::from("b"),
                Arc::from(DType::Utf8(Nullability::Nullable)),
                None,
            )))
        )]
        target_dtype: DType,
    ) {
        assert_eq!(
            Scalar::null(source_dtype)
                .cast(&target_dtype)
                .unwrap()
                .dtype(),
            &target_dtype
        );
    }

    #[test]
    fn list_casts() {
        let list = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            ScalarValue(InnerScalarValue::List(Arc::from([ScalarValue(
                InnerScalarValue::Primitive(PValue::U16(6)),
            )]))),
        );

        let target_u32 = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(list.cast(&target_u32).unwrap().dtype(), &target_u32);

        let target_u32_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        assert_eq!(
            list.cast(&target_u32_nonnull).unwrap().dtype(),
            &target_u32_nonnull
        );

        let target_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::Nullable)),
            Nullability::NonNullable,
        );
        assert_eq!(list.cast(&target_nonnull).unwrap().dtype(), &target_nonnull);

        let target_u8 = DType::List(
            Arc::from(DType::Primitive(PType::U8, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(list.cast(&target_u8).unwrap().dtype(), &target_u8);

        let list_with_null = Scalar::new(
            DType::List(
                Arc::from(DType::Primitive(PType::U16, Nullability::Nullable)),
                Nullability::Nullable,
            ),
            ScalarValue(InnerScalarValue::List(Arc::from([
                ScalarValue(InnerScalarValue::Primitive(PValue::U16(6))),
                ScalarValue(InnerScalarValue::Null),
            ]))),
        );
        let target_u8 = DType::List(
            Arc::from(DType::Primitive(PType::U8, Nullability::Nullable)),
            Nullability::Nullable,
        );
        assert_eq!(list_with_null.cast(&target_u8).unwrap().dtype(), &target_u8);

        let target_u32_nonnull = DType::List(
            Arc::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
            Nullability::Nullable,
        );
        assert!(list_with_null.cast(&target_u32_nonnull).is_err());
    }

    #[test]
    fn cast_to_from_extension_types() {
        let apples = ExtDType::new(
            ExtID::new(Arc::from("apples")),
            Arc::from(DType::Primitive(PType::U16, Nullability::NonNullable)),
            None,
        );
        let ext_dtype = DType::Extension(Arc::from(apples.clone()));
        let ext_scalar = Scalar::new(ext_dtype.clone(), ScalarValue(InnerScalarValue::Bool(true)));
        let storage_scalar = Scalar::new(
            DType::clone(apples.storage_dtype()),
            ScalarValue(InnerScalarValue::Primitive(PValue::U16(1000))),
        );

        // to self
        let expected_dtype = &ext_dtype;
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // to nullable self
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type
        let expected_dtype = apples.storage_dtype();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type, nullable
        let expected_dtype = &apples.storage_dtype().as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from storage type to extension
        let expected_dtype = &ext_dtype;
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from storage type to extension, nullable
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from *compatible* storage type to extension
        let storage_scalar_u64 = Scalar::new(
            DType::clone(apples.storage_dtype()),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(1000))),
        );
        let expected_dtype = &ext_dtype;
        let actual = storage_scalar_u64.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from *incompatible* storage type to extension
        let apples_u8 = ExtDType::new(
            ExtID::new(Arc::from("apples")),
            Arc::from(DType::Primitive(PType::U8, Nullability::NonNullable)),
            None,
        );
        let expected_dtype = &DType::Extension(Arc::from(apples_u8));
        let result = storage_scalar.cast(expected_dtype);
        assert!(
            result.as_ref().is_err_and(|err| {
                err
                    .to_string()
                    .contains("Can't cast u16 scalar 1000u16 to u8 (cause: Cannot read primitive value U16(1000) as u8")
            }),
            "{result:?}"
        );
    }

    #[test]
    fn default_value_for_complex_dtype() {
        let struct_dtype = DType::struct_(
            [
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                (
                    "b",
                    DType::list(
                        DType::Primitive(PType::I8, Nullability::Nullable),
                        Nullability::NonNullable,
                    ),
                ),
                ("c", DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );

        let scalar = Scalar::default_value(struct_dtype.clone());
        assert_eq!(scalar.dtype(), &struct_dtype);

        let scalar = scalar.as_struct();

        let a_field = scalar.field("a").unwrap();
        assert_eq!(a_field.as_primitive().pvalue().unwrap(), PValue::I32(0));

        let b_field = scalar.field("b").unwrap();
        assert!(b_field.as_list().is_empty());

        let c_field = scalar.field("c").unwrap();
        assert!(c_field.is_null());
    }

    #[test]
    fn test_f16_coercion_from_u64() {
        let f16_value = f16::from_f32(5.722046e-6);
        let u64_bits = f16_value.to_bits() as u64;

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::F16(v))) => {
                assert_eq!(*v, f16_value);
            }
            _ => panic!("Expected F16 value after coercion"),
        }
    }

    #[test]
    fn test_f16_no_coercion_from_u32() {
        let f16_value = f16::from_f32(0.42);
        let u32_bits = f16_value.to_bits() as u32;

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(u32_bits))),
        );

        // No coercion expected from u32
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(v))) => {
                assert_eq!(*v, u32_bits);
            }
            _ => panic!("Expected U32 value (no coercion)"),
        }
    }

    #[test]
    fn test_f16_no_coercion_from_u16() {
        let f16_value = f16::from_f32(1.5);
        let u16_bits = f16_value.to_bits();

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U16(u16_bits))),
        );

        // No coercion expected from u16
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U16(v))) => {
                assert_eq!(*v, u16_bits);
            }
            _ => panic!("Expected U16 value (no coercion)"),
        }
    }

    #[test]
    fn test_f32_no_coercion_from_u32() {
        let f32_value = std::f32::consts::PI;
        let u32_bits = f32_value.to_bits();

        let scalar = Scalar::new(
            DType::Primitive(PType::F32, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(u32_bits))),
        );

        // No coercion expected from u32
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(v))) => {
                assert_eq!(*v, u32_bits);
            }
            _ => panic!("Expected U32 value (no coercion)"),
        }
    }

    #[test]
    fn test_f64_no_coercion_from_u64() {
        let f64_value = std::f64::consts::E;
        let u64_bits = f64_value.to_bits();

        let scalar = Scalar::new(
            DType::Primitive(PType::F64, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        // No coercion expected from u64
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(v))) => {
                assert_eq!(*v, u64_bits);
            }
            _ => panic!("Expected U64 value (no coercion)"),
        }
    }

    #[test]
    fn test_struct_field_coercion() {
        let f16_value = f16::from_f32(0.42);
        let f32_value = std::f32::consts::PI;

        let struct_dtype = DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    FieldDType::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
                ),
                (
                    "b",
                    FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable)),
                ),
                (
                    "c",
                    FieldDType::from(DType::Primitive(PType::F32, Nullability::NonNullable)),
                ),
            ]),
            Nullability::NonNullable,
        );

        let field_values = vec![
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(42))),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value.to_bits() as u64,
            ))),
            ScalarValue(InnerScalarValue::Primitive(PValue::F32(f32_value))),
        ];

        let scalar = Scalar::new(
            struct_dtype,
            ScalarValue(InnerScalarValue::List(field_values.into())),
        );

        let struct_scalar = scalar.as_struct();
        let fields = struct_scalar.fields().unwrap();

        // Check first field (no coercion needed)
        match fields[0].value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(v))) => {
                assert_eq!(*v, 42);
            }
            _ => panic!("Expected U32 value for field 'a'"),
        }

        // Check second field (f16 coerced from u64)
        match fields[1].value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::F16(v))) => {
                assert_eq!(*v, f16_value);
            }
            _ => panic!("Expected F16 value for field 'b' after coercion"),
        }

        // Check third field (no coercion needed)
        match fields[2].value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::F32(v))) => {
                assert_eq!(*v, f32_value);
            }
            _ => panic!("Expected F32 value for field 'c'"),
        }
    }

    #[test]
    fn test_no_coercion_for_matching_types() {
        // Test that when types already match, no coercion happens
        let i32_value = 42i32;
        let scalar = Scalar::new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::I32(i32_value))),
        );

        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::I32(v))) => {
                assert_eq!(*v, i32_value);
            }
            _ => panic!("Expected I32 value"),
        }
    }

    #[test]
    fn test_list_element_coercion() {
        let f16_value1 = f16::from_f32(1.0);
        let f16_value2 = f16::from_f32(2.0);

        let list_dtype = DType::List(
            Arc::new(DType::Primitive(PType::F16, Nullability::NonNullable)),
            Nullability::NonNullable,
        );

        let elements = vec![
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value1.to_bits() as u64,
            ))),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value2.to_bits() as u64,
            ))),
        ];

        let scalar = Scalar::new(
            list_dtype,
            ScalarValue(InnerScalarValue::List(elements.into())),
        );

        let list_scalar = scalar.as_list();
        let elements = list_scalar.elements().unwrap();

        for (i, expected) in [f16_value1, f16_value2].iter().enumerate() {
            match elements[i].value() {
                ScalarValue(InnerScalarValue::Primitive(PValue::F16(v))) => {
                    assert_eq!(v, expected, "Element {i} mismatch");
                }
                _ => panic!("Expected F16 value for element {i} after coercion"),
            }
        }
    }

    #[test]
    fn test_coercion_with_overflow_protection() {
        // Test that values too large for target type are not coerced
        let large_u64 = u64::MAX;

        // This should NOT be coerced to F16 because it's too large
        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(large_u64))),
        );

        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(v))) => {
                assert_eq!(*v, large_u64);
            }
            _ => panic!("Expected U64 value to remain unchanged when too large for F16"),
        }
    }

    #[test]
    fn test_extension_dtype_coercion() {
        // Create an extension type with f16 storage
        let ext_id = ExtID::new("test_f16_ext".into());
        let storage_dtype = Arc::new(DType::Primitive(PType::F16, Nullability::NonNullable));
        let ext_dtype = Arc::new(ExtDType::new(ext_id, storage_dtype, None));

        // Test f16 value stored as u64 gets coerced through extension type
        let f16_value = f16::from_f32(0.42);
        let u64_bits = f16_value.to_bits() as u64;

        let scalar = Scalar::new(
            DType::Extension(ext_dtype),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        // Verify the value was coerced to f16
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::F16(v))) => {
                assert_eq!(*v, f16_value);
            }
            _ => panic!("Expected F16 value after extension type coercion"),
        }
    }

    #[test]
    fn test_extension_dtype_no_coercion() {
        // Create an extension type with u32 storage
        let ext_id = ExtID::new("test_u32_ext".into());
        let storage_dtype = Arc::new(DType::Primitive(PType::U32, Nullability::NonNullable));
        let ext_dtype = Arc::new(ExtDType::new(ext_id, storage_dtype, None));

        // Test u32 value is not coerced
        let u32_value = 42u32;

        let scalar = Scalar::new(
            DType::Extension(ext_dtype),
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(u32_value))),
        );

        // Verify the value remains u32
        match scalar.value() {
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(v))) => {
                assert_eq!(*v, u32_value);
            }
            _ => panic!("Expected U32 value to remain unchanged"),
        }
    }

    #[test]
    fn test_extension_dtype_nested_struct_coercion() {
        // Create an extension type with struct storage that contains f16 field
        let ext_id = ExtID::new("test_struct_ext".into());
        let struct_dtype = Arc::new(DType::Struct(
            StructFields::from_iter([
                (
                    "id",
                    FieldDType::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
                ),
                (
                    "value",
                    FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable)),
                ),
            ]),
            Nullability::NonNullable,
        ));
        let ext_dtype = Arc::new(ExtDType::new(ext_id, struct_dtype, None));

        // Create struct value with f16 stored as u64
        let f16_value = f16::from_f32(1.5);
        let field_values = vec![
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(123))),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value.to_bits() as u64,
            ))),
        ];

        let scalar = Scalar::new(
            DType::Extension(ext_dtype),
            ScalarValue(InnerScalarValue::List(field_values.into())),
        );

        // Verify the struct field was coerced
        match scalar.value() {
            ScalarValue(InnerScalarValue::List(fields)) => {
                assert_eq!(fields.len(), 2);

                // Check ID field (no coercion)
                match &fields[0].0 {
                    InnerScalarValue::Primitive(PValue::U32(v)) => {
                        assert_eq!(*v, 123);
                    }
                    _ => panic!("Expected U32 value for ID field"),
                }

                // Check value field (f16 coerced from u64)
                match &fields[1].0 {
                    InnerScalarValue::Primitive(PValue::F16(v)) => {
                        assert_eq!(*v, f16_value);
                    }
                    _ => panic!("Expected F16 value for value field after coercion"),
                }
            }
            _ => panic!("Expected List value for struct storage in extension type"),
        }
    }
}
