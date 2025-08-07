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
use vortex_dtype::{DECIMAL128_MAX_PRECISION, DType, Nullability};
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
#[cfg(test)]
mod tests;
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
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

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
    pub fn new(dtype: DType, value: ScalarValue) -> Self {
        Self { dtype, value }
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
                vortex_bail!(
                    "Cannot cast null to {}: target type is non-nullable",
                    target
                )
            }
        }

        if self.dtype().eq_ignore_nullability(target) {
            return Ok(Scalar::new(target.clone(), self.value.clone()));
        }

        match &self.dtype {
            DType::Null => unreachable!(), // handled by if is_null case
            DType::Bool(_) => self.as_bool().cast(target),
            DType::Primitive(..) => self.as_primitive().cast(target),
            DType::Decimal(..) => self.as_decimal().cast(target),
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
