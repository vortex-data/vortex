// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::hash::Hash;
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_dtype::{DECIMAL128_MAX_PRECISION, DType, Nullability};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};

use super::*;

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
    /// The type of the scalar.
    dtype: DType,

    /// The value of the scalar.
    ///
    /// Invariant: If the `dtype` is non-nullable, then this value _cannot_ be equal to
    /// [`ScalarValue::null()`](ScalarValue::null).
    value: ScalarValue,
}

impl Scalar {
    /// Creates a new scalar with the given data type and value.
    pub fn new(dtype: DType, value: ScalarValue) -> Self {
        if !dtype.is_nullable() {
            assert!(
                !value.is_null(),
                "Tried to construct a null scalar when the `DType` is non-nullable: {dtype}",
            );
        }

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

    /// Consumes the scalar and returns its underlying [`DType`].
    #[inline]
    pub fn into_dtype(self) -> DType {
        self.dtype
    }

    /// Consumes the scalar and returns its underlying [`ScalarValue`].
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
            "Tried to construct a null scalar when the `DType` is non-nullable: {dtype}"
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
            }

            vortex_bail!("Cannot cast null to {target}: target type is non-nullable")
        }

        match &self.dtype {
            DType::Null => unreachable!(), // Handled by `if self.is_null()` case.
            DType::Bool(_) => self.as_bool().cast(target),
            DType::Primitive(..) => self.as_primitive().cast(target),
            DType::Decimal(..) => self.as_decimal().cast(target),
            DType::Utf8(_) => self.as_utf8().cast(target),
            DType::Binary(_) => self.as_binary().cast(target),
            DType::Struct(..) => self.as_struct().cast(target),
            DType::List(..) | DType::FixedSizeList(..) => self.as_list().cast(target),
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
                if dt.precision() <= DECIMAL128_MAX_PRECISION {
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
            DType::List(..) | DType::FixedSizeList(..) => self
                .as_list()
                .elements()
                .map(|fields| fields.into_iter().map(|f| f.nbytes()).sum::<usize>())
                .unwrap_or_default(),
            DType::Extension(_ext_dtype) => self.as_extension().storage().nbytes(),
        }
    }

    /// Creates a "default" scalar value for the given data type.
    ///
    /// For nullable types, returns null. For non-nullable types, returns an appropriate zero/empty
    /// value.
    ///
    /// # Default Values
    ///
    /// Here is the list of default values for each [`DType`] (when the [`DType`] is non-nullable):
    ///
    /// - `Null`: `null`
    /// - `Bool`: `false`
    /// - `Primitive`: `0`
    /// - `Decimal`: `0`
    /// - `Utf8`: `""`
    /// - `Binary`: An empty buffer
    /// - `List`: An empty list
    /// - `FixedSizeList`: A list (with correct size) of default values, which is determined by the
    ///   element [`DType`]
    /// - `Struct`: A struct where each field has a default value, which is determined by the field
    ///   [`DType`]
    /// - `Extension`: The default value of the storage [`DType`]
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
            DType::List(edt, nullability) => Self::list(edt, vec![], nullability),
            DType::FixedSizeList(edt, size, nullability) => {
                let elements = (0..size)
                    .map(|_| Scalar::default_value(edt.as_ref().clone()))
                    .collect();
                Self::fixed_size_list(edt, elements, nullability)
            }
            DType::Struct(sf, nullability) => {
                let fields: Vec<_> = sf.fields().map(Scalar::default_value).collect();
                Self::struct_(DType::Struct(sf, nullability), fields)
            }
            DType::Extension(dt) => {
                let scalar = Self::default_value(dt.storage_dtype().clone());
                Self::extension(dt, scalar)
            }
        }
    }
}

/// This implementation block contains only `TryFrom` and `From` wrappers (`as_something`).
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
    /// Note that we use [`ListScalar`] to represent **both** [`DType::List`] and
    /// [`DType::FixedSizeList`].
    ///
    /// # Panics
    ///
    /// Panics if the scalar is not a list type.
    pub fn as_list(&self) -> ListScalar<'_> {
        ListScalar::try_from(self).vortex_expect("Failed to convert scalar to list")
    }

    /// Returns a view of the scalar as a list scalar if it has a list type.
    ///
    /// Note that we use [`ListScalar`] to represent **both** [`DType::List`] and
    /// [`DType::FixedSizeList`].
    pub fn as_list_opt(&self) -> Option<ListScalar<'_>> {
        matches!(self.dtype, DType::List(..) | DType::FixedSizeList(..)).then(|| self.as_list())
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

/// It is common to represent a nullable type `T` as an `Option<T>`, so we implement a blanket
/// implementation for all `Option<T>` to simply be a nullable `T`.
impl<T> From<Option<T>> for Scalar
where
    T: ScalarType,
    Scalar: From<T>,
{
    /// A blanket implementation for all `Option<T>`.
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

impl<T> From<Vec<T>> for Scalar
where
    T: ScalarType,
    Scalar: From<T>,
{
    /// Converts a vector into a `Scalar` (where the value is a `ListScalar`).
    fn from(vec: Vec<T>) -> Self {
        Scalar {
            dtype: DType::List(Arc::from(T::dtype()), Nullability::NonNullable),
            value: ScalarValue::from(vec),
        }
    }
}

impl<T> TryFrom<Scalar> for Vec<T>
where
    T: for<'b> TryFrom<&'b Scalar, Error = VortexError>,
{
    type Error = VortexError;

    fn try_from(value: Scalar) -> Result<Self, Self::Error> {
        Vec::try_from(&value)
    }
}

impl<'a, T> TryFrom<&'a Scalar> for Vec<T>
where
    T: for<'b> TryFrom<&'b Scalar, Error = VortexError>,
{
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        ListScalar::try_from(value)?
            .elements()
            .ok_or_else(|| vortex_err!("Expected non-null list"))?
            .into_iter()
            .map(|e| T::try_from(&e))
            .collect::<VortexResult<Vec<T>>>()
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
            DType::List(..) | DType::FixedSizeList(..) => self.as_list() == other.as_list(),
            DType::Extension(_) => self.as_extension() == other.as_extension(),
        }
    }
}

impl Eq for Scalar {}

impl PartialOrd for Scalar {
    /// Compares two scalar values for ordering.
    ///
    /// # Returns
    /// - `Some(Ordering)` if both scalars have the same data type (ignoring nullability)
    /// - `None` if the scalars have different data types
    ///
    /// # Ordering Rules
    /// When types match, the ordering follows these rules:
    /// - Null values are considered less than all non-null values
    /// - Non-null values are compared according to their natural ordering
    ///
    /// # Examples
    /// ```ignore
    /// // Same types compare successfully
    /// let a = Scalar::primitive(10i32, Nullability::NonNullable);
    /// let b = Scalar::primitive(20i32, Nullability::NonNullable);
    /// assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    ///
    /// // Different types return None
    /// let int_scalar = Scalar::primitive(10i32, Nullability::NonNullable);
    /// let str_scalar = Scalar::utf8("hello", Nullability::NonNullable);
    /// assert_eq!(int_scalar.partial_cmp(&str_scalar), None);
    ///
    /// // Nulls are less than non-nulls
    /// let null = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));
    /// let value = Scalar::primitive(0i32, Nullability::Nullable);
    /// assert_eq!(null.partial_cmp(&value), Some(Ordering::Less));
    /// ```
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
            DType::List(..) | DType::FixedSizeList(..) => {
                self.as_list().partial_cmp(&other.as_list())
            }
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
            DType::List(..) | DType::FixedSizeList(..) => self.as_list().hash(state),
            DType::Extension(_) => self.as_extension().hash(state),
        }
    }
}

impl AsRef<Self> for Scalar {
    fn as_ref(&self) -> &Self {
        self
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
