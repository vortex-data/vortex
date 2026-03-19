// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core [`Scalar`] type definition.

use std::cmp::Ordering;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::dtype::NativeDType;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl Scalar {
    // Constructors for null scalars.

    /// Creates a new null [`Scalar`] with the given [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the given [`DType`] is non-nullable.
    pub fn null(dtype: DType) -> Self {
        assert!(
            dtype.is_nullable(),
            "Cannot create null scalar with non-nullable dtype {dtype}"
        );

        Self { dtype, value: None }
    }

    // TODO(connor): This method arguably shouldn't exist...
    /// Creates a new null [`Scalar`] for the given scalar type.
    ///
    /// The resulting scalar will have a nullable version of the type's data type.
    pub fn null_native<T: NativeDType>() -> Self {
        Self {
            dtype: T::dtype().as_nullable(),
            value: None,
        }
    }

    // Constructors for potentially null scalars.

    /// Creates a new [`Scalar`] with the given [`DType`] and potentially null [`ScalarValue`].
    ///
    /// This is just a helper function for tests.
    ///
    /// # Panics
    ///
    /// Panics if the given [`DType`] and [`ScalarValue`] are incompatible.
    #[cfg(test)]
    pub fn new(dtype: DType, value: Option<ScalarValue>) -> Self {
        use vortex_error::VortexExpect;

        Self::try_new(dtype, value).vortex_expect("Failed to create Scalar")
    }

    /// Attempts to create a new [`Scalar`] with the given [`DType`] and potentially null
    /// [`ScalarValue`].
    ///
    /// # Errors
    ///
    /// Returns an error if the given [`DType`] and [`ScalarValue`] are incompatible.
    pub fn try_new(dtype: DType, value: Option<ScalarValue>) -> VortexResult<Self> {
        Self::validate(&dtype, value.as_ref())?;

        Ok(Self { dtype, value })
    }

    /// Creates a new [`Scalar`] with the given [`DType`] and potentially null [`ScalarValue`]
    /// without checking compatibility.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given [`DType`] and [`ScalarValue`] are compatible per the
    /// rules defined in [`Self::validate`].
    pub unsafe fn new_unchecked(dtype: DType, value: Option<ScalarValue>) -> Self {
        #[cfg(debug_assertions)]
        {
            use vortex_error::VortexExpect;

            Self::validate(&dtype, value.as_ref())
                .vortex_expect("Scalar::new_unchecked called with incompatible dtype and value");
        }

        Self { dtype, value }
    }

    /// Returns a default value for the given [`DType`].
    ///
    /// For nullable types, this returns a null scalar. For non-nullable and non-nested types, this
    /// returns the zero value for the type.
    ///
    /// See [`Scalar::zero_value`] for more details about "zero" values.
    ///
    /// For non-nullable and nested types that may need null values in their children (as of right
    /// now, that is _only_ `FixedSizeList` and `Struct`), this function will provide null default
    /// children.
    pub fn default_value(dtype: &DType) -> Self {
        let value = ScalarValue::default_value(dtype);

        // SAFETY: We assume that `default_value` creates a valid `ScalarValue` for the `DType`.
        unsafe { Self::new_unchecked(dtype.clone(), value) }
    }

    /// Returns a non-null zero / identity value for the given [`DType`].
    ///
    /// # Zero Values
    ///
    /// Here is the list of zero values for each [`DType`] (when the [`DType`] is non-nullable):
    ///
    /// - `Null`: Does not have a "zero" value
    /// - `Bool`: `false`
    /// - `Primitive`: `0`
    /// - `Decimal`: `0`
    /// - `Utf8`: `""`
    /// - `Binary`: An empty buffer
    /// - `List`: An empty list
    /// - `FixedSizeList`: A list (with correct size) of zero values, which is determined by the
    ///   element [`DType`]
    /// - `Struct`: A struct where each field has a zero value, which is determined by the field
    ///   [`DType`]
    /// - `Extension`: The zero value of the storage [`DType`]
    pub fn zero_value(dtype: &DType) -> Self {
        let value = ScalarValue::zero_value(dtype);

        // SAFETY: We assume that `zero_value` creates a valid `ScalarValue` for the `DType`.
        unsafe { Self::new_unchecked(dtype.clone(), Some(value)) }
    }

    // Other methods.

    /// Check if two scalars are equal, ignoring nullability of the [`DType`].
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(&other.dtype) && self.value == other.value
    }

    /// Returns the parts of the [`Scalar`].
    pub fn into_parts(self) -> (DType, Option<ScalarValue>) {
        (self.dtype, self.value)
    }

    /// Returns the [`DType`] of the [`Scalar`].
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns an optional [`ScalarValue`] of the [`Scalar`], where `None` means the value is null.
    pub fn value(&self) -> Option<&ScalarValue> {
        self.value.as_ref()
    }

    /// Returns the internal optional [`ScalarValue`], where `None` means the value is null,
    /// consuming the [`Scalar`].
    pub fn into_value(self) -> Option<ScalarValue> {
        self.value
    }

    /// Returns `true` if the [`Scalar`] has a non-null value.
    pub fn is_valid(&self) -> bool {
        self.value.is_some()
    }

    /// Returns `true` if the [`Scalar`] is null.
    pub fn is_null(&self) -> bool {
        self.value.is_none()
    }

    /// Returns `true` if the [`Scalar`] has a non-null zero value.
    ///
    /// Returns `None` if the scalar is null, otherwise returns `Some(true)` if the value is zero
    /// and `Some(false)` otherwise.
    pub fn is_zero(&self) -> Option<bool> {
        let value = self.value()?;

        let is_zero = match self.dtype() {
            DType::Null => vortex_panic!("non-null value somehow had `DType::Null`"),
            DType::Bool(_) => !value.as_bool(),
            DType::Primitive(..) => value.as_primitive().is_zero(),
            DType::Decimal(..) => value.as_decimal().is_zero(),
            DType::Utf8(_) => value.as_utf8().is_empty(),
            DType::Binary(_) => value.as_binary().is_empty(),
            DType::List(..) => value.as_list().is_empty(),
            DType::FixedSizeList(_, list_size, _) => value.as_list().len() == *list_size as usize,
            DType::Struct(struct_fields, _) => value.as_list().len() == struct_fields.nfields(),
            DType::Extension(_) => self.as_extension().to_storage_scalar().is_zero()?,
            DType::Variant(_) => self.as_variant().is_zero()?,
        };

        Some(is_zero)
    }

    /// Reinterprets the bytes of this scalar as a different primitive type.
    ///
    /// # Errors
    ///
    /// Panics if the scalar is not a primitive type or if the types have different byte widths.
    pub fn primitive_reinterpret_cast(&self, ptype: PType) -> VortexResult<Self> {
        let primitive = self.as_primitive();
        if primitive.ptype() == ptype {
            return Ok(self.clone());
        }

        vortex_ensure_eq!(
            primitive.ptype().byte_width(),
            ptype.byte_width(),
            "can't reinterpret cast between integers of two different widths"
        );

        Scalar::try_new(
            DType::Primitive(ptype, self.dtype().nullability()),
            primitive
                .pvalue()
                .map(|p| p.reinterpret_cast(ptype))
                .map(ScalarValue::Primitive),
        )
    }

    /// Returns an **ESTIMATE** of the size of the scalar in bytes, uncompressed.
    ///
    /// Note that the protobuf serialization of scalars will likely have a different (but roughly
    /// similar) length.
    pub fn approx_nbytes(&self) -> usize {
        use crate::dtype::NativeDecimalType;
        use crate::dtype::i256;

        match self.dtype() {
            DType::Null => 0,
            DType::Bool(_) => 1,
            DType::Primitive(ptype, _) => ptype.byte_width(),
            DType::Decimal(dt, _) => {
                if dt.precision() <= i128::MAX_PRECISION {
                    size_of::<i128>()
                } else {
                    size_of::<i256>()
                }
            }
            DType::Utf8(_) => self
                .value()
                .map_or_else(|| 0, |value| value.as_utf8().len()),
            DType::Binary(_) => self
                .value()
                .map_or_else(|| 0, |value| value.as_binary().len()),
            DType::Struct(..) => self
                .as_struct()
                .fields_iter()
                .map(|fields| fields.into_iter().map(|f| f.approx_nbytes()).sum::<usize>())
                .unwrap_or_default(),
            DType::List(..) | DType::FixedSizeList(..) => self
                .as_list()
                .elements()
                .map(|fields| fields.into_iter().map(|f| f.approx_nbytes()).sum::<usize>())
                .unwrap_or_default(),
            DType::Extension(_) => self.as_extension().to_storage_scalar().approx_nbytes(),
            DType::Variant(_) => self.as_variant().value().map_or(0, Scalar::approx_nbytes),
        }
    }
}

/// We implement `PartialEq` manually because we want to ignore nullability when comparing scalars.
/// Two scalars with the same value but different nullability should be considered equal.
///
/// Note that this has **different** behavior than the [`PartialOrd`] implementation since the
/// [`PartialOrd`] returns `None` if the types are different, whereas this `PartialEq`
/// implementation simply returns `false`.
impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(&other.dtype) && self.value == other.value
    }
}

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
        self.value().partial_cmp(&other.value())
    }
}

/// We implement `Hash` manually to be consistent with `PartialEq`. Since we ignore nullability
/// in equality comparisons, we must also ignore it when hashing to maintain the invariant that
/// equal values have equal hashes.
impl Hash for Scalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.as_nonnullable().hash(state);
        self.value.hash(state);
    }
}
