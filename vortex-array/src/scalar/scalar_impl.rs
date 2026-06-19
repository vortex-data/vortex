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
use crate::dtype::StructFields;
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
            // A fixed-size list is zero only if it has the expected number of elements and every
            // element is itself a non-null zero value.
            DType::FixedSizeList(_, list_size, _) => {
                let list = self.as_list();
                list.len() == *list_size as usize
                    && (0..list.len())
                        .all(|i| list.element(i).is_some_and(|e| e.is_zero() == Some(true)))
            }
            // A struct is zero only if every one of its fields is itself a non-null zero value.
            DType::Struct(..) => self
                .as_struct()
                .fields_iter()
                .is_some_and(|mut fields| fields.all(|f| f.is_zero() == Some(true))),
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::Variant(_) => self.as_variant().is_zero()?,
            DType::Extension(_) => self.as_extension().to_storage_scalar().is_zero()?,
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
            DType::List(..) | DType::FixedSizeList(..) => self
                .as_list()
                .elements()
                .map(|fields| fields.into_iter().map(|f| f.approx_nbytes()).sum::<usize>())
                .unwrap_or_default(),
            DType::Struct(..) => self
                .as_struct()
                .fields_iter()
                .map(|fields| fields.into_iter().map(|f| f.approx_nbytes()).sum::<usize>())
                .unwrap_or_default(),
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::Variant(_) => self.as_variant().value().map_or(0, Scalar::approx_nbytes),
            DType::Extension(_) => self.as_extension().to_storage_scalar().approx_nbytes(),
        }
    }
}

/// We implement `Hash` manually to be consistent with `PartialEq`. Since we ignore nullability in
/// equality comparisons, we must also ignore it when hashing to maintain the invariant that equal
/// values have equal hashes.
impl Hash for Scalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dtype.as_nonnullable().hash(state);
        self.value.hash(state);
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
    ///
    /// ```
    /// use std::cmp::Ordering;
    /// use vortex_array::dtype::DType;
    /// use vortex_array::dtype::Nullability;
    /// use vortex_array::dtype::PType;
    /// use vortex_array::scalar::Scalar;
    ///
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

        partial_cmp_scalar_values(self.dtype(), self.value(), other.value())
    }
}

/// Compare two optional scalar values using `dtype` for nested tuple interpretation.
fn partial_cmp_scalar_values(
    dtype: &DType,
    lhs: Option<&ScalarValue>,
    rhs: Option<&ScalarValue>,
) -> Option<Ordering> {
    match (lhs, rhs) {
        (None, None) => Some(Ordering::Equal),
        (None, Some(_)) => Some(Ordering::Less),
        (Some(_), None) => Some(Ordering::Greater),
        (Some(lhs), Some(rhs)) => partial_cmp_non_null_scalar_values(dtype, lhs, rhs),
    }
}

/// Compare two non-null scalar values, consulting `dtype` only for tuple-backed values.
fn partial_cmp_non_null_scalar_values(
    dtype: &DType,
    lhs: &ScalarValue,
    rhs: &ScalarValue,
) -> Option<Ordering> {
    // `Scalar::validate` guarantees that a scalar's value matches its dtype. Most of the scalar
    // value variants have only 1 method of comparison, regardless of the dtype.
    match (lhs, rhs) {
        (ScalarValue::Bool(lhs), ScalarValue::Bool(rhs)) => lhs.partial_cmp(rhs),
        (ScalarValue::Primitive(lhs), ScalarValue::Primitive(rhs)) => lhs.partial_cmp(rhs),
        (ScalarValue::Decimal(lhs), ScalarValue::Decimal(rhs)) => lhs.partial_cmp(rhs),
        (ScalarValue::Utf8(lhs), ScalarValue::Utf8(rhs)) => lhs.partial_cmp(rhs),
        (ScalarValue::Binary(lhs), ScalarValue::Binary(rhs)) => lhs.partial_cmp(rhs),
        // `Tuple` is the exception here. Since it backs lists, fixed-size lists, and structs, we
        // need the dtype to know whether children share one element dtype or use per-field dtypes.
        (ScalarValue::Tuple(lhs), ScalarValue::Tuple(rhs)) => {
            partial_cmp_tuple_values(dtype, lhs, rhs)
        }
        // Variant values can have a different dtype in each row, so it doesn't make sense to
        // compare them.
        (ScalarValue::Variant(_), ScalarValue::Variant(_)) => None,
        _ => None,
    }
}

/// Compare tuple values according to the list, fixed-size list, or struct dtype layout.
fn partial_cmp_tuple_values(
    dtype: &DType,
    lhs: &[Option<ScalarValue>],
    rhs: &[Option<ScalarValue>],
) -> Option<Ordering> {
    match dtype {
        DType::List(element_dtype, _) | DType::FixedSizeList(element_dtype, ..) => {
            partial_cmp_list_values(element_dtype, lhs, rhs)
        }
        DType::Struct(fields, _) => partial_cmp_struct_values(fields, lhs, rhs),
        DType::Extension(ext_dtype) => {
            partial_cmp_tuple_values(ext_dtype.storage_dtype(), lhs, rhs)
        }
        _ => None,
    }
}

/// Compare list tuple values using the shared element dtype for each element.
fn partial_cmp_list_values(
    element_dtype: &DType,
    lhs: &[Option<ScalarValue>],
    rhs: &[Option<ScalarValue>],
) -> Option<Ordering> {
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        match partial_cmp_scalar_values(element_dtype, lhs.as_ref(), rhs.as_ref())? {
            Ordering::Equal => continue,
            ordering => return Some(ordering),
        }
    }

    Some(lhs.len().cmp(&rhs.len()))
}

/// Compare struct tuple values using each field's dtype in field order.
fn partial_cmp_struct_values(
    fields: &StructFields,
    lhs: &[Option<ScalarValue>],
    rhs: &[Option<ScalarValue>],
) -> Option<Ordering> {
    if lhs.len() != fields.nfields() || rhs.len() != fields.nfields() {
        return None;
    }

    for ((field_dtype, lhs), rhs) in fields.fields().zip(lhs.iter()).zip(rhs.iter()) {
        match partial_cmp_scalar_values(&field_dtype, lhs.as_ref(), rhs.as_ref())? {
            Ordering::Equal => continue,
            ordering => return Some(ordering),
        }
    }

    Some(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::scalar::Scalar;

    fn i32_scalar(value: i32) -> Scalar {
        Scalar::primitive::<i32>(value, Nullability::NonNullable)
    }

    fn nullable_i32(value: Option<i32>) -> Scalar {
        match value {
            Some(value) => Scalar::primitive::<i32>(value, Nullability::Nullable),
            None => Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        }
    }

    fn ab_struct_dtype(nullability: Nullability) -> DType {
        DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Utf8(Nullability::NonNullable),
                ],
            ),
            nullability,
        )
    }

    #[rstest]
    // A fixed-size list of all-zero elements is itself zero.
    #[case(vec![0, 0], Some(true))]
    #[case(vec![0], Some(true))]
    // A single non-zero element makes the whole list non-zero. On `develop` these incorrectly
    // returned `Some(true)` because only the element count was checked.
    #[case(vec![0, 5], Some(false))]
    #[case(vec![5, 0], Some(false))]
    #[case(vec![1, 2], Some(false))]
    fn fixed_size_list_is_zero(#[case] values: Vec<i32>, #[case] expected: Option<bool>) {
        let element_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let children: Vec<Scalar> = values.into_iter().map(i32_scalar).collect();
        let scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
        assert_eq!(scalar.is_zero(), expected);
    }

    #[test]
    fn null_fixed_size_list_is_zero_is_none() {
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let scalar = Scalar::null(DType::FixedSizeList(
            element_dtype,
            2,
            Nullability::Nullable,
        ));
        assert_eq!(scalar.is_zero(), None);
    }

    #[test]
    fn fixed_size_list_with_null_element_is_not_zero() {
        // A non-null fixed-size list containing a null element is not a zero value. On `develop`
        // this incorrectly returned `Some(true)`.
        let element_dtype = DType::Primitive(PType::I32, Nullability::Nullable);
        let children = vec![nullable_i32(Some(0)), nullable_i32(None)];
        let scalar = Scalar::fixed_size_list(element_dtype, children, Nullability::NonNullable);
        assert_eq!(scalar.is_zero(), Some(false));
    }

    #[test]
    fn struct_with_all_zero_fields_is_zero() {
        let scalar = Scalar::struct_(
            ab_struct_dtype(Nullability::NonNullable),
            vec![i32_scalar(0), Scalar::utf8("", Nullability::NonNullable)],
        );
        assert_eq!(scalar.is_zero(), Some(true));
    }

    #[rstest]
    // A non-zero primitive field, a non-empty string field, or both, make the struct non-zero. On
    // `develop` all of these incorrectly returned `Some(true)`.
    #[case(5, "")]
    #[case(0, "x")]
    #[case(7, "y")]
    fn struct_with_non_zero_field_is_not_zero(#[case] a: i32, #[case] b: &str) {
        let scalar = Scalar::struct_(
            ab_struct_dtype(Nullability::NonNullable),
            vec![i32_scalar(a), Scalar::utf8(b, Nullability::NonNullable)],
        );
        assert_eq!(scalar.is_zero(), Some(false));
    }

    #[test]
    fn null_struct_is_zero_is_none() {
        let scalar = Scalar::null(ab_struct_dtype(Nullability::Nullable));
        assert_eq!(scalar.is_zero(), None);
    }

    #[test]
    fn struct_with_null_field_is_not_zero() {
        // A non-null struct with a null field is not a zero value. On `develop` this incorrectly
        // returned `Some(true)`.
        let dtype = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::Nullable),
                    DType::Primitive(PType::I32, Nullability::Nullable),
                ],
            ),
            Nullability::NonNullable,
        );
        let scalar = Scalar::struct_(dtype, vec![nullable_i32(Some(0)), nullable_i32(None)]);
        assert_eq!(scalar.is_zero(), Some(false));
    }

    #[test]
    fn nested_struct_of_fixed_size_list_recurses() {
        // Zero-checking must recurse through both structs and fixed-size lists. On `develop` the
        // non-zero case incorrectly returned `Some(true)`.
        let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let fsl_dtype =
            DType::FixedSizeList(Arc::clone(&element_dtype), 2, Nullability::NonNullable);
        let struct_dtype = DType::Struct(
            StructFields::new(["fsl"].into(), vec![fsl_dtype]),
            Nullability::NonNullable,
        );

        let all_zero = Scalar::struct_(
            struct_dtype.clone(),
            vec![Scalar::fixed_size_list(
                Arc::clone(&element_dtype),
                vec![i32_scalar(0), i32_scalar(0)],
                Nullability::NonNullable,
            )],
        );
        assert_eq!(all_zero.is_zero(), Some(true));

        let with_non_zero = Scalar::struct_(
            struct_dtype,
            vec![Scalar::fixed_size_list(
                element_dtype,
                vec![i32_scalar(0), i32_scalar(9)],
                Nullability::NonNullable,
            )],
        );
        assert_eq!(with_non_zero.is_zero(), Some(false));
    }
}
