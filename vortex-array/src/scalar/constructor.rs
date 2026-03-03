// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed constructors for [`Scalar`].

use std::sync::Arc;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::builders::build_array_from_scalars;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

// TODO(connor): Really, we want `try_` constructors that return errors instead of just panic.
impl Scalar {
    /// Creates a new boolean scalar with the given value and nullability.
    pub fn bool(value: bool, nullability: Nullability) -> Self {
        Self::try_new(DType::Bool(nullability), Some(ScalarValue::Bool(value)))
            .vortex_expect("unable to construct a boolean `Scalar`")
    }

    /// Creates a new primitive scalar from a native value.
    pub fn primitive<T: NativePType + Into<PValue>>(value: T, nullability: Nullability) -> Self {
        Self::primitive_value(value.into(), T::PTYPE, nullability)
    }

    /// Create a PrimitiveScalar from a PValue.
    ///
    /// Note that an explicit PType is passed since any compatible PValue may be used as the value
    /// for a primitive type.
    pub fn primitive_value(value: PValue, ptype: PType, nullability: Nullability) -> Self {
        Self::try_new(
            DType::Primitive(ptype, nullability),
            Some(ScalarValue::Primitive(value)),
        )
        .vortex_expect("unable to construct a primitive `Scalar`")
    }

    /// Creates a new decimal scalar with the given value, precision, scale, and nullability.
    pub fn decimal(
        value: DecimalValue,
        decimal_type: DecimalDType,
        nullability: Nullability,
    ) -> Self {
        Self::try_new(
            DType::Decimal(decimal_type, nullability),
            Some(ScalarValue::Decimal(value)),
        )
        .vortex_expect("unable to construct a decimal `Scalar`")
    }

    /// Creates a new UTF-8 scalar from a string-like value.
    ///
    /// # Panics
    ///
    /// Panics if the input cannot be converted to a valid UTF-8 string.
    pub fn utf8<B>(str: B, nullability: Nullability) -> Self
    where
        B: Into<BufferString>,
    {
        Self::try_utf8(str, nullability).unwrap()
    }

    /// Tries to create a new UTF-8 scalar from a string-like value.
    ///
    /// # Errors
    ///
    /// Returns an error if the input cannot be converted to a valid UTF-8 string.
    pub fn try_utf8<B>(
        str: B,
        nullability: Nullability,
    ) -> Result<Self, <B as TryInto<BufferString>>::Error>
    where
        B: TryInto<BufferString>,
    {
        Ok(Self::try_new(
            DType::Utf8(nullability),
            Some(ScalarValue::Utf8(str.try_into()?)),
        )
        .vortex_expect("unable to construct a UTF-8 `Scalar`"))
    }

    /// Creates a new binary scalar from a byte buffer.
    pub fn binary(buffer: impl Into<ByteBuffer>, nullability: Nullability) -> Self {
        Self::try_new(
            DType::Binary(nullability),
            Some(ScalarValue::Binary(buffer.into())),
        )
        .vortex_expect("unable to construct a binary `Scalar`")
    }

    /// Creates a new list scalar using the [`DType`] of the parent array and a child array
    /// representing the list as an array.
    ///
    /// # Panics
    ///
    /// Panics if the parent dtype is not a `List` or `FixedSixeList` with an `elem_dtype` equal to
    /// the `elem_list`'s [`DType`].
    ///
    /// Also panics if the parent dtype is a `FixedSizedList` but the `list_size` is not equal to
    /// the array length.
    pub fn list_array(parent_dtype: DType, elem_list: ArrayRef) -> Self {
        match &parent_dtype {
            DType::List(elem_dtype, _) => {
                assert_eq!(elem_dtype.as_ref(), elem_list.dtype());
            }
            DType::FixedSizeList(elem_dtype, list_size, _) => {
                assert_eq!(elem_dtype.as_ref(), elem_list.dtype());
                assert_eq!(elem_list.len(), *list_size as usize);
            }
            _ => vortex_panic!("expected List or FixedSizeList dtype, got {parent_dtype}"),
        };

        let scalar_value = ScalarValue::Array(elem_list);

        // SAFETY: We just checked that the scalar value is valid for this dtype.
        unsafe { Scalar::new_unchecked(parent_dtype, Some(scalar_value)) }
    }

    /// Creates a new list scalar with the given element type and children.
    ///
    /// Callers should prefer to use the faster `Scalar::list_array` constructor when possible.
    ///
    /// # Panics
    ///
    /// Panics if any child scalar has a different type than the element type, or if there are too
    /// many children.
    pub fn list_from_scalars(
        element_dtype: impl Into<Arc<DType>>,
        children: Vec<Scalar>,
        nullability: Nullability,
    ) -> Self {
        Self::create_list_from_scalars(element_dtype, children, nullability, ListKind::Variable)
    }

    /// Creates a new empty list scalar with the given element type.
    pub fn list_empty(element_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self::create_list_from_scalars(element_dtype, vec![], nullability, ListKind::Variable)
    }

    /// Creates a new fixed-size list scalar with the given element type and children.
    ///
    /// Callers should prefer to use the faster `Scalar::list_array` constructor when possible.
    ///
    /// # Panics
    ///
    /// Panics if any child scalar has a different type than the element type, or if there are too
    /// many children.
    pub fn fixed_size_list_from_scalars(
        element_dtype: impl Into<Arc<DType>>,
        children: Vec<Scalar>,
        nullability: Nullability,
    ) -> Self {
        Self::create_list_from_scalars(element_dtype, children, nullability, ListKind::FixedSize)
    }

    /// Creates a list [`Scalar`] from an element dtype, children, nullability, and list kind.
    fn create_list_from_scalars(
        element_dtype: impl Into<Arc<DType>>,
        children: Vec<Scalar>,
        nullability: Nullability,
        list_kind: ListKind,
    ) -> Self {
        let element_dtype = element_dtype.into();

        // Validate all children have the correct dtype.
        for child in &children {
            if child.dtype() != &*element_dtype {
                vortex_panic!(
                    "tried to create list of {} with values of type {}",
                    element_dtype,
                    child.dtype()
                );
            }
        }

        let size: u32 = children
            .len()
            .try_into()
            .vortex_expect("tried to create a list that was too large");

        let array = build_array_from_scalars(&element_dtype, &children);

        let dtype = match list_kind {
            ListKind::Variable => DType::List(element_dtype, nullability),
            ListKind::FixedSize => DType::FixedSizeList(element_dtype, size, nullability),
        };

        Self::try_new(dtype, Some(ScalarValue::Array(array)))
            .vortex_expect("unable to construct a list `Scalar`")
    }

    /// Creates a new extension scalar wrapping the given storage value.
    pub fn extension<V: ExtVTable + Default>(options: V::Metadata, storage_scalar: Scalar) -> Self {
        let ext_dtype = ExtDType::<V>::try_new(options, storage_scalar.dtype().clone())
            .vortex_expect("Failed to create extension dtype");

        Self::extension_ref(ext_dtype.erased(), storage_scalar)
    }

    /// Creates a new extension scalar wrapping the given storage value.
    ///
    /// # Panics
    ///
    /// Panics if the storage dtype of `ext_dtype` does not match `value`'s dtype.
    pub fn extension_ref(ext_dtype: ExtDTypeRef, storage_scalar: Scalar) -> Self {
        assert_eq!(ext_dtype.storage_dtype(), storage_scalar.dtype());

        Self::try_new(DType::Extension(ext_dtype), storage_scalar.into_value())
            .vortex_expect("unable to construct an extension `Scalar`")
    }

    /// Creates a new variant scalar from a row-specific nested scalar.
    ///
    /// Use [`Scalar::null(DType::Variant(Nullability::Nullable))`][Scalar::null] for a top-level
    /// null variant value, and
    /// `Scalar::variant(Scalar::null(DType::Null))` for a defined variant-null.
    pub fn variant(value: Scalar) -> Self {
        Self::try_new(
            DType::Variant(Nullability::NonNullable),
            Some(ScalarValue::Variant(Box::new(value))),
        )
        .vortex_expect("unable to construct a variant `Scalar`")
    }
}

/// A helper enum for creating a [`ListScalar`].
enum ListKind {
    /// Variable-length list.
    Variable,
    /// Fixed-size list.
    FixedSize,
}
