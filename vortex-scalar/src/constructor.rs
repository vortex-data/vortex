// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed constructors for [`Scalar`].

use std::sync::Arc;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_dtype::DecimalDType;
use vortex_dtype::ExtDType;
use vortex_dtype::ExtDTypeRef;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::DecimalValue;
use crate::PValue;
use crate::Scalar;
use crate::ScalarValue;
use crate::extension::ExtScalarVTable;
use crate::extension::ExtScalarValue;
use crate::session::ScalarSessionExt;

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

    /// Creates a new list scalar with the given element type and children.
    ///
    /// # Panics
    ///
    /// Panics if any child scalar has a different type than the element type, or if there are too
    /// many children.
    pub fn list(
        element_dtype: impl Into<Arc<DType>>,
        children: Vec<Scalar>,
        nullability: Nullability,
    ) -> Self {
        Self::create_list(element_dtype, children, nullability, ListKind::Variable)
    }

    /// Creates a new empty list scalar with the given element type.
    pub fn list_empty(element_dtype: Arc<DType>, nullability: Nullability) -> Self {
        Self::create_list(element_dtype, vec![], nullability, ListKind::Variable)
    }

    /// Creates a new fixed-size list scalar with the given element type and children.
    ///
    /// # Panics
    ///
    /// Panics if any child scalar has a different type than the element type, or if there are too
    /// many children.
    pub fn fixed_size_list(
        element_dtype: impl Into<Arc<DType>>,
        children: Vec<Scalar>,
        nullability: Nullability,
    ) -> Self {
        Self::create_list(element_dtype, children, nullability, ListKind::FixedSize)
    }

    /// Creates a list [`Scalar`] from an element dtype, children, nullability, and list kind.
    fn create_list(
        element_dtype: impl Into<Arc<DType>>,
        children: Vec<Scalar>,
        nullability: Nullability,
        list_kind: ListKind,
    ) -> Self {
        let element_dtype = element_dtype.into();

        let children: Vec<Option<ScalarValue>> = children
            .into_iter()
            .map(|child| {
                if child.dtype() != &*element_dtype {
                    vortex_panic!(
                        "tried to create list of {} with values of type {}",
                        element_dtype,
                        child.dtype()
                    );
                }
                child.into_value()
            })
            .collect();
        let size: u32 = children
            .len()
            .try_into()
            .vortex_expect("tried to create a list that was too large");

        let dtype = match list_kind {
            ListKind::Variable => DType::List(element_dtype, nullability),
            ListKind::FixedSize => DType::FixedSizeList(element_dtype, size, nullability),
        };

        Self::try_new(dtype, Some(ScalarValue::List(children)))
            .vortex_expect("unable to construct a list `Scalar`")
    }

    // TODO(connor): This needs to return a `VortexResult` instead.
    /// Creates a new extension scalar wrapping the given storage value.
    ///
    /// # Panics
    ///
    /// Panics if the storage dtype is incompatible with the extension type, or if the storage
    /// value fails validation.
    pub fn extension<V: ExtScalarVTable + Default>(
        metadata: V::Metadata,
        storage_scalar: Scalar,
    ) -> Self {
        let ext_dtype = ExtDType::<V>::try_new(metadata, storage_scalar.dtype().clone())
            .vortex_expect("Failed to create extension dtype");
        let storage_value = storage_scalar.into_value();

        let ext_value = storage_value.map(|sv| {
            let owned = ExtScalarValue::<V>::try_new(ext_dtype.clone(), sv)
                .vortex_expect("unable to construct an extension `Scalar`");
            ScalarValue::Extension(owned.erased())
        });

        Self::try_new(DType::Extension(ext_dtype.erased()), ext_value)
            .vortex_expect("unable to construct an extension `Scalar`")
    }

    /// TODO docs.
    pub fn extension_ref(
        ext_dtype: ExtDTypeRef,
        storage_scalar: Scalar,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let (storage_dtype, storage_value) = storage_scalar.into_parts();
        Self::extension_ref_from_value(ext_dtype, &storage_dtype, storage_value, session)
    }

    /// TODO docs.
    pub fn extension_ref_from_value(
        ext_dtype: ExtDTypeRef,
        storage_dtype: &DType,
        storage_value: Option<ScalarValue>,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        vortex_ensure_eq!(ext_dtype.storage_dtype(), storage_dtype);

        let ext_value = Self::extension_value(&ext_dtype, storage_value, session)?;

        Ok(
            // SAFETY: `create_ext_scalar_value_ref` validates that the scalar value is compatible.
            unsafe { Scalar::new_unchecked(DType::Extension(ext_dtype), ext_value) },
        )
    }
}

/// A helper enum for creating a list scalar.
enum ListKind {
    /// Variable-length list.
    Variable,
    /// Fixed-size list.
    FixedSize,
}
