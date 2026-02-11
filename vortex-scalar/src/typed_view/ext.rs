// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ExtScalar`] typed view implementation.

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_dtype::DType;
use vortex_dtype::extension::ExtDTypeRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::Scalar;
use crate::ScalarValue;
use crate::extension::ExtScalarValueRef;
use crate::extension::Matcher;

/// A scalar value representing an extension type.
///
/// Extension types allow wrapping a storage type with custom semantics.
#[derive(Debug, Clone)]
pub struct ExtScalar<'a> {
    /// The dtype reference..
    dtype: &'a DType,

    /// The extension dtype reference.
    ext_dtype: &'a ExtDTypeRef,

    /// The underlying erased extension value, or [`None`] if null.
    ext_value: Option<&'a ExtScalarValueRef>,
}

impl<'a> ExtScalar<'a> {
    /// TODO docs.
    ///
    /// # Errors
    ///
    /// TODO errors.
    pub fn try_new(dtype: &'a DType, value: Option<&'a ScalarValue>) -> VortexResult<Self> {
        let ext_dtype = dtype
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("Cannot create an ExtScalar with a non-extension dtype"))?;

        let ext_value = value
            .map(|value| {
                value.as_extension_opt().ok_or_else(|| {
                    vortex_err!("Cannot create an ExtScalar with a non-extension scalar value")
                })
            })
            .transpose()?;

        Ok(Self {
            dtype,
            ext_dtype,
            ext_value,
        })
    }

    /// Return the dtype of the extension scalar.
    pub fn dtype(&self) -> &DType {
        self.dtype
    }

    /// Return the type-erased dtype of the extension scalar.
    pub fn ext_dtype(&self) -> &ExtDTypeRef {
        self.ext_dtype
    }

    /// Return the typed-erased storage scalar value of the extension scalar.
    pub fn ext_value(&self) -> Option<&ExtScalarValueRef> {
        self.ext_value
    }

    /// Match on the extension type and extract an underlying value.
    ///
    /// # Panics
    ///
    /// Panics if the matcher fails.
    pub fn as_value<M: Matcher>(&self) -> M::Match<'_> {
        self.as_value_opt::<M>()
            .vortex_expect("Failed to match extension scalar value")
    }

    /// Match on the extension type and extract an underlying value.
    pub fn as_value_opt<M: Matcher>(&self) -> Option<M::Match<'_>> {
        M::try_match(self)
    }

    /// Return the underlying storage value.
    pub fn storage_value(&self) -> Option<&ScalarValue> {
        self.ext_value.as_ref().map(|s| s.storage_value())
    }

    /// Convert the underlying storage value into a standard [`Scalar`] that has a [`DType`] equal
    /// to this [`ExtScalar`]'s storage type.
    pub fn to_storage_scalar(&self) -> Scalar {
        let storage_value = self.ext_value.as_ref().map(|s| s.storage_value()).cloned();

        unsafe { Scalar::new_unchecked(self.ext_dtype.storage_dtype().clone(), storage_value) }
    }

    // TODO(connor): This is somewhat naive behavior...
    /// Casts this scalar to the given `dtype`.
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if self.ext_value.is_none() && !dtype.is_nullable() {
            vortex_bail!(
                "cannot cast extension dtype with id {} and storage type {} to {}",
                self.ext_dtype.id(),
                self.ext_dtype.storage_dtype(),
                dtype
            );
        }

        if self.ext_dtype.storage_dtype().eq_ignore_nullability(dtype) {
            // Casting from an extension type to the underlying storage type is OK.
            return Scalar::try_new(dtype.clone(), self.storage_value().cloned());
        }

        if let DType::Extension(ext_dtype) = dtype
            && self.ext_dtype.eq_ignore_nullability(ext_dtype)
        {
            return Scalar::try_new(dtype.clone(), self.storage_value().cloned());
        }

        vortex_bail!(
            "cannot cast extension dtype with id {} and storage type {} to {}",
            self.ext_dtype.id(),
            self.ext_dtype.storage_dtype(),
            dtype
        );
    }
}

impl Display for ExtScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.ext_value.as_ref() {
            Some(scalar) => scalar.fmt_ext_scalar(self.ext_dtype, f),
            None => write!(f, "null"),
        }
    }
}
