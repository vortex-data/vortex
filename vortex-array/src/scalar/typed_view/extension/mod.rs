// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ExtScalar`] typed view implementation.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use vortex_dtype::DType;
use vortex_dtype::datetime::AnyTemporal;
use vortex_dtype::extension::ExtDTypeRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing an extension type.
///
/// Extension types allow wrapping a storage type with custom semantics.
#[derive(Debug, Clone)]
pub struct ExtScalar<'a> {
    /// A reference to the `DType` of the extension type. This **must** be the [`DType::Extension`
    /// variant.
    dtype: &'a DType,

    /// The extension data type reference.
    ext_dtype: &'a ExtDTypeRef,

    /// The underlying scalar value, or [`None`] if null.
    value: Option<&'a ScalarValue>,
}

impl Display for ExtScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Specialized handling for date/time/timestamp builtin extension types.
        if let Some(temporal) = self.ext_dtype.metadata_opt::<AnyTemporal>() {
            let maybe_timestamp = self
                .to_storage_scalar()
                .as_primitive()
                .as_::<i64>()
                .map(|maybe_timestamp| temporal.to_jiff(maybe_timestamp))
                .transpose()
                .map_err(|_| std::fmt::Error)?;

            match maybe_timestamp {
                None => write!(f, "null"),
                Some(v) => write!(f, "{v}"),
            }
        } else {
            write!(f, "{}({})", self.ext_dtype().id(), self.to_storage_scalar())
        }
    }
}

impl PartialEq for ExtScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ext_dtype.eq_ignore_nullability(other.ext_dtype)
            && self.to_storage_scalar() == other.to_storage_scalar()
    }
}

impl Eq for ExtScalar<'_> {}

// Ord is not implemented since it's undefined for different Extension DTypes
impl PartialOrd for ExtScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if !self.ext_dtype.eq_ignore_nullability(other.ext_dtype) {
            return None;
        }
        self.to_storage_scalar()
            .partial_cmp(&other.to_storage_scalar())
    }
}

impl Hash for ExtScalar<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ext_dtype.hash(state);
        self.to_storage_scalar().hash(state);
    }
}

impl<'a> ExtScalar<'a> {
    // TODO(connor): This should really be validating the data on construction!!!
    /// Creates a new extension scalar from a data type and scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the data type is not an extension type.
    pub fn try_new(dtype: &'a DType, value: Option<&'a ScalarValue>) -> VortexResult<Self> {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Expected extension scalar, found {}", dtype)
        };

        Ok(Self {
            dtype,
            ext_dtype,
            value,
        })
    }

    /// Returns the storage scalar of the extension scalar.
    pub fn to_storage_scalar(&self) -> Scalar {
        Scalar::try_new(self.ext_dtype.storage_dtype().clone(), self.value.cloned())
            .vortex_expect("ExtScalar is invalid")
    }

    /// Return the [`DType`] of the extension scalar.
    pub fn dtype(&self) -> &DType {
        self.dtype
    }

    /// Returns the extension data type.
    pub fn ext_dtype(&self) -> &'a ExtDTypeRef {
        self.ext_dtype
    }

    /// Casts this scalar to the given `dtype`.
    pub(crate) fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        if self.value.is_none() && !dtype.is_nullable() {
            vortex_bail!(
                "cannot cast extension dtype with id {} and storage type {} to {}",
                self.ext_dtype.id(),
                self.ext_dtype.storage_dtype(),
                dtype
            );
        }

        if self.ext_dtype.storage_dtype().eq_ignore_nullability(dtype) {
            // Casting from an extension type to the underlying storage type is OK.
            return Scalar::try_new(dtype.clone(), self.value.cloned());
        }

        if let DType::Extension(ext_dtype) = dtype
            && self.ext_dtype.eq_ignore_nullability(ext_dtype)
        {
            return Scalar::try_new(dtype.clone(), self.value.cloned());
        }

        vortex_bail!(
            "cannot cast extension dtype with id {} and storage type {} to {}",
            self.ext_dtype.id(),
            self.ext_dtype.storage_dtype(),
            dtype
        );
    }
}

#[cfg(test)]
mod tests;
