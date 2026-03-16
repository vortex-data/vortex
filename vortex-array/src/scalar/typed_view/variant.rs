// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`VariantScalar`] typed view implementation.

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A typed view into a [`DType::Variant`] scalar.
#[derive(Debug, Clone, Copy)]
pub struct VariantScalar<'a> {
    dtype: &'a DType,
    value: Option<&'a Scalar>,
}

impl Display for VariantScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.value {
            None => write!(f, "null"),
            Some(value) => write!(f, "variant({value})"),
        }
    }
}

impl<'a> VariantScalar<'a> {
    /// Creates a new [`VariantScalar`] from a [`DType`] and optional [`ScalarValue`].
    #[inline]
    pub(crate) fn try_new(dtype: &'a DType, value: Option<&'a ScalarValue>) -> VortexResult<Self> {
        if !matches!(dtype, DType::Variant(_)) {
            vortex_bail!("Expected variant scalar, found {}", dtype)
        }

        Ok(Self {
            dtype,
            value: value.map(ScalarValue::as_variant),
        })
    }

    /// Returns the data type of this variant scalar.
    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// Returns `true` if the outer variant scalar is null.
    pub fn is_null(&self) -> bool {
        self.value.is_none()
    }

    /// Returns the nested row-specific scalar if the outer variant is present.
    pub fn value(&self) -> Option<&'a Scalar> {
        self.value
    }

    /// Returns whether the present variant payload is the variant-null value.
    ///
    /// Returns `None` if the outer variant is null.
    pub fn is_variant_null(&self) -> Option<bool> {
        Some(self.value?.is_null())
    }
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;

    #[test]
    fn outer_null_variant_view() {
        let scalar = Scalar::null(DType::Variant(Nullability::Nullable));
        let variant = scalar.as_variant();

        assert!(variant.is_null());
        assert_eq!(variant.is_variant_null(), None);
        assert!(variant.value().is_none());
        assert_eq!(scalar.is_variant_null(), None);
        assert_eq!(scalar.is_zero(), None);
    }

    #[test]
    fn present_variant_null_view() {
        let scalar = Scalar::variant(Scalar::null(DType::Null));
        let variant = scalar.as_variant();

        assert!(!variant.is_null());
        assert_eq!(variant.is_variant_null(), Some(true));
        assert!(variant.value().is_some_and(Scalar::is_null));
        assert_eq!(scalar.is_variant_null(), Some(true));
        assert_eq!(scalar.is_zero(), Some(true));
    }

    #[test]
    fn present_variant_value_view() {
        let scalar = Scalar::variant(Scalar::from(42_u32));
        let variant = scalar.as_variant();

        assert!(!variant.is_null());
        assert_eq!(variant.is_variant_null(), Some(false));
        assert_eq!(
            variant.value().map(ToString::to_string).as_deref(),
            Some("42u32")
        );
        assert_eq!(scalar.is_variant_null(), Some(false));
        assert_eq!(scalar.is_zero(), Some(false));
    }
}
