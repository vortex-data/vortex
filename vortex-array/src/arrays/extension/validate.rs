// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Domain validation for extension types.
//!
//! This module provides validators to ensure that extension array values
//! are always within the valid domain for their type. For example, temporal
//! arrays should not overflow when converted to Jiff types.

use vortex_dtype::ExtDType;
use vortex_dtype::PType;
use vortex_dtype::datetime::TemporalMetadata;
use vortex_dtype::datetime::is_temporal_ext_type;
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

/// Type alias for a domain validator function.
///
/// A domain validator checks whether a scalar value is valid for a given extension type.
/// For example, temporal extension types validate that values don't overflow when converted to Jiff types.
pub type DomainValidator = Box<dyn Fn(&Scalar) -> bool + Send + Sync>;

/// Creates a domain validator for the given extension type.
///
/// This function returns a validator that checks if scalar values are in the valid domain
/// for the extension type. For temporal types (date, time, timestamp), it validates that
/// the values can be successfully converted to Jiff types without overflow.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use vortex_array::arrays::extension::validator_for_ext_type;
/// use vortex_dtype::{ExtDType, ExtMetadata, DType, PType, Nullability};
/// use vortex_dtype::datetime::{TemporalMetadata, TimeUnit, DATE_ID};
/// use vortex_scalar::Scalar;
///
/// let metadata: ExtMetadata = TemporalMetadata::Date(TimeUnit::Days).into();
/// let ext_dtype = ExtDType::new(
///     DATE_ID.clone(),
///     Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
///     Some(metadata),
/// );
///
/// let validator = validator_for_ext_type(&ext_dtype);
///
/// // Valid date value
/// let valid_scalar = Scalar::extension(
///     Arc::new(ext_dtype.clone()),
///     Scalar::primitive(18000i32, Nullability::NonNullable),
/// );
/// assert!(validator(&valid_scalar));
///
/// // Null is always valid
/// let null_scalar = Scalar::null(DType::Extension(Arc::new(ext_dtype)));
/// assert!(validator(&null_scalar));
/// ```
pub fn validator_for_ext_type(ext_dtype: &ExtDType) -> DomainValidator {
    if is_temporal_ext_type(ext_dtype.id()) {
        // For temporal types, validate that the value can be converted to Jiff
        let metadata = TemporalMetadata::try_from(ext_dtype)
            .vortex_expect("temporal ext_dtype should have valid metadata");

        Box::new(move |scalar: &Scalar| {
            if scalar.is_null() {
                return true;
            }

            // Extract the storage value and validate it can be converted to Jiff
            let ext_scalar = scalar.as_extension();
            let storage = ext_scalar.storage();
            let primitive = storage.as_primitive();

            // Get the i64 value from the primitive (temporal types use i32 or i64)
            let value = match primitive.ptype() {
                PType::I32 => primitive.typed_value::<i32>().map(|v| v as i64),
                PType::I64 => primitive.typed_value::<i64>(),
                _ => None,
            };

            value.map(|v| metadata.to_jiff(v).is_ok()).unwrap_or(false)
        })
    } else {
        // Unknown extension type - accept all values
        Box::new(|_| true)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::DType;
    use vortex_dtype::ExtDType;
    use vortex_dtype::ExtMetadata;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_dtype::datetime::DATE_ID;
    use vortex_dtype::datetime::TemporalMetadata;
    use vortex_dtype::datetime::TimeUnit;
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn test_temporal_validator_accepts_valid_values() {
        let metadata: ExtMetadata = TemporalMetadata::Date(TimeUnit::Days).into();
        let ext_dtype = ExtDType::new(
            DATE_ID.clone(),
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Some(metadata),
        );

        let validator = validator_for_ext_type(&ext_dtype);

        // Valid date (days since epoch)
        let valid_scalar = Scalar::extension(
            Arc::new(ext_dtype.clone()),
            Scalar::primitive(18000i32, Nullability::NonNullable),
        );
        assert!(validator(&valid_scalar));
    }

    #[test]
    fn test_temporal_validator_accepts_null() {
        let metadata: ExtMetadata = TemporalMetadata::Date(TimeUnit::Days).into();
        let ext_dtype = ExtDType::new(
            DATE_ID.clone(),
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            Some(metadata),
        );

        let validator = validator_for_ext_type(&ext_dtype);

        let null_scalar = Scalar::null(DType::Extension(Arc::new(ext_dtype)));
        assert!(validator(&null_scalar));
    }
}
