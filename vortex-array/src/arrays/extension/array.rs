// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_dtype::ExtDType;
use vortex_dtype::ExtID;

use crate::ArrayRef;
use crate::stats::ArrayStats;

/// An extension array that wraps another array with additional type information.
///
/// **⚠️ Unstable API**: This is an experimental feature that may change significantly
/// in future versions. The extension type system is still evolving.
///
/// Unlike Apache Arrow's extension arrays, Vortex extension arrays provide a more flexible
/// mechanism for adding semantic meaning to existing array types without requiring
/// changes to the core type system.
///
/// ## Design Philosophy
///
/// Extension arrays serve as a type-safe wrapper that:
/// - Preserves the underlying storage format and operations
/// - Adds semantic type information via `ExtDType`
/// - Enables custom serialization and deserialization logic
/// - Allows domain-specific interpretations of generic data
///
/// ## Storage and Type Relationship
///
/// The extension array maintains a strict contract:
/// - **Storage array**: Contains the actual data in a standard Vortex encoding
/// - **Extension type**: Defines how to interpret the storage data semantically
/// - **Type safety**: The storage array's dtype must match the extension type's storage dtype
///
/// ## Use Cases
///
/// Extension arrays are ideal for:
/// - **Custom numeric types**: Units of measurement, currencies
/// - **Temporal types**: Custom date/time formats, time zones, calendars
/// - **Domain-specific types**: UUIDs, IP addresses, geographic coordinates
/// - **Encoded types**: Base64 strings, compressed data, encrypted values
///
/// ## Validity and Operations
///
/// Extension arrays delegate validity and most operations to their storage array:
/// - Validity is inherited from the underlying storage
/// - Slicing preserves the extension type
/// - Scalar access wraps storage scalars with extension metadata
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use vortex_array::arrays::{ExtensionArray, PrimitiveArray};
/// use vortex_dtype::{ExtDType, ExtID, DType, Nullability, PType};
/// use vortex_array::validity::Validity;
/// use vortex_array::IntoArray;
/// use vortex_buffer::buffer;
///
/// // Define a custom extension type for representing currency values
/// let currency_id = ExtID::from("example.currency");
/// let currency_dtype = Arc::new(ExtDType::new(
///     currency_id,
///     Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)), // Storage as i64 cents
///     None, // No additional metadata needed
/// ));
///
/// // Create storage array with currency values in cents
/// let cents_storage = PrimitiveArray::new(
///     buffer![12345i64, 67890, 99999], // $123.45, $678.90, $999.99
///     Validity::NonNullable
/// );
///
/// // Wrap with extension type
/// let currency_array = ExtensionArray::new(
///     currency_dtype.clone(),
///     cents_storage.into_array()
/// );
///
/// assert_eq!(currency_array.len(), 3);
/// assert_eq!(currency_array.id().as_ref(), "example.currency");
///
/// // Access maintains extension type information
/// let first_value = currency_array.scalar_at(0);
/// assert!(first_value.as_extension_opt().is_some());
/// ```
#[derive(Clone, Debug)]
pub struct ExtensionArray {
    pub(super) dtype: DType,
    pub(super) storage: ArrayRef,
    pub(super) stats_set: ArrayStats,
}

impl ExtensionArray {
    pub fn new(ext_dtype: Arc<ExtDType>, storage: ArrayRef) -> Self {
        assert_eq!(
            ext_dtype.storage_dtype(),
            storage.dtype(),
            "ExtensionArray: storage_dtype must match storage array DType",
        );
        Self {
            dtype: DType::Extension(ext_dtype),
            storage,
            stats_set: ArrayStats::default(),
        }
    }

    pub fn ext_dtype(&self) -> &Arc<ExtDType> {
        let DType::Extension(ext) = &self.dtype else {
            unreachable!("ExtensionArray: dtype must be an ExtDType")
        };
        ext
    }

    pub fn storage(&self) -> &ArrayRef {
        &self.storage
    }

    #[allow(dead_code)]
    #[inline]
    pub fn id(&self) -> &ExtID {
        self.ext_dtype().id()
    }
}
