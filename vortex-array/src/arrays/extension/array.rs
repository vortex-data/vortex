// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
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
#[derive(Clone, Debug)]
pub struct ExtensionArray {
    pub(super) dtype: DType,
    pub(super) storage: ArrayRef,
    pub(super) stats_set: ArrayStats,
}

impl ExtensionArray {
    pub fn new(ext_dtype: ExtDTypeRef, storage: ArrayRef) -> Self {
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

    pub fn ext_dtype(&self) -> &ExtDTypeRef {
        let DType::Extension(ext) = &self.dtype else {
            unreachable!("ExtensionArray: dtype must be an ExtDType")
        };
        ext
    }

    pub fn storage(&self) -> &ArrayRef {
        &self.storage
    }

    #[inline]
    pub fn id(&self) -> ExtId {
        self.ext_dtype().id()
    }
}
