// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::Array;
use crate::arrays::Extension;
use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::stats::ArrayStats;

/// The backing storage array for this extension array.
pub(super) const STORAGE_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["storage"];

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
pub struct ExtensionData {
    /// The storage dtype. This **must** be a [`Extension::DType`] variant.
    pub(super) dtype: DType,
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) stats_set: ArrayStats,
}

impl ExtensionData {
    /// Constructs a new `ExtensionArray`.
    ///
    /// # Panics
    ///
    /// Panics if the storage array in not compatible with the extension dtype.
    pub fn new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> Self {
        Self::try_new(ext_dtype, storage_array).vortex_expect("Failed to create `ExtensionArray`")
    }

    /// Tries to construct a new `ExtensionArray`.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage array in not compatible with the extension dtype.
    pub fn try_new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> VortexResult<Self> {
        // TODO(connor): Replace these statements once we add `validate_storage_array`.
        // ext_dtype.validate_storage_array(&storage_array)?;
        assert_eq!(
            ext_dtype.storage_dtype(),
            storage_array.dtype(),
            "ExtensionArray: storage_dtype must match storage array DType",
        );

        // SAFETY: we validate that the inputs are valid above.
        Ok(unsafe { Self::new_unchecked(ext_dtype, storage_array) })
    }

    /// Creates a new `ExtensionArray`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the storage array is compatible with the extension dtype. In
    /// other words, they must know that `ext_dtype.validate_storage_array(&storage_array)` has been
    /// called successfully on this storage array.
    pub unsafe fn new_unchecked(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> Self {
        // TODO(connor): Replace these statements once we add `validate_storage_array`.
        // #[cfg(debug_assertions)]
        // ext_dtype
        //     .validate_storage_array(&storage_array)
        //     .vortex_expect("[Debug Assertion]: Invalid storage array for `ExtensionArray`");
        debug_assert_eq!(
            ext_dtype.storage_dtype(),
            storage_array.dtype(),
            "ExtensionArray: storage_dtype must match storage array DType",
        );

        Self {
            dtype: DType::Extension(ext_dtype),
            slots: vec![Some(storage_array)],
            stats_set: ArrayStats::default(),
        }
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.storage_array().len()
    }

    /// Returns the [`DType`] of this array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The extension dtype of this array.
    pub fn ext_dtype(&self) -> &ExtDTypeRef {
        let DType::Extension(ext) = &self.dtype else {
            unreachable!("ExtensionArray: dtype must be an ExtDType")
        };

        ext
    }

    pub fn storage_array(&self) -> &ArrayRef {
        self.slots[STORAGE_SLOT]
            .as_ref()
            .vortex_expect("ExtensionArray storage slot")
    }
}

impl Array<Extension> {
    /// Constructs a new `ExtensionArray`.
    ///
    /// # Panics
    ///
    /// Panics if the storage array is not compatible with the extension dtype.
    pub fn new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> Self {
        Array::try_from_data(ExtensionData::new(ext_dtype, storage_array))
            .vortex_expect("ExtensionData is always valid")
    }

    /// Tries to construct a new `ExtensionArray`.
    pub fn try_new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> VortexResult<Self> {
        Array::try_from_data(ExtensionData::try_new(ext_dtype, storage_array)?)
    }
}
