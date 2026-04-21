// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::Extension;
use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;

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
    pub(super) ext_dtype: ExtDTypeRef,
}

impl Display for ExtensionData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ext_dtype: {}", self.ext_dtype)
    }
}

impl ExtensionData {
    /// Constructs a new `ExtensionArray`.
    ///
    /// # Panics
    ///
    /// Panics if the storage array in not compatible with the extension dtype.
    pub fn new(ext_dtype: ExtDTypeRef, storage_dtype: &DType) -> Self {
        Self::try_new(ext_dtype, storage_dtype).vortex_expect("Failed to create `ExtensionArray`")
    }

    /// Tries to construct a new `ExtensionArray`.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage array in not compatible with the extension dtype.
    pub fn try_new(ext_dtype: ExtDTypeRef, storage_dtype: &DType) -> VortexResult<Self> {
        // TODO(connor): Replace these statements once we add `validate_storage_array`.
        // ext_dtype.validate_storage_array(&storage_array)?;
        //
        // The storage array's outer nullability is allowed to differ from the extension's declared
        // storage outer nullability. Nested storage nullability must still match exactly.
        vortex_ensure!(
            storage_dtypes_match_ignoring_outer_nullability(
                ext_dtype.storage_dtype(),
                storage_dtype
            ),
            "ExtensionArray: storage_dtype must match storage array DType (ignoring outer \
             nullability only), got extension storage {} and array storage {}",
            ext_dtype.storage_dtype(),
            storage_dtype,
        );

        // SAFETY: we validate that the inputs are valid above.
        Ok(unsafe { Self::new_unchecked(ext_dtype, storage_dtype) })
    }

    /// Creates a new `ExtensionArray`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the storage array is compatible with the extension dtype. In
    /// other words, they must know that `ext_dtype.validate_storage_array(&storage_array)` has been
    /// called successfully on this storage array.
    pub unsafe fn new_unchecked(ext_dtype: ExtDTypeRef, storage_dtype: &DType) -> Self {
        // TODO(connor): Replace these statements once we add `validate_storage_array`.
        // #[cfg(debug_assertions)]
        // ext_dtype
        //     .validate_storage_array(&storage_array)
        //     .vortex_expect("[Debug Assertion]: Invalid storage array for `ExtensionArray`");
        //
        // Match the contract of [`Self::try_new`]: the storage dtype must match the extension's
        // declared storage dtype ignoring only outer nullability.
        debug_assert!(
            storage_dtypes_match_ignoring_outer_nullability(
                ext_dtype.storage_dtype(),
                storage_dtype
            ),
            "ExtensionArray: storage_dtype must match storage array DType (ignoring outer \
             nullability only), got extension storage {} and array storage {}",
            ext_dtype.storage_dtype(),
            storage_dtype,
        );

        Self { ext_dtype }
    }

    /// The extension dtype of this array.
    pub fn ext_dtype(&self) -> &ExtDTypeRef {
        &self.ext_dtype
    }
}

fn storage_dtypes_match_ignoring_outer_nullability(
    ext_storage_dtype: &DType,
    storage_dtype: &DType,
) -> bool {
    ext_storage_dtype.with_nullability(storage_dtype.nullability()) == *storage_dtype
}

pub trait ExtensionArrayExt: TypedArrayRef<Extension> {
    fn storage_array(&self) -> &ArrayRef {
        self.as_ref().slots()[STORAGE_SLOT]
            .as_ref()
            .vortex_expect("ExtensionArray storage slot")
    }
}
impl<T: TypedArrayRef<Extension>> ExtensionArrayExt for T {}

impl Array<Extension> {
    /// Constructs a new `ExtensionArray`.
    ///
    /// # Panics
    ///
    /// Panics if the storage array is not compatible with the extension dtype.
    pub fn new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> Self {
        let dtype = DType::Extension(ext_dtype.clone());
        let len = storage_array.len();
        let data = ExtensionData::new(ext_dtype, storage_array.dtype());
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Extension, dtype, len, data).with_slots(vec![Some(storage_array)]),
            )
        }
    }

    /// Tries to construct a new `ExtensionArray`.
    pub fn try_new(ext_dtype: ExtDTypeRef, storage_array: ArrayRef) -> VortexResult<Self> {
        let dtype = DType::Extension(ext_dtype.clone());
        let len = storage_array.len();
        let data = ExtensionData::try_new(ext_dtype, storage_array.dtype())?;
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Extension, dtype, len, data).with_slots(vec![Some(storage_array)]),
            )
        })
    }

    /// Creates a new [`ExtensionArray`](crate::arrays::ExtensionArray) from a vtable, metadata, and
    /// a storage array.
    pub fn try_new_from_vtable<V: ExtVTable>(
        vtable: V,
        metadata: V::Metadata,
        storage_array: ArrayRef,
    ) -> VortexResult<Self> {
        let ext_dtype =
            ExtDType::<V>::try_with_vtable(vtable, metadata, storage_array.dtype().clone())?
                .erased();
        Self::try_new(ext_dtype, storage_array)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::Buffer;

    use super::*;
    use crate::IntoArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtId;
    use crate::extension::EmptyMetadata;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct TestExt;

    impl ExtVTable for TestExt {
        type Metadata = EmptyMetadata;
        type NativeValue<'a> = &'a ScalarValue;

        fn id(&self) -> ExtId {
            ExtId::new("vortex.test.extension")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
            Ok(EmptyMetadata)
        }

        fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
            Ok(())
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok(storage_value)
        }
    }

    fn fsl_dtype(element_nullability: Nullability, list_nullability: Nullability) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::F32, element_nullability)),
            2,
            list_nullability,
        )
    }

    #[test]
    fn extension_storage_allows_outer_nullability_mismatch() -> VortexResult<()> {
        let ext_dtype = ExtDType::<TestExt>::try_new(
            EmptyMetadata,
            fsl_dtype(Nullability::NonNullable, Nullability::NonNullable),
        )?
        .erased();

        let elements = PrimitiveArray::from_iter([1.0f32, 0.0]).into_array();
        let storage = FixedSizeListArray::try_new(elements, 2, Validity::AllValid, 1)?.into_array();

        ExtensionArray::try_new(ext_dtype, storage)?;
        Ok(())
    }

    #[test]
    fn extension_storage_rejects_nested_nullability_mismatch() -> VortexResult<()> {
        let ext_dtype = ExtDType::<TestExt>::try_new(
            EmptyMetadata,
            fsl_dtype(Nullability::NonNullable, Nullability::NonNullable),
        )?
        .erased();

        let elements =
            PrimitiveArray::new(Buffer::copy_from([1.0f32, 0.0]), Validity::AllValid).into_array();
        let storage =
            FixedSizeListArray::try_new(elements, 2, Validity::NonNullable, 1)?.into_array();

        assert!(ExtensionArray::try_new(ext_dtype, storage).is_err());
        Ok(())
    }
}
