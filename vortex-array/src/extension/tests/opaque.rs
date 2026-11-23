// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An extension fixture for testing operations that require an explicit claim.

use vortex_error::VortexResult;

use crate::DeserializeMetadata;
use crate::EmptyMetadata;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;

/// An extension type over `i64` with no claims about its storage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Opaque;

impl ExtVTable for Opaque {
    type Metadata = EmptyMetadata;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = i64;

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    fn id(&self) -> ExtId {
        ExtId::new("test.opaque")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::StoredMetadata> {
        EmptyMetadata::deserialize(bytes)
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DType::Primitive(PType::I64, Nullability::NonNullable)
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        storage_value.as_primitive().cast::<i64>()
    }
}
