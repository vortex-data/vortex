// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The protobuf extension dtype.

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::ProtobufMetadata;
use crate::ProtobufValue;

/// Protobuf messages of one schema, stored as a struct with one field per message field.
///
/// The [`ProtobufMetadata`] names the message type, the storage mapping version and the mapping
/// options. The crate documentation describes the mapping.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Protobuf;

impl Protobuf {
    /// The extension ID, `vortex.protobuf`.
    pub const ID: &'static str = "vortex.protobuf";
}

impl ExtVTable for Protobuf {
    type Metadata = ProtobufMetadata;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = ProtobufValue<'a>;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new(Protobuf::ID);
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.serialize())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        ProtobufMetadata::deserialize(metadata)
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(metadata: &Self::Metadata) -> DType {
        metadata.plan().dtype(Nullability::NonNullable)
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        ProtobufValue::try_new(ext_dtype.metadata(), storage_value)
    }
}
