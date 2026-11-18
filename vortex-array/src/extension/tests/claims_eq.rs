// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An equality-only extension fixture with a storage dtype supplied by the test.

use vortex_error::VortexResult;

use crate::DeserializeMetadata;
use crate::EmptyMetadata;
use crate::dtype::Claim;
use crate::dtype::Claims;
use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;

/// Declares equality even when the supplied storage lacks it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct ClaimsEq;

impl ExtVTable for ClaimsEq {
    type Metadata = DType;
    type StoredMetadata = EmptyMetadata;
    type NativeValue<'a> = &'a ScalarValue;

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    fn id(&self) -> ExtId {
        ExtId::new("test.claims_eq")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::StoredMetadata> {
        EmptyMetadata::deserialize(bytes)
    }

    fn metadata_from_storage(
        &self,
        _stored_metadata: Self::StoredMetadata,
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(storage_dtype.clone())
    }

    fn storage_dtype(metadata: &Self::Metadata) -> DType {
        metadata.clone()
    }

    fn claims(_metadata: &Self::Metadata) -> Claims {
        Claims::none().with(Claim::Eq)
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(storage_value)
    }
}
