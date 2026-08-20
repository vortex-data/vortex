// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension dtype definition for the JSON type

use vortex_array::EmptyMetadata;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

/// JSON logical type backed by UTF-8 string storage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Json;

impl ExtVTable for Json {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = &'a str;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.json");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_ensure!(metadata.is_empty(), "JSON metadata must be empty");
        Ok(EmptyMetadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(
            ext_dtype.storage_dtype().is_utf8(),
            "JSON storage dtype must be utf8, got {}",
            ext_dtype.storage_dtype()
        );
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let ScalarValue::Utf8(value) = storage_value else {
            vortex_bail!("JSON storage scalar must be utf8, got {storage_value}");
        };
        Ok(value.as_str())
    }
}
