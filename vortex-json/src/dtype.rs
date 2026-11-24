// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension dtype definition for the JSON type

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
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
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = &'a str;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.json");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        vortex_ensure!(metadata.is_empty(), "JSON metadata must be empty");
        Ok(EmptyMetadata)
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DType::Utf8(Nullability::NonNullable)
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

#[cfg(test)]
mod tests {
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::compute::conformance::extension::test_ext_vtable_conformance;
    use vortex_array::dtype::Claim;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::Json;

    #[test]
    fn json_claims_nothing() -> VortexResult<()> {
        let ext_dtype = ExtDType::<Json>::try_new(EmptyMetadata, Nullability::Nullable)?;
        let storage = VarBinViewArray::from_iter_nullable_str([
            Some(r#"{"a":1,"b":2}"#),
            None,
            Some(r#"{"b":2,"a":1}"#),
        ])
        .into_array();
        test_ext_vtable_conformance(
            &ext_dtype,
            &storage,
            None,
            &mut VortexSession::empty().create_execution_ctx(),
        );

        let dtype = DType::Extension(ext_dtype.erased());
        assert!(!dtype.holds(Claim::Eq));
        assert!(!dtype.holds(Claim::Ord));

        Ok(())
    }
}
