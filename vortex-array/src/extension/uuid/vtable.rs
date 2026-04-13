// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use uuid;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::uuid::Uuid;
use crate::extension::uuid::UuidMetadata;
use crate::extension::uuid::metadata::u8_to_version;
use crate::scalar::PValue;
use crate::scalar::ScalarValue;

/// The number of bytes in a UUID.
pub(crate) const UUID_BYTE_LEN: usize = 16;

impl ExtVTable for Uuid {
    type Metadata = UuidMetadata;
    type NativeValue<'a> = uuid::Uuid;

    fn id(&self) -> ExtId {
        ExtId::new("vortex.uuid")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        match metadata.version {
            None => Ok(Vec::new()),
            Some(v) => Ok(vec![v as u8]),
        }
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        let version = match metadata.len() {
            0 => None,
            1 => Some(u8_to_version(metadata[0])?),
            other => vortex_bail!("UUID metadata must be 0 or 1 bytes, got {other}"),
        };

        Ok(UuidMetadata { version })
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        let storage_dtype = ext_dtype.storage_dtype();
        let DType::FixedSizeList(element_dtype, list_size, _nullability) = storage_dtype else {
            vortex_bail!("UUID storage dtype must be a FixedSizeList, got {storage_dtype}");
        };

        vortex_ensure_eq!(
            *list_size as usize,
            UUID_BYTE_LEN,
            "UUID storage FixedSizeList must have size {UUID_BYTE_LEN}, got {list_size}"
        );

        let DType::Primitive(ptype, elem_nullability) = element_dtype.as_ref() else {
            vortex_bail!("UUID element dtype must be Primitive(U8), got {element_dtype}");
        };

        vortex_ensure_eq!(
            *ptype,
            PType::U8,
            "UUID element dtype must be U8, got {ptype}"
        );
        vortex_ensure!(
            !elem_nullability.is_nullable(),
            "UUID element dtype must be non-nullable"
        );

        Ok(())
    }

    fn unpack_native<'a>(
        ext_dtype: &ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let elements = storage_value.as_list();
        vortex_ensure_eq!(
            elements.len(),
            UUID_BYTE_LEN,
            "UUID scalar must have exactly {UUID_BYTE_LEN} bytes, got {}",
            elements.len()
        );

        let mut bytes = [0u8; UUID_BYTE_LEN];
        for (i, elem) in elements.iter().enumerate() {
            let Some(scalar_value) = elem else {
                vortex_bail!("UUID byte at index {i} must not be null");
            };
            let PValue::U8(b) = scalar_value.as_primitive() else {
                vortex_bail!("UUID byte at index {i} must be U8");
            };
            bytes[i] = *b;
        }

        let parsed = uuid::Uuid::from_bytes(bytes);

        // Verify the parsed UUID matches the expected version, if one is set.
        if let Some(expected) = ext_dtype.metadata().version {
            let expected = expected as u8;
            let actual = parsed
                .get_version()
                .ok_or_else(|| vortex_err!("UUID has unrecognized version nibble"))?
                as u8;

            vortex_ensure_eq!(
                expected,
                actual,
                "UUID version mismatch: expected v{expected}, got v{actual}",
            );
        }

        Ok(parsed)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "UUID_BYTE_LEN always fits both usize and u32"
)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use uuid::Version;
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtVTable;
    use crate::extension::uuid::Uuid;
    use crate::extension::uuid::UuidMetadata;
    use crate::extension::uuid::vtable::UUID_BYTE_LEN;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[rstest]
    #[case::no_version(None)]
    #[case::v4_random(Some(Version::Random))]
    #[case::v7_sort_rand(Some(Version::SortRand))]
    #[case::nil(Some(Version::Nil))]
    #[case::max(Some(Version::Max))]
    fn roundtrip_metadata(#[case] version: Option<Version>) -> VortexResult<()> {
        let metadata = UuidMetadata { version };
        let bytes = Uuid.serialize_metadata(&metadata)?;
        let expected_len = if version.is_none() { 0 } else { 1 };
        assert_eq!(bytes.len(), expected_len);
        let deserialized = Uuid.deserialize_metadata(&bytes)?;
        assert_eq!(deserialized, metadata);
        Ok(())
    }

    #[test]
    fn metadata_display_no_version() {
        let metadata = UuidMetadata { version: None };
        assert_eq!(metadata.to_string(), "");
    }

    #[test]
    fn metadata_display_with_version() {
        let metadata = UuidMetadata {
            version: Some(Version::Random),
        };
        assert_eq!(metadata.to_string(), "v4");

        let metadata = UuidMetadata {
            version: Some(Version::SortRand),
        };
        assert_eq!(metadata.to_string(), "v7");
    }

    #[rstest]
    #[case::non_nullable(Nullability::NonNullable)]
    #[case::nullable(Nullability::Nullable)]
    fn validate_correct_storage_dtype(#[case] nullability: Nullability) -> VortexResult<()> {
        let metadata = UuidMetadata::default();
        let storage_dtype = uuid_storage_dtype(nullability);
        ExtDType::try_with_vtable(Uuid, metadata, storage_dtype)?;
        Ok(())
    }

    #[test]
    fn validate_rejects_wrong_list_size() {
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            8,
            Nullability::NonNullable,
        );
        assert!(ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage_dtype).is_err());
    }

    #[test]
    fn validate_rejects_wrong_element_type() {
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            Nullability::NonNullable,
        );
        assert!(ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage_dtype).is_err());
    }

    #[test]
    fn validate_rejects_nullable_elements() {
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::Nullable)),
            UUID_BYTE_LEN as u32,
            Nullability::NonNullable,
        );
        assert!(ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage_dtype).is_err());
    }

    #[test]
    fn validate_rejects_non_fsl() {
        let storage_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        assert!(ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage_dtype).is_err());
    }

    #[test]
    fn unpack_native_uuid() -> VortexResult<()> {
        let expected = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")
            .map_err(|e| vortex_error::vortex_err!("{e}"))?;

        let ext_dtype = ExtDType::try_new(
            UuidMetadata::default(),
            uuid_storage_dtype(Nullability::NonNullable),
        )?;
        let children: Vec<Scalar> = expected
            .as_bytes()
            .iter()
            .map(|&b| Scalar::primitive(b, Nullability::NonNullable))
            .collect();
        let storage_scalar = Scalar::fixed_size_list(
            DType::Primitive(PType::U8, Nullability::NonNullable),
            children,
            Nullability::NonNullable,
        );

        let storage_value = storage_scalar
            .value()
            .ok_or_else(|| vortex_error::vortex_err!("expected non-null scalar"))?;
        let result = Uuid::unpack_native(&ext_dtype, storage_value)?;
        assert_eq!(result, expected);
        assert_eq!(result.to_string(), "550e8400-e29b-41d4-a716-446655440000");
        Ok(())
    }

    #[test]
    fn unpack_native_rejects_version_mismatch() -> VortexResult<()> {
        // This is a v4 UUID.
        let v4_uuid = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")
            .map_err(|e| vortex_error::vortex_err!("{e}"))?;
        assert_eq!(v4_uuid.get_version(), Some(Version::Random));

        // Metadata says v7, but the UUID is v4.
        let ext_dtype = ExtDType::try_with_vtable(
            Uuid,
            UuidMetadata {
                version: Some(Version::SortRand),
            },
            uuid_storage_dtype(Nullability::NonNullable),
        )?;
        let children: Vec<Scalar> = v4_uuid
            .as_bytes()
            .iter()
            .map(|&b| Scalar::primitive(b, Nullability::NonNullable))
            .collect();
        let storage_scalar = Scalar::fixed_size_list(
            DType::Primitive(PType::U8, Nullability::NonNullable),
            children,
            Nullability::NonNullable,
        );

        let storage_value = storage_scalar
            .value()
            .ok_or_else(|| vortex_error::vortex_err!("expected non-null scalar"))?;
        assert!(Uuid::unpack_native(&ext_dtype, storage_value).is_err());
        Ok(())
    }

    /// Builds a [`ScalarValue`] for a UUID's 16 bytes, suitable for passing to `unpack_native`.
    fn uuid_storage_scalar(uuid: &uuid::Uuid) -> ScalarValue {
        let children: Vec<Scalar> = uuid
            .as_bytes()
            .iter()
            .map(|&b| Scalar::primitive(b, Nullability::NonNullable))
            .collect();
        let scalar = Scalar::fixed_size_list(
            DType::Primitive(PType::U8, Nullability::NonNullable),
            children,
            Nullability::NonNullable,
        );
        scalar.value().unwrap().clone()
    }

    #[test]
    fn unpack_native_accepts_matching_version() -> VortexResult<()> {
        // This is a v4 UUID.
        let v4_uuid = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")
            .map_err(|e| vortex_error::vortex_err!("{e}"))?;

        let ext_dtype = ExtDType::try_new(
            UuidMetadata {
                version: Some(Version::Random),
            },
            uuid_storage_dtype(Nullability::NonNullable),
        )
        .unwrap();
        let storage_value = uuid_storage_scalar(&v4_uuid);

        let result = Uuid::unpack_native(&ext_dtype, &storage_value)?;
        assert_eq!(result, v4_uuid);
        Ok(())
    }

    #[test]
    fn unpack_native_any_version_accepts_all() -> VortexResult<()> {
        // A v4 UUID should be accepted when metadata has no version constraint.
        let v4_uuid = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")
            .map_err(|e| vortex_error::vortex_err!("{e}"))?;

        let ext_dtype = ExtDType::try_new(
            UuidMetadata::default(),
            uuid_storage_dtype(Nullability::NonNullable),
        )
        .unwrap();
        let storage_value = uuid_storage_scalar(&v4_uuid);

        let result = Uuid::unpack_native(&ext_dtype, &storage_value)?;
        assert_eq!(result, v4_uuid);
        Ok(())
    }

    fn uuid_storage_dtype(nullability: Nullability) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            nullability,
        )
    }
}
