// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The extension dtype vtable for UUIDs.

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::PValue;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::Uuid;
use crate::UuidMetadata;
use crate::metadata::u8_to_version;

/// The number of bytes in a UUID.
pub(crate) const UUID_BYTE_LEN: usize = 16;

impl ExtVTable for Uuid {
    type Metadata = UuidMetadata;
    type NativeValue<'a> = uuid::Uuid;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.uuid");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        match metadata.version {
            None => Ok(Vec::new()),
            Some(v) => Ok(vec![v as u8]),
        }
    }

    fn deserialize_metadata(
        &self,
        metadata: &[u8],
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        let version = match metadata.len() {
            0 => None,
            1 => Some(u8_to_version(metadata[0])?),
            other => {
                vortex_bail!("UUID metadata must be 0 or 1 bytes, got {other}")
            }
        };

        Ok(UuidMetadata { version })
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "UUID_BYTE_LEN always fits u32"
    )]
    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            Nullability::NonNullable,
        )
    }

    fn unpack_native<'a>(
        ext_dtype: &ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let elements = storage_value.as_list();
        vortex_ensure_eq!(
            elements.len(),
            UUID_BYTE_LEN,
            "UUID scalar byte length mismatch"
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

            vortex_ensure_eq!(expected, actual, "UUID version mismatch");
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
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtDTypePlugin;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_error::VortexResult;

    use crate::Uuid;
    use crate::UuidMetadata;
    use crate::vtable::UUID_BYTE_LEN;

    #[rstest]
    #[case::no_version(None, &[])]
    #[case::v4_random(Some(Version::Random), &[4])]
    #[case::v7_sort_rand(Some(Version::SortRand), &[7])]
    #[case::nil(Some(Version::Nil), &[0])]
    #[case::max(Some(Version::Max), &[15])]
    fn roundtrip_metadata(
        #[case] version: Option<Version>,
        #[case] expected: &[u8],
    ) -> VortexResult<()> {
        let metadata = UuidMetadata { version };
        let bytes = Uuid.serialize_metadata(&metadata)?;
        assert_eq!(bytes, expected);
        let deserialized =
            Uuid.deserialize_metadata(&bytes, &uuid_storage_dtype(Nullability::NonNullable))?;
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
    fn read_accepts_declared_storage_dtype(#[case] nullability: Nullability) -> VortexResult<()> {
        Uuid.deserialize(&[], uuid_storage_dtype(nullability))?;
        Ok(())
    }

    /// A read fails unless the stored storage dtype is the declared one, including the
    /// nullability of the bytes, and the error names both dtypes.
    #[rstest]
    #[case::wrong_list_size(DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
        8,
        Nullability::NonNullable,
    ))]
    #[case::wrong_element_type(DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
        UUID_BYTE_LEN as u32,
        Nullability::NonNullable,
    ))]
    #[case::nullable_elements(DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::U8, Nullability::Nullable)),
        UUID_BYTE_LEN as u32,
        Nullability::NonNullable,
    ))]
    #[case::non_fsl(DType::Primitive(PType::U8, Nullability::NonNullable))]
    fn read_rejects_other_storage_dtype(#[case] storage_dtype: DType) {
        let expected =
            format!("vortex.uuid storage must be fixed_size_list(u8)[16], got {storage_dtype}");
        let result = Uuid.deserialize(&[], storage_dtype);
        assert!(result.is_err_and(|error| error.to_string().contains(&expected)));
    }

    #[test]
    fn unpack_native_uuid() -> VortexResult<()> {
        let expected = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")
            .map_err(|e| vortex_error::vortex_err!("{e}"))?;

        let ext_dtype = ExtDType::try_new(UuidMetadata::default(), Nullability::NonNullable)?;
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
            Nullability::NonNullable,
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
            Nullability::NonNullable,
        )?;
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
