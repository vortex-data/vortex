// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use uuid;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::EmptyMetadata;
use crate::extension::uuid::Uuid;
use crate::scalar::PValue;
use crate::scalar::ScalarValue;

/// The number of bytes in a UUID.
pub(crate) const UUID_BYTE_LEN: usize = 16;

impl ExtVTable for Uuid {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = uuid::Uuid;

    fn id(&self) -> ExtId {
        ExtId::new_ref("vortex.uuid")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(
        &self,
        _metadata: &Self::Metadata,
        storage_dtype: &DType,
    ) -> VortexResult<()> {
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
        &self,
        _metadata: &'a Self::Metadata,
        _storage_dtype: &'a DType,
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

        Ok(uuid::Uuid::from_bytes(bytes))
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
    use vortex_error::VortexResult;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtVTable;
    use crate::extension::EmptyMetadata;
    use crate::extension::uuid::Uuid;
    use crate::extension::uuid::vtable::UUID_BYTE_LEN;
    use crate::scalar::Scalar;

    #[test]
    fn roundtrip_metadata() -> VortexResult<()> {
        let vtable = Uuid;
        let bytes = vtable.serialize_metadata(&EmptyMetadata)?;
        let deserialized = vtable.deserialize_metadata(&bytes)?;
        assert_eq!(deserialized, EmptyMetadata);
        Ok(())
    }

    #[rstest]
    #[case::non_nullable(Nullability::NonNullable)]
    #[case::nullable(Nullability::Nullable)]
    fn validate_correct_storage_dtype(#[case] nullability: Nullability) -> VortexResult<()> {
        let storage_dtype = uuid_storage_dtype(nullability);
        Uuid.validate_dtype(&EmptyMetadata, &storage_dtype)
    }

    #[test]
    fn validate_rejects_wrong_list_size() {
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            8,
            Nullability::NonNullable,
        );
        assert!(Uuid.validate_dtype(&EmptyMetadata, &storage_dtype).is_err());
    }

    #[test]
    fn validate_rejects_wrong_element_type() {
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U64, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            Nullability::NonNullable,
        );
        assert!(Uuid.validate_dtype(&EmptyMetadata, &storage_dtype).is_err());
    }

    #[test]
    fn validate_rejects_nullable_elements() {
        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::Nullable)),
            UUID_BYTE_LEN as u32,
            Nullability::NonNullable,
        );
        assert!(Uuid.validate_dtype(&EmptyMetadata, &storage_dtype).is_err());
    }

    #[test]
    fn validate_rejects_non_fsl() {
        let storage_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        assert!(Uuid.validate_dtype(&EmptyMetadata, &storage_dtype).is_err());
    }

    #[test]
    fn unpack_native_uuid() -> VortexResult<()> {
        let expected = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000")
            .map_err(|e| vortex_error::vortex_err!("{e}"))?;

        let storage_dtype = uuid_storage_dtype(Nullability::NonNullable);
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
        let result = Uuid.unpack_native(&EmptyMetadata, &storage_dtype, storage_value)?;
        assert_eq!(result, expected);
        assert_eq!(result.to_string(), "550e8400-e29b-41d4-a716-446655440000");
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
