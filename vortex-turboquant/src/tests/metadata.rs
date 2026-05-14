// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use prost::Message;
use rstest::rstest;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::TurboQuant;
use crate::TurboQuantMetadata;
use crate::vector::storage::CODES_FIELD;
use crate::vector::storage::NORMS_FIELD;
use crate::vector::tq_padded_dim;

#[derive(Clone, PartialEq, Message)]
struct MetadataWire {
    #[prost(enumeration = "PType", tag = "1")]
    element_ptype: i32,
    #[prost(uint32, tag = "2")]
    dimensions: u32,
    #[prost(uint32, tag = "3")]
    bit_width: u32,
    #[prost(uint64, tag = "4")]
    seed: u64,
    #[prost(uint32, tag = "5")]
    num_rounds: u32,
}

fn tq_storage_dtype(
    metadata: &TurboQuantMetadata,
    row_nullability: Nullability,
) -> VortexResult<DType> {
    let padded_dim = u32::try_from(tq_padded_dim(metadata.dimensions)?)
        .map_err(|_| vortex_err!("TurboQuant padded dimension does not fit u32"))?;
    Ok(DType::Struct(
        StructFields::new(
            FieldNames::from([NORMS_FIELD, CODES_FIELD]),
            vec![
                DType::Primitive(metadata.element_ptype, row_nullability),
                DType::FixedSizeList(
                    Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
                    padded_dim,
                    row_nullability,
                ),
            ],
        ),
        row_nullability,
    ))
}

#[rstest]
#[case::f16(PType::F16)]
#[case::f32(PType::F32)]
#[case::f64(PType::F64)]
fn metadata_serialization_roundtrips(#[case] element_ptype: PType) -> VortexResult<()> {
    let metadata = TurboQuantMetadata {
        element_ptype,
        dimensions: 128,
        bit_width: 4,
        seed: 7,
        num_rounds: 3,
    };

    let encoded = TurboQuant.serialize_metadata(&metadata)?;
    let decoded = TurboQuant.deserialize_metadata(&encoded)?;

    assert_eq!(decoded, metadata);
    Ok(())
}

#[test]
fn metadata_serialization_uses_ptype_discriminants() -> VortexResult<()> {
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 128,
        bit_width: 4,
        seed: 7,
        num_rounds: 3,
    };

    let encoded = TurboQuant.serialize_metadata(&metadata)?;
    let wire = MetadataWire::decode(encoded.as_slice())?;

    assert_eq!(wire.element_ptype, PType::F32 as i32);
    assert_eq!(wire.dimensions, 128);
    Ok(())
}

#[test]
fn metadata_display_matches_field_order() {
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 128,
        bit_width: 4,
        seed: 7,
        num_rounds: 3,
    };

    assert_eq!(
        metadata.to_string(),
        "element_ptype: f32, dimensions: 128, bit_width: 4, seed: 7, num_rounds: 3"
    );
}

#[test]
fn dtype_validation_accepts_expected_storage() -> VortexResult<()> {
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 129,
        bit_width: 2,
        seed: 42,
        num_rounds: 3,
    };

    ExtDType::<TurboQuant>::try_new(
        metadata,
        tq_storage_dtype(&metadata, Nullability::Nullable)?,
    )?;
    Ok(())
}

#[test]
fn dtype_validation_accepts_nonnullable_storage() -> VortexResult<()> {
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 129,
        bit_width: 2,
        seed: 42,
        num_rounds: 3,
    };

    ExtDType::<TurboQuant>::try_new(
        metadata,
        tq_storage_dtype(&metadata, Nullability::NonNullable)?,
    )?;
    Ok(())
}

#[test]
fn dtype_validation_rejects_malformed_storage() {
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 128,
        bit_width: 2,
        seed: 42,
        num_rounds: 3,
    };
    let storage = DType::Struct(
        StructFields::new(
            FieldNames::from(["norms", "codes"]),
            vec![
                DType::Primitive(PType::F32, Nullability::Nullable),
                DType::FixedSizeList(
                    DType::Primitive(PType::U8, Nullability::Nullable).into(),
                    128,
                    Nullability::NonNullable,
                ),
            ],
        ),
        Nullability::Nullable,
    );

    assert!(ExtDType::<TurboQuant>::try_new(metadata, storage).is_err());
}
