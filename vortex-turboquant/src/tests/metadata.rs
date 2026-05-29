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
use crate::vtable::tq_storage_dtype;

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
    #[prost(uint32, repeated, tag = "6")]
    block_sizes: Vec<u32>,
}

#[rstest]
#[case::f16(PType::F16, vec![128])]
#[case::f32(PType::F32, vec![128])]
#[case::f64(PType::F64, vec![128])]
#[case::two_block(PType::F32, vec![512, 256])]
#[case::four_block(PType::F32, vec![512, 256, 64, 64])]
fn metadata_serialization_roundtrips(
    #[case] element_ptype: PType,
    #[case] block_sizes: Vec<u32>,
) -> VortexResult<()> {
    let dimensions = block_sizes.iter().sum::<u32>();
    let metadata = TurboQuantMetadata {
        element_ptype,
        dimensions,
        bit_width: 4,
        seed: 7,
        num_rounds: 3,
        block_sizes,
    };

    let encoded = TurboQuant.serialize_metadata(&metadata)?;
    let decoded = TurboQuant.deserialize_metadata(&encoded)?;

    assert_eq!(decoded, metadata);
    Ok(())
}

/// A pre-block / corrupt array whose on-the-wire `block_sizes` is empty (legacy) or sums below
/// `dimensions` must be rejected by `deserialize_metadata` with a clean error, never a panic. This
/// pins the documented on-disk format break. Built by corrupting a valid serialization so the test
/// does not depend on the `PType` wire discriminant.
#[rstest]
#[case::empty_legacy(vec![])]
#[case::sum_below_dimensions(vec![64])]
fn deserialize_rejects_malformed_block_sizes(#[case] block_sizes: Vec<u32>) -> VortexResult<()> {
    let valid = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 128,
        bit_width: 4,
        seed: 7,
        num_rounds: 3,
        block_sizes: vec![128],
    };
    let encoded = TurboQuant.serialize_metadata(&valid)?;
    let mut wire = MetadataWire::decode(encoded.as_slice())
        .map_err(|e| vortex_err!("decode MetadataWire: {e}"))?;
    wire.block_sizes = block_sizes;

    assert!(
        TurboQuant
            .deserialize_metadata(&wire.encode_to_vec())
            .is_err()
    );
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
        block_sizes: vec![128],
    };

    let encoded = TurboQuant.serialize_metadata(&metadata)?;
    let wire = MetadataWire::decode(encoded.as_slice())?;

    assert_eq!(wire.element_ptype, PType::F32 as i32);
    assert_eq!(wire.dimensions, 128);
    assert_eq!(wire.block_sizes, vec![128u32]);
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
        block_sizes: vec![128],
    };

    assert_eq!(
        metadata.to_string(),
        "element_ptype: f32, dimensions: 128, bit_width: 4, seed: 7, num_rounds: 3, \
         block_sizes: [128]"
    );
}

#[test]
fn dtype_validation_accepts_expected_storage() -> VortexResult<()> {
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 768,
        bit_width: 2,
        seed: 42,
        num_rounds: 3,
        block_sizes: vec![512, 256],
    };
    let storage = tq_storage_dtype(&metadata, Nullability::Nullable)?;

    ExtDType::<TurboQuant>::try_new(metadata, storage)?;
    Ok(())
}

#[test]
fn dtype_validation_accepts_nonnullable_storage() -> VortexResult<()> {
    let metadata = TurboQuantMetadata {
        element_ptype: PType::F32,
        dimensions: 768,
        bit_width: 2,
        seed: 42,
        num_rounds: 3,
        block_sizes: vec![512, 256],
    };
    let storage = tq_storage_dtype(&metadata, Nullability::NonNullable)?;

    ExtDType::<TurboQuant>::try_new(metadata, storage)?;
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
        block_sizes: vec![128],
    };
    // Outer struct fields do not match the expected `block_0` schema.
    let storage = DType::Struct(
        StructFields::new(
            FieldNames::from(["norms", "codes"]),
            vec![
                DType::Primitive(PType::F32, Nullability::Nullable),
                DType::FixedSizeList(
                    Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
                    128,
                    Nullability::Nullable,
                ),
            ],
        ),
        Nullability::Nullable,
    );

    assert!(ExtDType::<TurboQuant>::try_new(metadata, storage).is_err());
}
