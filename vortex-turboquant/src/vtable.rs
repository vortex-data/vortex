// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use prost::Message;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::TurboQuantConfig;
use crate::config::MIN_DIMENSION;
use crate::vector::storage::CODES_FIELD;
use crate::vector::storage::NORMS_FIELD;
use crate::vector::tq_padded_dim;

/// TurboQuant logical extension type. Per-array configuration lives in [`TurboQuantMetadata`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TurboQuant;

/// Serialized metadata for a TurboQuant extension array. The fields together suffice to
/// reconstruct the SORF transform and centroid codebook at decode time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TurboQuantMetadata {
    /// Original vector element ptype and stored row-norm ptype. Restricted to `f16` / `f32` /
    /// `f64`.
    pub element_ptype: PType,
    /// Original vector dimension before SORF padding to the next power of two.
    pub dimensions: u32,
    /// Bits per coordinate in the scalar quantizer codebook (`1..=8`).
    pub bit_width: u8,
    /// Seed used to derive the deterministic SORF transform.
    pub seed: u64,
    /// Number of sign-diagonal plus Walsh-Hadamard rounds in the SORF transform.
    pub num_rounds: u8,
}

impl ExtVTable for TurboQuant {
    type Metadata = TurboQuantMetadata;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        ExtId::new("vortex.turboquant")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        validate_tq_metadata(metadata)?;

        let proto = TurboQuantMetadataProto {
            element_ptype: metadata.element_ptype as i32,
            dimensions: metadata.dimensions,
            bit_width: u32::from(metadata.bit_width),
            seed: metadata.seed,
            num_rounds: u32::from(metadata.num_rounds),
        };

        Ok(proto.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        let proto = TurboQuantMetadataProto::decode(metadata)
            .map_err(|e| vortex_err!("Failed to decode TurboQuantMetadata: {e}"))?;
        let bit_width = u8::try_from(proto.bit_width)
            .map_err(|_| vortex_err!("TurboQuant bit_width does not fit u8"))?;
        let num_rounds = u8::try_from(proto.num_rounds)
            .map_err(|_| vortex_err!("TurboQuant num_rounds does not fit u8"))?;
        let element_ptype = PType::try_from(proto.element_ptype).map_err(|e| {
            vortex_err!(
                "invalid TurboQuant metadata element_ptype code {}: {e}",
                proto.element_ptype
            )
        })?;

        let metadata = TurboQuantMetadata {
            element_ptype,
            dimensions: proto.dimensions,
            bit_width,
            seed: proto.seed,
            num_rounds,
        };
        validate_tq_metadata(&metadata)?;

        Ok(metadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        validate_tq_metadata(ext_dtype.metadata())?;
        validate_tq_storage_dtype(ext_dtype.metadata(), ext_dtype.storage_dtype())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(storage_value)
    }
}

/// Wire-format representation of [`TurboQuantMetadata`]. Field tags MUST NOT change once
/// shipped; new fields must use unused tags and remain optional.
#[derive(Clone, PartialEq, Message)]
struct TurboQuantMetadataProto {
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

/// Extract TurboQuant metadata from a dtype.
///
/// Returns an error when the dtype is not the TurboQuant extension type.
pub(crate) fn tq_metadata(dtype: &DType) -> VortexResult<TurboQuantMetadata> {
    let ext_dtype = dtype
        .as_extension_opt()
        .ok_or_else(|| vortex_err!("expected a TurboQuant extension array, got {dtype}"))?;

    let metadata = ext_dtype
        .metadata_opt::<TurboQuant>()
        .ok_or_else(|| vortex_err!("expected a TurboQuant extension array, got {dtype}"))?;

    Ok(*metadata)
}

pub(crate) fn tq_storage_dtype(
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

/// Validate [`TurboQuantMetadata`] invariants. Called on both serialize and deserialize so a
/// corrupted on-disk metadata block errors out rather than decoding into nonsense.
fn validate_tq_metadata(metadata: &TurboQuantMetadata) -> VortexResult<()> {
    vortex_ensure!(
        metadata.dimensions >= MIN_DIMENSION,
        "TurboQuant dimensions must be >= {MIN_DIMENSION}, got {}",
        metadata.dimensions
    );
    vortex_ensure!(
        metadata.element_ptype.is_float(),
        "TurboQuant element_ptype must be a float, got {:?}",
        metadata.element_ptype
    );

    tq_padded_dim(metadata.dimensions)?;

    TurboQuantConfig::try_new(metadata.bit_width, metadata.seed, metadata.num_rounds).map(|_| ())
}

/// Validate that `dtype` matches the storage shape produced by [`tq_storage_dtype`] for
/// `metadata`. Called from [`TurboQuant::validate_dtype`].
fn validate_tq_storage_dtype(metadata: &TurboQuantMetadata, dtype: &DType) -> VortexResult<()> {
    let DType::Struct(fields, _) = dtype else {
        vortex_bail!("TurboQuant storage dtype must be a Struct, got {dtype}");
    };
    let expected_names = FieldNames::from([NORMS_FIELD, CODES_FIELD]);
    vortex_ensure_eq!(
        fields.names(),
        &expected_names,
        "TurboQuant storage fields must be {expected_names}, got {}",
        fields.names()
    );

    let Some(norms_dtype) = fields.field(NORMS_FIELD) else {
        vortex_bail!("TurboQuant storage missing {NORMS_FIELD} field");
    };
    let DType::Primitive(norms_ptype, _) = norms_dtype else {
        vortex_bail!("TurboQuant {NORMS_FIELD} field must be primitive, got {norms_dtype}");
    };
    vortex_ensure_eq!(
        norms_ptype,
        metadata.element_ptype,
        "TurboQuant {NORMS_FIELD} ptype must be {}, got {norms_ptype}",
        metadata.element_ptype
    );

    let Some(codes_dtype) = fields.field(CODES_FIELD) else {
        vortex_bail!("TurboQuant storage missing {CODES_FIELD} field");
    };
    let DType::FixedSizeList(element_dtype, list_size, _) = codes_dtype else {
        vortex_bail!("TurboQuant {CODES_FIELD} field must be fixed-size-list, got {codes_dtype}");
    };
    let padded_dim = u32::try_from(tq_padded_dim(metadata.dimensions)?)
        .map_err(|_| vortex_err!("TurboQuant padded dimension does not fit u32"))?;
    vortex_ensure_eq!(
        list_size,
        padded_dim,
        "TurboQuant {CODES_FIELD} list size must be {padded_dim}, got {list_size}"
    );
    vortex_ensure_eq!(
        element_dtype.as_ref(),
        &DType::Primitive(PType::U8, Nullability::NonNullable),
        "TurboQuant {CODES_FIELD} elements must be non-nullable u8, got {element_dtype}"
    );

    Ok(())
}

impl fmt::Display for TurboQuantMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "element_ptype: {}, dimensions: {}, bit_width: {}, seed: {}, num_rounds: {}",
            self.element_ptype, self.dimensions, self.bit_width, self.seed, self.num_rounds
        )
    }
}
