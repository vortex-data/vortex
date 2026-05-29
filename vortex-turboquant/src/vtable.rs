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
use crate::config::validate_block_shape;
use crate::config::validate_block_sum;
use crate::vector::storage::CODES_FIELD;
use crate::vector::storage::NORMS_FIELD;
use crate::vector::storage::block_field_name;

/// TurboQuant logical extension type. Per-array configuration lives in [`TurboQuantMetadata`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TurboQuant;

/// Serialized metadata for a TurboQuant extension array. The fields together suffice to reconstruct
/// the SORF transforms, centroid codebooks, and storage layout at decode time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TurboQuantMetadata {
    /// Original vector element ptype and stored row-norm ptype. Restricted to `f16`/`f32`/`f64`.
    pub element_ptype: PType,

    /// Original vector dimension before block decomposition.
    pub dimensions: u32,

    /// Bits per coordinate in the scalar quantizer codebook (`1..=8`).
    pub bit_width: u8,

    /// Global seed used to derive each block's deterministic SORF transform.
    pub seed: u64,

    /// Number of sign-diagonal plus Walsh-Hadamard rounds in each block's SORF transform.
    pub num_rounds: u8,

    /// Powers-of-two block sizes the encoder used.
    ///
    /// Note that this is always non-empty. Additionally, `sum(block_sizes) >= dimensions` and each
    /// entry is at least `MIN_BLOCK_SIZE`.
    pub block_sizes: Vec<u32>,
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
            bit_width: metadata.bit_width as u32,
            seed: metadata.seed,
            num_rounds: metadata.num_rounds as u32,
            block_sizes: metadata.block_sizes.clone(),
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
            // Block decomposition intentionally breaks the pre-block on-disk format: arrays
            // written before this field existed decode to an empty `block_sizes`, which
            // `validate_tq_metadata` rejects below with a clean error (not a panic). There is no
            // backward-compatibility shim because the TurboQuant on-disk format is not yet stable.
            block_sizes: proto.block_sizes,
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
/// shipped; new fields must use unused tags.
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
    #[prost(uint32, repeated, tag = "6")]
    block_sizes: Vec<u32>,
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

    Ok(metadata.clone())
}

/// Construct the storage dtype for a given metadata and row nullability.
///
/// Produces an outer struct of `metadata.block_sizes.len()` inner `Struct { norms, codes }` fields,
/// each parameterized by its own block size.
pub(crate) fn tq_storage_dtype(
    metadata: &TurboQuantMetadata,
    row_nullability: Nullability,
) -> VortexResult<DType> {
    let mut names = Vec::with_capacity(metadata.block_sizes.len());
    let mut fields = Vec::with_capacity(metadata.block_sizes.len());

    for (index, &block_size) in metadata.block_sizes.iter().enumerate() {
        names.push(block_field_name(index));
        fields.push(inner_block_dtype(
            metadata.element_ptype,
            block_size,
            row_nullability,
        ));
    }

    Ok(DType::Struct(
        StructFields::new(FieldNames::from_iter(names), fields),
        row_nullability,
    ))
}

/// The struct type for each block.
///
/// Note that we propagate the nullability through both the fields and the outer struct itself for
/// simplicity.
fn inner_block_dtype(element_ptype: PType, block_size: u32, row_nullability: Nullability) -> DType {
    DType::Struct(
        StructFields::new(
            FieldNames::from([NORMS_FIELD, CODES_FIELD]),
            vec![
                DType::Primitive(element_ptype, row_nullability),
                DType::FixedSizeList(
                    Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
                    block_size,
                    row_nullability,
                ),
            ],
        ),
        row_nullability,
    )
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

    validate_block_shape(&metadata.block_sizes)?;
    validate_block_sum(&metadata.block_sizes, metadata.dimensions)?;

    TurboQuantConfig::try_new(
        metadata.bit_width,
        metadata.seed,
        metadata.num_rounds,
        Some(metadata.block_sizes.clone()),
    )
    .map(|_| ())
}

/// Validate that `dtype` matches the storage shape produced by [`tq_storage_dtype`] for
/// `metadata`. Called from [`TurboQuant::validate_dtype`].
fn validate_tq_storage_dtype(metadata: &TurboQuantMetadata, dtype: &DType) -> VortexResult<()> {
    let DType::Struct(outer_fields, _) = dtype else {
        vortex_bail!("TurboQuant storage dtype must be a Struct, got {dtype}");
    };

    let expected_names: Vec<_> = (0..metadata.block_sizes.len())
        .map(block_field_name)
        .collect();
    vortex_ensure_eq!(
        outer_fields.names(),
        &FieldNames::from_iter(expected_names.iter().cloned()),
        "TurboQuant storage outer fields must be {:?}, got {}",
        expected_names,
        outer_fields.names()
    );

    for (index, &block) in metadata.block_sizes.iter().enumerate() {
        let name = block_field_name(index);
        let Some(inner) = outer_fields.field(&name) else {
            vortex_bail!("TurboQuant storage missing inner field {name}");
        };
        validate_inner_block_dtype(metadata.element_ptype, block, &name, &inner)?;
    }

    Ok(())
}

fn validate_inner_block_dtype(
    element_ptype: PType,
    block: u32,
    name: &str,
    dtype: &DType,
) -> VortexResult<()> {
    let DType::Struct(fields, _) = dtype else {
        vortex_bail!("TurboQuant inner block {name} must be a Struct, got {dtype}");
    };
    let expected = FieldNames::from([NORMS_FIELD, CODES_FIELD]);
    vortex_ensure_eq!(
        fields.names(),
        &expected,
        "TurboQuant inner block {name} fields must be {expected}, got {}",
        fields.names()
    );

    let Some(norms_dtype) = fields.field(NORMS_FIELD) else {
        vortex_bail!("TurboQuant inner block {name} missing {NORMS_FIELD}");
    };
    let DType::Primitive(norms_ptype, _) = norms_dtype else {
        vortex_bail!(
            "TurboQuant inner block {name} {NORMS_FIELD} must be primitive, got {norms_dtype}"
        );
    };
    vortex_ensure_eq!(
        norms_ptype,
        element_ptype,
        "TurboQuant inner block {name} {NORMS_FIELD} ptype must be {element_ptype}, got \
         {norms_ptype}"
    );

    let Some(codes_dtype) = fields.field(CODES_FIELD) else {
        vortex_bail!("TurboQuant inner block {name} missing {CODES_FIELD}");
    };
    let DType::FixedSizeList(element_dtype, list_size, _) = codes_dtype else {
        vortex_bail!(
            "TurboQuant inner block {name} {CODES_FIELD} must be fixed-size-list, got \
             {codes_dtype}"
        );
    };
    vortex_ensure_eq!(
        list_size,
        block,
        "TurboQuant inner block {name} {CODES_FIELD} list size must be {block}, got {list_size}"
    );
    vortex_ensure_eq!(
        element_dtype.as_ref(),
        &DType::Primitive(PType::U8, Nullability::NonNullable),
        "TurboQuant inner block {name} {CODES_FIELD} elements must be non-nullable u8, got \
         {element_dtype}"
    );

    Ok(())
}

impl fmt::Display for TurboQuantMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "element_ptype: {}, dimensions: {}, bit_width: {}, seed: {}, num_rounds: {}, \
             block_sizes: [",
            self.element_ptype, self.dimensions, self.bit_width, self.seed, self.num_rounds,
        )?;
        for (index, block) in self.block_sizes.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{block}")?;
        }
        write!(f, "]")
    }
}
