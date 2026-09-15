// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for DBP values with unsigned lower parts.

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayParts;
use vortex_array::ArraySerialization;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use super::decimal_byte_parts_v2_id;
use crate::DecimalByteParts;
use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsArraySlotsExt;
use crate::DecimalBytePartsData;
use crate::decimal_byte_parts::MAX_LOWER_PARTS;

/// Metadata for decimal byte parts with lower parts.
#[derive(Clone, prost::Message)]
pub struct DecimalBytePartsV2Metadata {
    /// Ptype of the most significant part.
    #[prost(enumeration = "PType", tag = "1")]
    pub(super) msp_ptype: i32,
    /// Ptypes of the lower parts, ordered most significant first.
    #[prost(enumeration = "PType", repeated, tag = "2")]
    pub(super) lower_part_ptypes: Vec<i32>,
}

pub(super) fn serialize(
    array: ArrayView<'_, DecimalByteParts>,
) -> VortexResult<ArraySerialization> {
    let lower_parts = array.lower_parts();

    let metadata = DecimalBytePartsV2Metadata {
        msp_ptype: PType::try_from(array.msp().dtype())? as i32,
        lower_part_ptypes: lower_parts
            .iter()
            .map(|part| PType::try_from(part.dtype()).map(|ptype| ptype as i32))
            .collect::<VortexResult<_>>()?,
    }
    .encode_to_vec();

    let mut children = Vec::with_capacity(1 + lower_parts.len());
    children.push(array.msp().clone());
    children.extend(lower_parts.iter().cloned());

    Ok(ArraySerialization::new(
        decimal_byte_parts_v2_id(),
        metadata,
        vec![],
        children,
    ))
}

pub(super) fn deserialize(parts: ArrayDeserialization<'_>) -> VortexResult<DecimalBytePartsArray> {
    let metadata = DecimalBytePartsV2Metadata::decode(parts.metadata)?;
    vortex_ensure!(
        parts.dtype.as_decimal_opt().is_some(),
        "expected a decimal dtype"
    );

    let lower_part_count = metadata.lower_part_ptypes.len();
    vortex_ensure!(
        lower_part_count <= MAX_LOWER_PARTS,
        "v2 carries at most {MAX_LOWER_PARTS} lower parts, got {lower_part_count}"
    );
    vortex_ensure!(
        parts.children.len() == 1 + lower_part_count,
        "expected {} children, got {}",
        1 + lower_part_count,
        parts.children.len()
    );

    let msp_ptype = PType::try_from(metadata.msp_ptype)?;
    vortex_ensure!(
        msp_ptype.is_signed_int(),
        "MSP must have a signed integer dtype, got {msp_ptype}"
    );
    let msp_dtype = DType::Primitive(msp_ptype, parts.dtype.nullability());

    let mut slots = ArraySlots::with_capacity(parts.children.len());
    slots.push(Some(parts.children.get(0, &msp_dtype, parts.len)?));
    for (idx, raw_ptype) in metadata.lower_part_ptypes.into_iter().enumerate() {
        let ptype = PType::try_from(raw_ptype)
            .map_err(|_| vortex_err!("invalid PType {raw_ptype} for lower part {idx}"))?;
        vortex_ensure!(
            ptype.is_unsigned_int(),
            "lower part {idx} must have an unsigned integer dtype, got {ptype}"
        );
        slots.push(Some(parts.children.get(
            1 + idx,
            &DType::Primitive(ptype, Nullability::NonNullable),
            parts.len,
        )?));
    }
    Array::try_from_parts(
        ArrayParts::new(
            DecimalByteParts,
            parts.dtype.clone(),
            parts.len,
            DecimalBytePartsData,
        )
        .with_slots(slots),
    )
}
