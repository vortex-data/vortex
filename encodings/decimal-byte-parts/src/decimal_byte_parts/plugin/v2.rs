// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for DBP values with unsigned lower parts.

use num_traits::AsPrimitive;
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

/// Metadata for decimal byte parts with per-child storage types.
#[derive(Clone, prost::Message)]
pub struct DecimalBytePartsV2Metadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(super) zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    pub(super) lower_part_count: u32,
    /// Unsigned storage types of the lower parts, most significant first.
    #[prost(enumeration = "PType", repeated, tag = "3")]
    pub(super) lower_part_ptypes: Vec<i32>,
}

pub(super) fn serialize(
    array: ArrayView<'_, DecimalByteParts>,
) -> VortexResult<ArraySerialization> {
    let lower_parts = array.lower_parts();
    vortex_ensure!(!lower_parts.is_empty(), "v2 requires lower parts");
    let metadata = DecimalBytePartsV2Metadata {
        zeroth_child_ptype: PType::try_from(array.msp().dtype())? as i32,
        lower_part_count: u32::try_from(lower_parts.len())
            .map_err(|_| vortex_err!("lower part count exceeds u32"))?,
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
    vortex_ensure!(
        parts.serialized_id == decimal_byte_parts_v2_id(),
        "expected the v2 format"
    );
    let metadata = DecimalBytePartsV2Metadata::decode(parts.metadata)?;
    vortex_ensure!(
        parts.dtype.as_decimal_opt().is_some(),
        "expected a decimal dtype"
    );

    let n_lower_parts: usize = metadata.lower_part_count.as_();

    vortex_ensure!(
        n_lower_parts <= MAX_LOWER_PARTS,
        "expected at most {MAX_LOWER_PARTS} lower parts"
    );

    let n_lower_part_ptypes = metadata.lower_part_ptypes.len();
    vortex_ensure!(
        n_lower_part_ptypes == n_lower_parts,
        "got {n_lower_part_ptypes} lower part ptypes but {n_lower_parts} lower parts"
    );

    let n_children = parts.children.len();
    let n_children_expected = 1 + n_lower_parts;
    vortex_ensure!(
        n_children == n_children_expected,
        "expected {n_children_expected} children, got {n_children}"
    );

    let msp_ptype = PType::try_from(metadata.zeroth_child_ptype)?;
    vortex_ensure!(
        msp_ptype.is_signed_int(),
        "MSP must have a signed integer ptype"
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
