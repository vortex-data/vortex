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

/// Serialized metadata for `vortex.decimal_byte_parts.v2`.
///
/// The children are the MSP followed by one child per entry of `lower_part_ptypes`, in that
/// order. The MSP carries the array's validity, so there is no separate validity child, and the
/// position of every child follows from the metadata alone.
///
/// Readers require at least that many children and ignore any that follow. A later writer may
/// append a child under this ID only if decoding without it still yields the same values, such as
/// a statistic or an index. A child or metadata field that changes the decoded values needs a new
/// encoding ID.
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
        "expected a decimal dtype, got {}",
        parts.dtype
    );

    // Validate every ptype before reading any child: the in-memory `ArrayChildren` asserts that
    // each child matches the requested dtype and panics otherwise.
    let lower_part_count = metadata.lower_part_ptypes.len();
    vortex_ensure!(
        lower_part_count <= MAX_LOWER_PARTS,
        "v2 carries at most {MAX_LOWER_PARTS} lower parts, got {lower_part_count}"
    );
    let msp_ptype = PType::try_from(metadata.msp_ptype)
        .map_err(|_| vortex_err!("invalid PType {} for the MSP", metadata.msp_ptype))?;
    vortex_ensure!(
        msp_ptype.is_signed_int(),
        "MSP must have a signed integer dtype, got {msp_ptype}"
    );
    let lower_part_dtypes = metadata
        .lower_part_ptypes
        .iter()
        .enumerate()
        .map(|(idx, &raw_ptype)| -> VortexResult<DType> {
            let ptype = PType::try_from(raw_ptype)
                .map_err(|_| vortex_err!("invalid PType {raw_ptype} for lower part {idx}"))?;
            vortex_ensure!(
                ptype.is_unsigned_int(),
                "lower part {idx} must have an unsigned integer dtype, got {ptype}"
            );
            Ok(DType::Primitive(ptype, Nullability::NonNullable))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let required_children = 1 + lower_part_count;
    vortex_ensure!(
        parts.children.len() >= required_children,
        "v2 expects at least {required_children} children (the MSP plus one per lower part), \
         got {}",
        parts.children.len()
    );

    let msp_dtype = DType::Primitive(msp_ptype, parts.dtype.nullability());
    let mut slots = ArraySlots::with_capacity(required_children);
    slots.push(Some(parts.children.get(0, &msp_dtype, parts.len)?));
    for (idx, dtype) in lower_part_dtypes.iter().enumerate() {
        slots.push(Some(parts.children.get(1 + idx, dtype, parts.len)?));
    }
    Array::try_from_parts(ArrayParts::new(
        DecimalByteParts,
        parts.dtype.clone(),
        parts.len,
        DecimalBytePartsData,
        slots,
    ))
}
