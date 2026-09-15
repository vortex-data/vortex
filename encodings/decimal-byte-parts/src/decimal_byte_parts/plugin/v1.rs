// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for the frozen single-child DBP wire format.

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayParts;
use vortex_array::ArraySerialization;
use vortex_array::ArrayView;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::decimal_byte_parts_v1_id;
use crate::DecimalByteParts;
use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsArraySlotsExt;
use crate::DecimalBytePartsData;

#[derive(Clone, prost::Message)]
struct DecimalBytePartsMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    lower_part_count: u32,
}

pub(super) fn serialize(
    array: ArrayView<'_, DecimalByteParts>,
) -> VortexResult<ArraySerialization> {
    vortex_ensure!(
        array.lower_parts().is_empty(),
        "v1 must not carry lower parts"
    );
    let msp = array.msp();
    let metadata = DecimalBytePartsMetadata {
        zeroth_child_ptype: PType::try_from(msp.dtype())? as i32,
        lower_part_count: 0,
    }
    .encode_to_vec();
    Ok(ArraySerialization::new(
        decimal_byte_parts_v1_id(),
        metadata,
        vec![],
        vec![msp.clone()],
    ))
}

pub(super) fn deserialize(parts: ArrayDeserialization<'_>) -> VortexResult<DecimalBytePartsArray> {
    vortex_ensure!(
        parts.serialized_id == decimal_byte_parts_v1_id(),
        "expected the v1 format"
    );
    let metadata = DecimalBytePartsMetadata::decode(parts.metadata)?;
    vortex_ensure!(
        parts.dtype.as_decimal_opt().is_some(),
        "expected a decimal dtype"
    );
    vortex_ensure!(
        metadata.lower_part_count == 0,
        "v1 must not carry lower parts"
    );
    vortex_ensure!(parts.children.len() == 1, "v1 must carry exactly one child");
    let ptype = PType::try_from(metadata.zeroth_child_ptype)?;
    vortex_ensure!(
        ptype.is_signed_int(),
        "MSP must have a signed integer dtype"
    );
    let encoded_dtype = DType::Primitive(ptype, parts.dtype.nullability());
    let msp = parts.children.get(0, &encoded_dtype, parts.len)?;
    Array::try_from_parts(
        ArrayParts::new(
            DecimalByteParts,
            parts.dtype.clone(),
            parts.len,
            DecimalBytePartsData,
        )
        .with_slots(smallvec![Some(msp)]),
    )
}
