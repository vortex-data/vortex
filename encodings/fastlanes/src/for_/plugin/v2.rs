// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `fastlanes.for.v2`, which stores one reference per chunk as a child.

use prost::Message as _;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::for_v2_id;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;
use crate::for_::array::num_chunks;

/// Metadata for `fastlanes.for.v2`. The references are the second child.
#[derive(Clone, prost::Message)]
pub(super) struct FoRV2Metadata {
    /// The position of the first element within the first chunk.
    #[prost(uint32, tag = "1")]
    pub(super) offset: u32,
    /// Byte width of the `encoded` child when narrowed to a smaller unsigned integer
    /// (`1` = `u8`, `2` = `u16`, `4` = `u32`). Zero means the `encoded` child matches the array's
    /// logical primitive type.
    #[prost(uint32, tag = "2")]
    pub(super) encoded_bytes: u32,
}

pub(super) fn serialize(array: ArrayView<'_, FoR>, references: &ArrayRef) -> ArraySerialization {
    let encoded_ptype = array.encoded().dtype().as_ptype();
    let encoded_bytes = if encoded_ptype == array.ptype() {
        0
    } else {
        u32::try_from(encoded_ptype.byte_width()).vortex_expect("byte width fits in u32")
    };
    let metadata = FoRV2Metadata {
        offset: u32::from(array.offset()),
        encoded_bytes,
    };
    ArraySerialization::new(
        for_v2_id(),
        metadata.encode_to_vec(),
        vec![],
        vec![array.encoded().clone(), references.clone()],
    )
}

pub(super) fn deserialize(parts: ArrayDeserialization<'_>) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        parts.buffers.is_empty(),
        "FoRArray expects 0 buffers, got {}",
        parts.buffers.len()
    );
    vortex_ensure_eq!(
        parts.children.len(),
        2,
        "Expected 2 children for {}",
        for_v2_id()
    );
    let metadata = FoRV2Metadata::decode(parts.metadata)?;
    vortex_ensure!(
        usize::try_from(metadata.offset).is_ok_and(|offset| offset < FL_CHUNK_SIZE),
        "FoR offset must be less than {FL_CHUNK_SIZE}, got {}",
        metadata.offset
    );
    vortex_ensure!(
        parts.dtype.is_int(),
        "FoR requires an integer dtype, got {}",
        parts.dtype
    );
    let offset = u16::try_from(metadata.offset)?;
    let logical_bytes = parts.dtype.as_ptype().byte_width();
    let encoded_dtype = match metadata.encoded_bytes {
        0 => parts.dtype.clone(),
        1 if 1 < logical_bytes => DType::Primitive(PType::U8, parts.dtype.nullability()),
        2 if 2 < logical_bytes => DType::Primitive(PType::U16, parts.dtype.nullability()),
        4 if 4 < logical_bytes => DType::Primitive(PType::U32, parts.dtype.nullability()),
        other => vortex_bail!(
            "Invalid FoR v2 encoded_bytes {other} for dtype {}",
            parts.dtype
        ),
    };
    let encoded = parts.children.get(0, &encoded_dtype, parts.len)?;
    let references = parts.children.get(
        1,
        &parts.dtype.as_nonnullable(),
        num_chunks(offset, parts.len),
    )?;
    Ok(FoR::try_new_chunked(encoded, references, offset)?.into_array())
}
