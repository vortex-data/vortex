// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `fastlanes.bitpacked.v2`, which stores the byte boundaries of the packed blocks as a
//! child, so each block has its own bit width.

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::patches::PatchesMetadata;
use vortex_array::validity::Validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use super::bitpacked_v2_id;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::BitPackedSlots;

/// Metadata for `fastlanes.bitpacked.v2`.
///
/// The children are the patch indices, values and chunk offsets when there are patches, then the
/// validity when there is a validity child, then the block offsets.
#[derive(Clone, prost::Message)]
pub(super) struct BitPackedV2Metadata {
    /// The position of the first element within the first block.
    #[prost(uint32, tag = "1")]
    pub(super) offset: u32,
    /// The unsigned integer type of the block offsets child.
    #[prost(enumeration = "PType", tag = "2")]
    pub(super) block_offsets_ptype: i32,
    #[prost(message, optional, tag = "3")]
    pub(super) patches: Option<PatchesMetadata>,
}

pub(super) fn serialize(
    array: &ArrayRef,
    view: ArrayView<'_, BitPacked>,
    block_offsets: &ArrayRef,
) -> VortexResult<ArraySerialization> {
    let metadata = BitPackedV2Metadata {
        offset: u32::from(view.offset()),
        block_offsets_ptype: PType::try_from(block_offsets.dtype())? as i32,
        patches: view
            .patches()
            .map(|p| p.to_metadata(view.len(), view.dtype()))
            .transpose()?,
    };
    Ok(ArraySerialization::from_array(
        bitpacked_v2_id(),
        array,
        metadata.encode_to_vec(),
    ))
}

pub(super) fn deserialize(parts: ArrayDeserialization<'_>) -> VortexResult<ArrayRef> {
    let ArrayDeserialization {
        dtype,
        len,
        metadata,
        buffers,
        children,
        ..
    } = parts;

    let metadata = BitPackedV2Metadata::decode(metadata)?;
    vortex_ensure!(
        buffers.len() == 1,
        "Expected 1 buffer, got {}",
        buffers.len()
    );
    vortex_ensure!(
        usize::try_from(metadata.offset).is_ok_and(|offset| offset < FL_CHUNK_SIZE),
        "BitPacked offset must be less than {FL_CHUNK_SIZE}, got {}",
        metadata.offset
    );
    let offset = u16::try_from(metadata.offset)?;
    let block_offsets_ptype = PType::try_from(metadata.block_offsets_ptype)?;

    let num_patch_children = match &metadata.patches {
        None => 0,
        Some(patches_meta) if patches_meta.chunk_offsets_dtype()?.is_some() => 3,
        Some(_) => 2,
    };
    let validity = if children.len() == num_patch_children + 1 {
        Validity::from(dtype.nullability())
    } else if children.len() == num_patch_children + 2 {
        Validity::Array(children.get(num_patch_children, &Validity::DTYPE, len)?)
    } else {
        vortex_bail!(
            "Expected {} or {} children, got {}",
            num_patch_children + 1,
            num_patch_children + 2,
            children.len()
        );
    };

    let num_blocks = (usize::from(offset) + len).div_ceil(FL_CHUNK_SIZE);
    let block_offsets = children.get(
        children.len() - 1,
        &DType::Primitive(block_offsets_ptype, Nullability::NonNullable),
        num_blocks + 1,
    )?;

    let patches = metadata
        .patches
        .map(|p| {
            let indices = children.get(0, &p.indices_dtype()?, p.len()?)?;
            let values = children.get(1, dtype, p.len()?)?;
            let chunk_offsets = p
                .chunk_offsets_dtype()?
                .map(|dtype| children.get(2, &dtype, p.chunk_offsets_len() as usize))
                .transpose()?;

            Patches::new(len, p.offset()?, indices, values, chunk_offsets)
        })
        .transpose()?;

    let slots = {
        let mut s = ArraySlots::with_capacity(BitPackedSlots::COUNT);
        PatchesData::push_slots(&mut s, patches.as_ref());
        s.push(validity_to_child(&validity, len));
        s.push(Some(block_offsets));
        s
    };
    let data = BitPackedData::try_new_blocked(buffers[0].clone(), patches, offset)?;
    Ok(Array::<BitPacked>::try_from_parts(ArrayParts::new(
        BitPacked,
        dtype.clone(),
        len,
        data,
        slots,
    ))?
    .into_array())
}
