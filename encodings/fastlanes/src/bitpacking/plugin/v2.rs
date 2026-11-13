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
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::patches::PATCH_CHUNK_SIZE;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::patches::PatchesMetadata;
use vortex_array::validity::Validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::bitpacked_v2_id;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedArraySlotsExt;
use crate::BitPackedData;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::array::BitPackedSlots;
use crate::bitpacking::array::validate_host_block_offsets;

/// Serialized metadata for `fastlanes.bitpacked.v2`.
///
/// The children are the block offsets, then the patch indices and values when `patches` is set,
/// then the patch chunk offsets when `patches` records their type, then the validity child when
/// `has_validity` is set. A nullable array without a validity child is all valid.
///
/// Readers require at least that many children and ignore any that follow. A later writer may
/// append a child under this ID only if decoding without it still yields the same values, such as
/// a statistic or an index. A child or metadata field that changes the decoded values needs a new
/// encoding ID.
#[derive(Clone, prost::Message)]
pub(super) struct BitPackedV2Metadata {
    /// The position of the first element within the first block, less than 1024.
    #[prost(uint32, tag = "1")]
    pub(super) offset: u32,
    /// The unsigned integer type of the block offsets child.
    #[prost(enumeration = "PType", tag = "2")]
    pub(super) block_offsets_ptype: i32,
    /// The patches, whose children follow the block offsets.
    #[prost(message, optional, tag = "3")]
    pub(super) patches: Option<PatchesMetadata>,
    /// Whether the children include a validity child after the block offsets and patches.
    #[prost(bool, tag = "4")]
    pub(super) has_validity: bool,
}

pub(super) fn serialize(
    array: &ArrayRef,
    view: ArrayView<'_, BitPacked>,
    block_offsets: &ArrayRef,
) -> VortexResult<ArraySerialization> {
    let patches = view.patches();
    let validity_child = view.validity_child();
    let metadata = BitPackedV2Metadata {
        offset: u32::from(view.offset()),
        block_offsets_ptype: PType::try_from(block_offsets.dtype())? as i32,
        patches: patches
            .as_ref()
            .map(|p| p.to_metadata(view.len(), view.dtype()))
            .transpose()?,
        has_validity: validity_child.is_some(),
    };

    // The wire order differs from the slot order, which puts the block offsets last.
    let mut children = Vec::with_capacity(BitPackedSlots::COUNT);
    children.push(block_offsets.clone());
    if let Some(patches) = &patches {
        children.push(patches.indices().clone());
        children.push(patches.values().clone());
        children.extend(patches.chunk_offsets().clone());
    }
    children.extend(validity_child.cloned());

    Ok(ArraySerialization::new(
        bitpacked_v2_id(),
        metadata.encode_to_vec(),
        array.buffers(),
        children,
    ))
}

pub(super) fn deserialize(
    parts: ArrayDeserialization<'_>,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
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
    let packed = &buffers[0];
    vortex_ensure!(
        dtype.is_int(),
        "BitPacked dtype must be an integer, got {dtype}"
    );
    let ptype = PType::try_from(dtype)?;
    vortex_ensure!(
        !metadata.has_validity || dtype.is_nullable(),
        "BitPacked validity child requires a nullable dtype, got {dtype}"
    );
    vortex_ensure!(
        usize::try_from(metadata.offset).is_ok_and(|offset| offset < FL_CHUNK_SIZE),
        "BitPacked offset must be less than {FL_CHUNK_SIZE}, got {}",
        metadata.offset
    );
    let offset = u16::try_from(metadata.offset)?;

    let block_offsets_ptype = PType::try_from(metadata.block_offsets_ptype)?;
    vortex_ensure!(
        block_offsets_ptype.is_unsigned_int(),
        "BitPacked block offsets must be unsigned integers, got {block_offsets_ptype}"
    );
    let num_blocks = usize::from(offset)
        .checked_add(len)
        .map(|end| end.div_ceil(FL_CHUNK_SIZE))
        .ok_or_else(|| vortex_err!("BitPacked offset + len overflowed usize"))?;
    // Each block holds at most `ptype` bits per value, 128 bytes per bit.
    let max_packed_len = num_blocks
        .checked_mul(128 * ptype.bit_width())
        .ok_or_else(|| vortex_err!("BitPacked packed length overflowed usize"))?;
    vortex_ensure!(
        packed.len().is_multiple_of(128) && packed.len() <= max_packed_len,
        "BitPacked packed buffer of {} bytes cannot hold {num_blocks} blocks of {ptype}",
        packed.len()
    );

    let patch_children = match &metadata.patches {
        None => 0,
        Some(p) => {
            let patches_len = p.len()?;
            vortex_ensure!(
                patches_len > 0 && patches_len <= len,
                "BitPacked patch count must be in 1..={len}, got {patches_len}"
            );
            p.indices_dtype()?;
            match p.chunk_offsets_dtype()? {
                None => 2,
                Some(chunk_offsets_dtype) => {
                    vortex_ensure!(
                        chunk_offsets_dtype.is_unsigned_int(),
                        "BitPacked patch chunk offsets must be unsigned integers, got \
                         {chunk_offsets_dtype}"
                    );
                    let chunk_offsets_len = usize::try_from(p.chunk_offsets_len())?;
                    let max_chunks = len.div_ceil(PATCH_CHUNK_SIZE) + 1;
                    vortex_ensure!(
                        chunk_offsets_len > 0 && chunk_offsets_len <= max_chunks,
                        "BitPacked patch chunk offsets length must be in 1..={max_chunks}, got \
                         {chunk_offsets_len}"
                    );
                    3
                }
            }
        }
    };
    let validity_idx = 1 + patch_children;
    let required_children = validity_idx + usize::from(metadata.has_validity);
    vortex_ensure!(
        children.len() >= required_children,
        "BitPacked expects at least {required_children} children, got {}",
        children.len()
    );

    let block_offsets = children.get(
        0,
        &DType::Primitive(block_offsets_ptype, Nullability::NonNullable),
        num_blocks + 1,
    )?;
    // Debug builds would otherwise panic on these boundaries when validating the array.
    validate_host_block_offsets(&block_offsets, ptype, packed.len())?;

    let validity = if metadata.has_validity {
        Validity::Array(children.get(validity_idx, &Validity::DTYPE, len)?)
    } else {
        Validity::from(dtype.nullability())
    };

    let patches = metadata
        .patches
        .map(|p| {
            let indices = children.get(1, &p.indices_dtype()?, p.len()?)?;
            let values = children.get(2, dtype, p.len()?)?;
            let chunk_offsets = p
                .chunk_offsets_dtype()?
                .map(|dtype| children.get(3, &dtype, usize::try_from(p.chunk_offsets_len())?))
                .transpose()?;

            Patches::from_metadata(
                len,
                &p,
                indices,
                values,
                chunk_offsets,
                &mut session.create_execution_ctx(),
            )
        })
        .transpose()?;

    let slots = {
        let mut s = ArraySlots::with_capacity(BitPackedSlots::COUNT);
        PatchesData::push_slots(&mut s, patches.as_ref());
        s.push(validity_to_child(&validity, len));
        s.push(Some(block_offsets));
        s
    };
    let data = BitPackedData::try_new_blocked(packed.clone(), patches, offset)?;
    Ok(Array::<BitPacked>::try_from_parts(ArrayParts::new(
        BitPacked,
        dtype.clone(),
        len,
        data,
        slots,
    ))?
    .into_array())
}
