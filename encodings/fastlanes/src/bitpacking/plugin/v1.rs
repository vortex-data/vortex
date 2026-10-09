// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `fastlanes.bitpacked`, which stores a single bit width in its metadata.

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::patches::PatchesMetadata;
use vortex_array::validity::Validity;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::bitpacked_v1_id;
use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitPackedData;
use crate::bitpacking::array::BitPackedSlots;

#[derive(Clone, prost::Message)]
pub struct BitPackedMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) bit_width: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32, // must be <1024
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

pub(super) fn serialize(
    array: &ArrayRef,
    view: ArrayView<'_, BitPacked>,
    bit_width: u8,
) -> VortexResult<ArraySerialization> {
    let metadata = BitPackedMetadata {
        bit_width: u32::from(bit_width),
        offset: view.offset() as u32,
        patches: view
            .patches()
            .map(|p| p.to_metadata(view.len(), view.dtype()))
            .transpose()?,
    }
    .encode_to_vec();
    Ok(ArraySerialization::from_array(
        bitpacked_v1_id(),
        array,
        metadata,
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

    let metadata = BitPackedMetadata::decode(metadata)?;
    if buffers.len() != 1 {
        vortex_bail!("Expected 1 buffer, got {}", buffers.len());
    }
    let packed = buffers[0].clone();

    let load_validity = |child_idx: usize| {
        if children.len() == child_idx {
            Ok(Validity::from(dtype.nullability()))
        } else if children.len() == child_idx + 1 {
            let validity = children.get(child_idx, &Validity::DTYPE, len)?;
            Ok(Validity::Array(validity))
        } else {
            vortex_bail!(
                "Expected {} or {} children, got {}",
                child_idx,
                child_idx + 1,
                children.len()
            );
        }
    };

    let validity_idx = match &metadata.patches {
        None => 0,
        Some(patches_meta) if patches_meta.chunk_offsets_dtype()?.is_some() => 3,
        Some(_) => 2,
    };

    let validity = load_validity(validity_idx)?;

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
        s.push(None);
        s
    };
    let data = BitPackedData::try_new(
        packed,
        patches,
        u8::try_from(metadata.bit_width).map_err(|_| {
            vortex_err!(
                "BitPackedMetadata bit_width {} does not fit in u8",
                metadata.bit_width
            )
        })?,
        u16::try_from(metadata.offset).map_err(|_| {
            vortex_err!(
                "BitPackedMetadata offset {} does not fit in u16",
                metadata.offset
            )
        })?,
    )?;
    Ok(Array::<BitPacked>::try_from_parts(ArrayParts::new(
        BitPacked,
        dtype.clone(),
        len,
        data,
        slots,
    ))?
    .into_array())
}
