// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ProstMetadata;
use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::SerdeVTable;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};

use super::BitPackedEncoding;
use crate::{BitPackedArray, BitPackedVTable};

#[derive(Clone, prost::Message)]
pub struct BitPackedMetadata {
    #[prost(uint32, tag = "1")]
    pub bit_width: u32,
    #[prost(uint32, tag = "2")]
    pub offset: u32, // must be <1024
    #[prost(message, optional, tag = "3")]
    pub(crate) patches: Option<PatchesMetadata>,
}

impl SerdeVTable<BitPackedVTable> for BitPackedVTable {
    type Metadata = ProstMetadata<BitPackedMetadata>;

    fn metadata(array: &BitPackedArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(BitPackedMetadata {
            bit_width: array.bit_width() as u32,
            offset: array.offset() as u32,
            patches: array
                .patches()
                .map(|p| p.to_metadata(array.len(), array.dtype()))
                .transpose()?,
        })))
    }

    /// Deserialize a BitPackedArray from its components.
    ///
    /// Note that the layout depends on whether patches and chunk_offsets are present:
    /// - No patches: `[validity?]`
    /// - With patches: `[patch_indices, patch_values, chunk_offsets?, validity?]`
    fn build(
        _encoding: &BitPackedEncoding,
        dtype: &DType,
        len: usize,
        metadata: &BitPackedMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<BitPackedArray> {
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
            Some(patches_meta) if patches_meta.chunk_offsets_dtype().is_some() => 3,
            Some(_) => 2,
        };

        let validity = load_validity(validity_idx)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(0, &p.indices_dtype(), p.len())?;
                let values = children.get(1, dtype, p.len())?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()
                    .map(|dtype| children.get(2, &dtype, p.chunk_offsets_len() as usize))
                    .transpose()?;

                Ok::<_, VortexError>(Patches::new(
                    len,
                    p.offset(),
                    indices,
                    values,
                    chunk_offsets,
                ))
            })
            .transpose()?;

        BitPackedArray::try_new(
            packed,
            PType::try_from(dtype)?,
            validity,
            patches,
            u8::try_from(metadata.bit_width).map_err(|_| {
                vortex_err!(
                    "BitPackedMetadata bit_width {} does not fit in u8",
                    metadata.bit_width
                )
            })?,
            len,
            u16::try_from(metadata.offset).map_err(|_| {
                vortex_err!(
                    "BitPackedMetadata offset {} does not fit in u16",
                    metadata.offset
                )
            })?,
        )
    }
}
