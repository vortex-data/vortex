// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, ValidityHelper, VisitorVTable};
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor, Canonical, ProstMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_err};

use super::{BitPackedEncoding, bit_width_histogram, find_best_bit_width};
use crate::{BitPackedArray, BitPackedVTable, bitpack_encode};

#[derive(Clone, prost::Message)]
pub struct BitPackedMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) bit_width: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32, // must be <1024
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

impl EncodeVTable<BitPackedVTable> for BitPackedVTable {
    fn encode(
        _encoding: &BitPackedEncoding,
        canonical: &Canonical,
        like: Option<&BitPackedArray>,
    ) -> VortexResult<Option<BitPackedArray>> {
        let parray = canonical.clone().into_primitive();

        let bit_width = like
            .map(|like_array| like_array.bit_width())
            // Only reuse the bitwidth if its smaller than the array's original bitwidth.
            .filter(|bw| (*bw as usize) < parray.ptype().bit_width());

        // In our current benchmark suite this seems to be the faster option,
        // but it has an unbounded worst-case where some array becomes all patches.
        let (bit_width, bit_width_histogram) = match bit_width {
            Some(bw) => (bw, None),
            None => {
                let histogram = bit_width_histogram(&parray)?;
                let bit_width = find_best_bit_width(parray.ptype(), &histogram)?;
                (bit_width, Some(histogram))
            }
        };

        if bit_width as usize == parray.ptype().bit_width()
            || parray.ptype().is_signed_int()
                && parray.statistics().compute_min::<i64>().unwrap_or_default() < 0
        {
            // Bit-packed compression not supported.
            return Ok(None);
        }

        Ok(Some(bitpack_encode(
            &parray,
            bit_width,
            bit_width_histogram.as_deref(),
        )?))
    }
}

impl VisitorVTable<BitPackedVTable> for BitPackedVTable {
    fn visit_buffers(array: &BitPackedArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.packed());
    }

    fn visit_children(array: &BitPackedArray, visitor: &mut dyn ArrayChildVisitor) {
        if let Some(patches) = array.patches() {
            visitor.visit_patches(patches);
        }
        visitor.visit_validity(array.validity(), array.len());
    }
}
