use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::validity::Validity;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, ValidityHelper, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayExt, ArrayRef, Canonical,
    ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};

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

    fn metadata(array: &BitPackedArray) -> Option<Self::Metadata> {
        Some(ProstMetadata(BitPackedMetadata {
            bit_width: array.bit_width() as u32,
            offset: array.offset() as u32,
            patches: array
                .patches()
                .map(|p| p.to_metadata(array.len(), array.dtype()))
                .transpose()
                .vortex_expect("Failed to create patches metadata"),
        }))
    }

    fn decode(
        _encoding: &BitPackedEncoding,
        dtype: DType,
        len: usize,
        metadata: &BitPackedMetadata,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<BitPackedArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let packed = buffers[0].clone();

        let load_validity = |child_idx: usize| {
            if children.len() == child_idx {
                Ok(Validity::from(dtype.nullability()))
            } else if children.len() == child_idx + 1 {
                let validity = children[child_idx].decode(ctx, Validity::DTYPE, len)?;
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

        // Load validity from the zero'th or second child, depending on whether patches are present.
        let validity = if metadata.patches.is_some() {
            load_validity(2)?
        } else {
            load_validity(0)?
        };

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children[0].decode(ctx, p.indices_dtype(), p.len())?;
                let values = children[1].decode(ctx, dtype.clone(), p.len())?;
                Ok::<_, VortexError>(Patches::new(len, p.offset(), indices, values))
            })
            .transpose()?;

        unsafe {
            BitPackedArray::new_unchecked_with_offset(
                packed,
                PType::try_from(&dtype)?,
                validity,
                patches,
                u8::try_from(metadata.bit_width).vortex_expect("Bit width out of range"),
                len,
                u16::try_from(metadata.offset).vortex_expect("Offset out of range"),
            )
        }
    }
}

impl EncodeVTable<BitPackedVTable> for BitPackedVTable {
    fn encode(
        encoding: &BitPackedEncoding,
        canonical: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<BitPackedArray>> {
        let parray = canonical.clone().into_primitive()?;

        let bit_width = like
            .map(|like| {
                like.as_opt::<Self>().ok_or_else(|| {
                    vortex_err!(
                        "Expected {} encoded array but got {}",
                        encoding.id(),
                        like.encoding_id()
                    )
                })
            })
            .transpose()?
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

    fn with_children(
        array: &BitPackedArray,
        children: &[ArrayRef],
    ) -> VortexResult<BitPackedArray> {
        let patches = array.patches().map(|existing| {
            let indices = children[0].clone();
            let values = children[1].clone();
            Patches::new(existing.array_len(), existing.offset(), indices, values)
        });

        let validity = if array.validity().is_array() {
            Validity::Array(children[children.len() - 1].clone())
        } else {
            array.validity().clone()
        };

        unsafe {
            BitPackedArray::new_unchecked_with_offset(
                array.packed().clone(),
                array.ptype(),
                validity,
                patches,
                array.bit_width(),
                array.len(),
                array.offset(),
            )
        }
    }
}
