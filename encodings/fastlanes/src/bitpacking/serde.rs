use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayExt, ArrayRef,
    ArrayVisitorImpl, Canonical, DeserializeMetadata, Encoding, EncodingId, RkyvMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};

use super::{BitPackedEncoding, find_best_bit_width};
use crate::{BitPackedArray, bitpack_encode};

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct BitPackedMetadata {
    pub(crate) bit_width: u8,
    pub(crate) offset: u16, // must be <1024
    pub(crate) patches: Option<PatchesMetadata>,
}

impl EncodingVTable for BitPackedEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("fastlanes.bitpacked")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = RkyvMetadata::<BitPackedMetadata>::deserialize(parts.metadata())?;

        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let packed = parts.buffer(0)?;

        let load_validity = |child_idx: usize| {
            if parts.nchildren() == child_idx {
                Ok(Validity::from(dtype.nullability()))
            } else if parts.nchildren() == child_idx + 1 {
                let validity = parts.child(child_idx).decode(ctx, Validity::DTYPE, len)?;
                Ok(Validity::Array(validity))
            } else {
                vortex_bail!(
                    "Expected {} or {} children, got {}",
                    child_idx,
                    child_idx + 1,
                    parts.nchildren()
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
                let indices = parts.child(0).decode(ctx, p.indices_dtype(), p.len())?;
                let values = parts.child(1).decode(ctx, dtype.clone(), p.len())?;
                Ok::<_, VortexError>(Patches::new(len, p.offset(), indices, values))
            })
            .transpose()?;

        Ok(unsafe {
            BitPackedArray::new_unchecked_with_offset(
                packed,
                PType::try_from(&dtype)?,
                validity,
                patches,
                metadata.bit_width,
                len,
                metadata.offset,
            )?
            .into_array()
        })
    }

    fn encode(
        &self,
        input: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let parray = input.clone().into_primitive()?;

        let bit_width = like
            .map(|like| {
                like.as_opt::<<Self as Encoding>::Array>().ok_or_else(|| {
                    vortex_err!(
                        "Expected {} encoded array but got {}",
                        self.id(),
                        like.encoding()
                    )
                })
            })
            .transpose()?
            .map(|like_array| like_array.bit_width())
            // Only reuse the bitwidth if its smaller than the array's original bitwidth.
            .filter(|bw| (*bw as usize) < parray.ptype().bit_width());

        // In our current benchmark suite this seems to be the faster option,
        // but it has an unbounded worst-case where some array becomes all patches.
        let bit_width = match bit_width {
            Some(bw) => bw,
            None => find_best_bit_width(&parray)?,
        };

        let array = if bit_width as usize == parray.ptype().bit_width()
            || parray.ptype().is_signed_int()
                && parray.statistics().compute_min::<i64>().unwrap_or_default() < 0
        {
            parray.into_array()
        } else {
            bitpack_encode(&parray, bit_width)?.into_array()
        };

        Ok(Some(array))
    }
}

impl ArrayVisitorImpl<RkyvMetadata<BitPackedMetadata>> for BitPackedArray {
    fn _visit_buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(self.packed());
    }

    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        if let Some(patches) = self.patches() {
            visitor.visit_patches(patches);
        }
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> RkyvMetadata<BitPackedMetadata> {
        RkyvMetadata(BitPackedMetadata {
            bit_width: self.bit_width(),
            offset: self.offset(),
            patches: self
                .patches()
                .map(|p| p.to_metadata(self.len(), self.dtype()))
                .transpose()
                .vortex_expect("Failed to create patches metadata"),
        })
    }
}
