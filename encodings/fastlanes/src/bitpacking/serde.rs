use vortex_array::patches::{Patches, PatchesMetadata};
use vortex_array::serde::ArrayParts;
use vortex_array::validity::Validity;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl,
    DeserializeMetadata, RkyvMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail};

use crate::{BitPackedArray, BitPackedEncoding};

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct BitPackedMetadata {
    bit_width: u8,
    offset: u16, // must be <1024
    patches: Option<PatchesMetadata>,
}

impl ArrayVisitorImpl<RkyvMetadata<BitPackedMetadata>> for BitPackedArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(self.packed());
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
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

impl SerdeVTable<&BitPackedArray> for BitPackedEncoding {
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
}
