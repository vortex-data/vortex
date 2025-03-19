use vortex_array::patches::PatchesMetadata;
use vortex_array::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayVisitorImpl, RkyvMetadata};
use vortex_error::VortexExpect;

use crate::BitPackedArray;

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct BitPackedMetadata {
    pub(crate) bit_width: u8,
    pub(crate) offset: u16, // must be <1024
    pub(crate) patches: Option<PatchesMetadata>,
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
