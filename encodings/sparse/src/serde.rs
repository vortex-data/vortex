use vortex_array::patches::PatchesMetadata;
use vortex_array::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayVisitorImpl, RkyvMetadata};
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;

use crate::SparseArray;

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct SparseMetadata {
    pub(crate) patches: PatchesMetadata,
}

impl ArrayVisitorImpl<RkyvMetadata<SparseMetadata>> for SparseArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        let fill_value_buffer = self
            .fill_value
            .value()
            .to_flexbytes::<ByteBufferMut>()
            .freeze();
        visitor.visit_buffer(&fill_value_buffer);
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_patches(self.patches())
    }

    fn _metadata(&self) -> RkyvMetadata<SparseMetadata> {
        RkyvMetadata(SparseMetadata {
            patches: self
                .patches()
                .to_metadata(self.len(), self.dtype())
                .vortex_expect("Failed to create patches metadata"),
        })
    }
}
