use serde::{Deserialize, Serialize};
use vortex_array::patches::PatchesMetadata;
use vortex_array::{Array, ArrayChildVisitor, ArrayVisitorImpl, SerdeMetadata};
use vortex_error::VortexExpect;

use crate::{ALPArray, Exponents};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALPMetadata {
    pub(crate) exponents: Exponents,
    pub(crate) patches: Option<PatchesMetadata>,
}

impl ArrayVisitorImpl<SerdeMetadata<ALPMetadata>> for ALPArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", self.encoded());
        if let Some(patches) = self.patches() {
            visitor.visit_patches(patches);
        }
    }

    fn _metadata(&self) -> SerdeMetadata<ALPMetadata> {
        SerdeMetadata(ALPMetadata {
            exponents: self.exponents(),
            patches: self
                .patches()
                .map(|p| p.to_metadata(self.len(), self.dtype()))
                .transpose()
                .vortex_expect("Failed to create patches metadata"),
        })
    }
}
