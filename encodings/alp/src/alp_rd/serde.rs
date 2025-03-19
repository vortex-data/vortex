use serde::{Deserialize, Serialize};
use vortex_array::patches::PatchesMetadata;
use vortex_array::{Array, ArrayChildVisitor, ArrayVisitorImpl, SerdeMetadata};
use vortex_dtype::PType;
use vortex_error::VortexExpect;

use crate::ALPRDArray;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALPRDMetadata {
    pub(crate) right_bit_width: u8,
    pub(crate) dict_len: u8,
    pub(crate) dict: [u16; 8],
    pub(crate) left_parts_ptype: PType,
    pub(crate) patches: Option<PatchesMetadata>,
}

impl ArrayVisitorImpl<SerdeMetadata<ALPRDMetadata>> for ALPRDArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("left_parts", self.left_parts());
        visitor.visit_child("right_parts", self.right_parts());
        if let Some(patches) = self.left_parts_patches() {
            visitor.visit_patches(patches);
        }
    }

    fn _metadata(&self) -> SerdeMetadata<ALPRDMetadata> {
        let mut dict = [0u16; 8];
        dict[0..self.left_parts_dictionary().len()].copy_from_slice(self.left_parts_dictionary());

        SerdeMetadata(ALPRDMetadata {
            right_bit_width: self.right_bit_width(),
            dict_len: self.left_parts_dictionary().len() as u8,
            dict,
            left_parts_ptype: PType::try_from(self.left_parts().dtype())
                .vortex_expect("Must be a valid PType"),
            patches: self
                .left_parts_patches()
                .map(|p| p.to_metadata(self.len(), self.left_parts().dtype()))
                .transpose()
                .vortex_expect("Failed to create patches metadata"),
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_array::SerdeMetadata;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::alp_rd::serde::ALPRDMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alprd_metadata() {
        check_metadata(
            "alprd.metadata",
            SerdeMetadata(ALPRDMetadata {
                right_bit_width: u8::MAX,
                patches: Some(PatchesMetadata::new(usize::MAX, usize::MAX, PType::U64)),
                dict: [0u16; 8],
                left_parts_ptype: PType::U64,
                dict_len: 8,
            }),
        );
    }
}
