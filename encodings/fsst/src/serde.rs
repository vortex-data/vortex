use serde::{Deserialize, Serialize};
use vortex_array::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayVisitorImpl, SerdeMetadata};
use vortex_dtype::PType;
use vortex_error::VortexExpect;

use crate::FSSTArray;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FSSTMetadata {
    pub(crate) uncompressed_lengths_ptype: PType,
}

impl ArrayVisitorImpl<SerdeMetadata<FSSTMetadata>> for FSSTArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&self.symbols().clone().into_byte_buffer());
        visitor.visit_buffer(&self.symbol_lengths().clone().into_byte_buffer());
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", self.codes());
        visitor.visit_child("uncompressed_lengths", self.uncompressed_lengths());
    }

    fn _metadata(&self) -> SerdeMetadata<FSSTMetadata> {
        SerdeMetadata(FSSTMetadata {
            uncompressed_lengths_ptype: PType::try_from(self.uncompressed_lengths().dtype())
                .vortex_expect("Must be a valid PType"),
        })
    }
}

#[cfg(test)]
mod test {
    use vortex_array::SerdeMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::serde::FSSTMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fsst_metadata() {
        check_metadata(
            "fsst.metadata",
            SerdeMetadata(FSSTMetadata {
                uncompressed_lengths_ptype: PType::U64,
            }),
        );
    }
}
