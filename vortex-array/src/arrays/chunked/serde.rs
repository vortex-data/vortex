use crate::arrays::{ChunkedArray, PrimitiveArray};
use crate::validity::Validity;
use crate::{ArrayChildVisitor, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl for ChunkedArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        let chunk_offsets = PrimitiveArray::new(self.chunk_offsets.clone(), Validity::NonNullable);
        visitor.visit_child("chunk_offsets", &chunk_offsets);

        for (idx, chunk) in self.chunks().iter().enumerate() {
            visitor.visit_child(format!("chunks[{}]", idx).as_str(), chunk);
        }
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
