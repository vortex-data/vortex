use vortex_error::VortexExpect;

use crate::arrays::StructArray;
use crate::variants::StructArrayTrait;
use crate::{Array, ArrayChildVisitor, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl for StructArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len());
        for (idx, name) in self.names().iter().enumerate() {
            let child = self
                .maybe_null_field_by_idx(idx)
                .vortex_expect("no out of bounds");
            visitor.visit_child(name.as_ref(), &child);
        }
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
