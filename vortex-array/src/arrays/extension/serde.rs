use vortex_error::VortexResult;

use crate::arrays::{ExtensionArray, ExtensionEncoding};
use crate::vtable::SerdeVTable;
use crate::{ArrayChildVisitor, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl for ExtensionArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("storage", self.storage())
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl SerdeVTable<&ExtensionArray> for ExtensionEncoding {}
