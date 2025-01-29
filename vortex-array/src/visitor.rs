use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::patches::Patches;
use crate::validity::Validity;
use crate::ArrayData;

pub trait ArrayVisitor {
    /// Visit a child of this array.
    fn visit_child(&mut self, _name: &str, _array: &ArrayData) -> VortexResult<()> {
        Ok(())
    }

    /// Utility for visiting Array validity.
    fn visit_validity(&mut self, validity: &Validity) -> VortexResult<()> {
        if let Some(v) = validity.as_array() {
            self.visit_child("validity", v)
        } else {
            Ok(())
        }
    }

    /// Utility for visiting Array patches.
    fn visit_patches(&mut self, patches: &Patches) -> VortexResult<()> {
        self.visit_child("patch_indices", patches.indices())?;
        self.visit_child("patch_values", patches.values())
    }

    fn visit_buffer(&mut self, _buffer: &ByteBuffer) -> VortexResult<()> {
        Ok(())
    }
}
