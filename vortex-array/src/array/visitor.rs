use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::validity::Validity;
use crate::ArrayData;

pub trait AcceptArrayVisitor {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()>;
}

// TODO(ngates): maybe we make this more like the inverse of TryFromParts?
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

    fn visit_buffer(&mut self, _buffer: &Buffer) -> VortexResult<()> {
        Ok(())
    }
}
