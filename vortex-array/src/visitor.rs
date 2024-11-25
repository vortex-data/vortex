use vortex_buffer::Buffer;
use vortex_error::{vortex_err, VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::validity::Validity;
use crate::ArrayData;

pub trait VisitorVTable<Array> {
    fn accept(&self, array: &Array, visitor: &mut dyn ArrayVisitor) -> VortexResult<()>;
}

impl<E: Encoding> VisitorVTable<ArrayData> for E
where
    E: VisitorVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn accept(&self, array: &ArrayData, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        VisitorVTable::accept(encoding, array_ref, visitor)
    }
}

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
