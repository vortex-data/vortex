use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::patches::Patches;
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
        let (array_ref, encoding) = array.downcast_array_ref::<E>()?;
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

    /// Utility for visiting Array patches.
    fn visit_patches(&mut self, patches: &Patches) -> VortexResult<()> {
        self.visit_child("patch_indices", patches.indices())?;
        self.visit_child("patch_values", patches.values())
    }

    fn visit_buffer(&mut self, _buffer: &ByteBuffer) -> VortexResult<()> {
        Ok(())
    }
}
