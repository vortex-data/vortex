use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::varbin::VarBinArray;
use crate::arrays::VarBinEncoding;
use crate::visitor::ArrayVisitor;
use crate::vtable::{ValidityVTable, VisitorVTable};

impl ValidityVTable<VarBinArray> for VarBinEncoding {
    fn is_valid(&self, array: &VarBinArray, index: usize) -> VortexResult<bool> {
        array.validity().is_valid(index)
    }

    fn all_valid(&self, array: &VarBinArray) -> VortexResult<bool> {
        array.validity().all_valid()
    }

    fn all_invalid(&self, array: &VarBinArray) -> VortexResult<bool> {
        array.validity().all_invalid()
    }

    fn validity_mask(&self, array: &VarBinArray) -> VortexResult<Mask> {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<VarBinArray> for VarBinEncoding {
    fn accept(&self, array: &VarBinArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("offsets", &array.offsets())?;
        visitor.visit_buffer(&array.bytes())?;
        visitor.visit_validity(&array.validity())
    }
}
