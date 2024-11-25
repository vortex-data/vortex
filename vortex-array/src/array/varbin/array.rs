use vortex_error::VortexResult;

use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::validity::{LogicalValidity, ValidityVTable};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::ArrayLen;

impl ValidityVTable<VarBinArray> for VarBinEncoding {
    fn is_valid(&self, array: &VarBinArray, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &VarBinArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<VarBinArray> for VarBinEncoding {
    fn accept(&self, array: &VarBinArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("offsets", &array.offsets())?;
        visitor.visit_child("bytes", &array.bytes())?;
        visitor.visit_validity(&array.validity())
    }
}
