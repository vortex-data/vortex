use vortex_error::VortexResult;

use crate::array::varbin::VarBinArray;
use crate::array::VarBinEncoding;
use crate::validity::{ArrayValidity, LogicalValidity};
use crate::visitor::{ArrayVisitor, VisitorVTable};
use crate::ArrayLen;

impl ArrayValidity for VarBinArray {
    fn is_valid(&self, index: usize) -> bool {
        self.validity().is_valid(index)
    }

    fn logical_validity(&self) -> LogicalValidity {
        self.validity().to_logical(self.len())
    }
}

impl VisitorVTable<VarBinArray> for VarBinEncoding {
    fn accept(&self, array: &VarBinArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("offsets", &array.offsets())?;
        visitor.visit_child("bytes", &array.bytes())?;
        visitor.visit_validity(&array.validity())
    }
}
