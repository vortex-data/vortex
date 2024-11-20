use vortex_error::VortexResult;

use crate::array::varbin::VarBinArray;
use crate::array::visitor::{AcceptArrayVisitor, ArrayVisitor};
use crate::validity::{ArrayValidity, LogicalValidity};
use crate::ArrayLen;

impl ArrayValidity for VarBinArray {
    fn is_valid(&self, index: usize) -> bool {
        self.validity().is_valid(index)
    }

    fn logical_validity(&self) -> LogicalValidity {
        self.validity().to_logical(self.len())
    }
}

impl AcceptArrayVisitor for VarBinArray {
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("offsets", &self.offsets())?;
        visitor.visit_child("bytes", &self.bytes())?;
        visitor.visit_validity(&self.validity())
    }
}
