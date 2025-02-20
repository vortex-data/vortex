use vortex_error::VortexResult;

use crate::visitor::ArrayVisitor;

pub trait ArrayVisitorImpl {
    fn _accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()>;
}
