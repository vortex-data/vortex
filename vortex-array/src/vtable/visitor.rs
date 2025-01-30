use vortex_error::{VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::visitor::ArrayVisitor;
use crate::Array;

pub trait VisitorVTable<Array> {
    fn accept(&self, array: &Array, visitor: &mut dyn ArrayVisitor) -> VortexResult<()>;
}

impl<E: Encoding> VisitorVTable<Array> for E
where
    E: VisitorVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn accept(&self, array: &Array, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        VisitorVTable::accept(encoding, array_ref, visitor)
    }
}
