use vortex_error::{VortexError, VortexResult};

use crate::encoding::Encoding;
use crate::visitor::ArrayVisitor;
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
        let (array_ref, encoding) = array.try_downcast_ref::<E>()?;
        VisitorVTable::accept(encoding, array_ref, visitor)
    }
}
