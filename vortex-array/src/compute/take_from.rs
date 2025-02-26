use vortex_error::{VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::{Array, ArrayRef};

pub trait TakeFromFn<A> {
    fn take_from(&self, indices: A, array: &dyn Array) -> VortexResult<Option<ArrayRef>>;
}

impl<E: Encoding> TakeFromFn<&dyn Array> for E
where
    E: for<'a> TakeFromFn<&'a E::Array>,
{
    fn take_from(&self, indices: &dyn Array, array: &dyn Array) -> VortexResult<Option<ArrayRef>> {
        let indices = indices
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");

        TakeFromFn::take_from(self, indices, array)
    }
}
