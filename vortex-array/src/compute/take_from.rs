use vortex_error::{VortexExpect, VortexResult, vortex_bail};

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

pub fn take_from(indices: &dyn Array, array: &dyn Array) -> VortexResult<Option<ArrayRef>> {
    let taken = indices
        .vtable()
        .take_from_fn()
        .map(|f| f.take_from(indices, array))
        .unwrap_or_else(|| vortex_bail!(NotImplemented: "take_from", array.encoding()))?;

    Ok(taken)
}
