use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::ConstantArray;
use crate::{Array, ArrayRef, Encoding};

pub trait OptimizeFn<A> {
    fn optimize(&self, array: A) -> VortexResult<ArrayRef>;
}

impl<E: Encoding> OptimizeFn<&dyn Array> for E
where
    E: for<'a> OptimizeFn<&'a E::Array>,
{
    fn optimize(&self, array: &dyn Array) -> VortexResult<ArrayRef> {
        let array_ref = array
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        OptimizeFn::optimize(self, array_ref)
    }
}

pub fn optimize(array: &dyn Array) -> VortexResult<ArrayRef> {
    if let Some(v) = array.as_constant() {
        return Ok(ConstantArray::new(v, array.len()).into_array());
    }

    if let Some(optimize_fn) = array.vtable().optimize_fn() {
        optimize_fn.optimize(array)
    } else {
        log::debug!("No optimize implementation found for {}", array.encoding());
        Ok(array.to_array())
    }
}
