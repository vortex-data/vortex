use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::array::Array;
use crate::arrays::PrimitiveArray;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayOperationsImpl, ArrayRef};

impl ArrayOperationsImpl for PrimitiveArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        match_each_native_ptype!(self.ptype(), |$T| {
            Ok(PrimitiveArray::new(
                self.buffer::<$T>().slice(start..stop),
                self.validity().slice(start, stop)?,
            )
            .into_array())
        })
    }
}
