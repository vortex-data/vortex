use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::builtins::PrimitiveEncoding;
use crate::builtins::primitive::PrimitiveArray;
use crate::compute::SliceFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef};

impl SliceFn<&PrimitiveArray> for PrimitiveEncoding {
    fn slice(&self, array: &PrimitiveArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        match_each_native_ptype!(array.ptype(), |$T| {
            Ok(PrimitiveArray::new(
                array.buffer::<$T>().slice(start..stop),
                array.validity().slice(start, stop)?,
            )
            .into_array())
        })
    }
}
