use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::SliceFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, IntoArray};

impl SliceFn<PrimitiveArray> for PrimitiveEncoding {
    fn slice(&self, array: &PrimitiveArray, start: usize, stop: usize) -> VortexResult<Array> {
        match_each_native_ptype!(array.ptype(), |$T| {
            Ok(PrimitiveArray::new(
                array.buffer::<$T>().slice(start..stop),
                array.validity().slice(start, stop)?,
            )
            .into_array())
        })
    }
}
