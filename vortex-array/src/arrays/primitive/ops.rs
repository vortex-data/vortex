use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<PrimitiveVTable> for PrimitiveVTable {
    fn slice(array: &PrimitiveArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        match_each_native_ptype!(array.ptype(), |T| {
            Ok(PrimitiveArray::new(
                array.buffer::<T>().slice(start..stop),
                array.validity().slice(start, stop)?,
            )
            .into_array())
        })
    }

    fn scalar_at(array: &PrimitiveArray, index: usize) -> VortexResult<Scalar> {
        Ok(match_each_native_ptype!(array.ptype(), |T| {
            Scalar::primitive(array.as_slice::<T>()[index], array.dtype().nullability())
        }))
    }
}
