use vortex_array::arrays::{ConstantArray, ConstantVTable};
use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayExt, ArrayRef, IntoArray};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{DictArray, DictVTable};

impl OperationsVTable<DictVTable> for DictVTable {
    fn slice(array: &DictArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let sliced_code = array.codes().slice(start, stop)?;
        if sliced_code.is::<ConstantVTable>() {
            let code = usize::try_from(&sliced_code.scalar_at(0)?)?;
            return Ok(
                ConstantArray::new(array.values().scalar_at(code)?, sliced_code.len()).to_array(),
            );
        }
        DictArray::try_new(sliced_code, array.values().clone()).map(|a| a.into_array())
    }

    fn scalar_at(array: &DictArray, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = array.codes().scalar_at(index)?.as_ref().try_into()?;
        array.values().scalar_at(dict_index)
    }
}
