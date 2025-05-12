use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{DictArray, DictVTable};

impl OperationsVTable<DictVTable> for DictVTable {
    fn slice(array: &DictArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        DictArray::try_new(array.codes().slice(start, stop)?, array.values().clone())
            .map(|a| a.into_array())
    }

    fn scalar_at(array: &DictArray, index: usize) -> VortexResult<Scalar> {
        let dict_index: usize = array.codes().scalar_at(index)?.as_ref().try_into()?;
        array.values().scalar_at(dict_index)
    }
}
