use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{VarBinArray, VarBinVTable, varbin_scalar};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{Array, ArrayRef, IntoArray};

impl OperationsVTable<VarBinVTable> for VarBinVTable {
    fn slice(array: &VarBinArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        VarBinArray::try_new(
            array.offsets().slice(start, stop + 1)?,
            array.bytes().clone(),
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )
        .map(|a| a.into_array())
    }

    fn scalar_at(array: &VarBinArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index)?, array.dtype()))
    }
}
