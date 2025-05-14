use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{VarBinViewArray, VarBinViewVTable, varbin_scalar};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperationsVTable<VarBinViewVTable> for VarBinViewVTable {
    fn slice(array: &VarBinViewArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let views = array.views().slice(start..stop);

        Ok(VarBinViewArray::try_new(
            views,
            array.buffers().to_vec(),
            array.dtype().clone(),
            array.validity().slice(start, stop)?,
        )?
        .into_array())
    }

    fn scalar_at(array: &VarBinViewArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index), array.dtype()))
    }
}
