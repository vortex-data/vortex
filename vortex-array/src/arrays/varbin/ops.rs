use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::accessor::ArrayAccessor;
use crate::arrays::{VarBinArray, VarBinVTable, varbin_scalar};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{Array, ArrayRef, Cost, IntoArray};

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

    fn is_constant(array: &VarBinArray, cost: Cost) -> VortexResult<Option<bool>> {
        if cost.is_negligible() {
            return Ok(None);
        }
        array
            .with_iterator(|iter| {
                let a = iter
                    .next()
                    .vortex_expect("is_constant is only invoked for len > 1");
                for x in iter {
                    if a != x {
                        return false;
                    }
                }
                true
            })
            .map(Some)
    }
}
