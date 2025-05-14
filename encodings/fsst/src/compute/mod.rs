mod compare;
mod filter;

use vortex_array::arrays::VarBinVTable;
use vortex_array::compute::{TakeKernel, TakeKernelAdapter, fill_null, take};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_scalar::{Scalar, ScalarValue};

use crate::{FSSTArray, FSSTVTable};

impl TakeKernel for FSSTVTable {
    // Take on an FSSTArray is a simple take on the codes array.
    fn take(&self, array: &FSSTArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols().clone(),
            array.symbol_lengths().clone(),
            take(array.codes().as_ref(), indices)?
                .as_::<VarBinVTable>()
                .clone(),
            fill_null(
                &take(array.uncompressed_lengths(), indices)?,
                &Scalar::new(
                    array.uncompressed_lengths_dtype().clone(),
                    ScalarValue::from(0),
                ),
            )?,
        )?
        .into_array())
    }
}

register_kernel!(TakeKernelAdapter(FSSTVTable).lift());
