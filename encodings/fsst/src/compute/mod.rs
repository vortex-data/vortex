mod compare;
mod filter;

use vortex_array::arrays::VarBinArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::compute::{TakeFn, fill_null, take};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayExt, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::{Scalar, ScalarValue};

use crate::{FSSTArray, FSSTEncoding};

impl ComputeVTable for FSSTEncoding {
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}

impl TakeFn<&FSSTArray> for FSSTEncoding {
    // Take on an FSSTArray is a simple take on the codes array.
    fn take(&self, array: &FSSTArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols().clone(),
            array.symbol_lengths().clone(),
            take(array.codes(), indices)?.as_::<VarBinArray>().clone(),
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

    fn take_into(
        &self,
        array: &FSSTArray,
        indices: &dyn Array,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        builder.extend_from_array(&take(array, indices)?)
    }
}
