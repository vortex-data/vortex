mod compare;
mod filter;

use vortex_array::arrays::{VarBinArray, varbin_scalar};
use vortex_array::builders::ArrayBuilder;
use vortex_array::compute::{ScalarAtFn, TakeFn, fill_null, scalar_at, take};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayExt, ArrayRef};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::{Scalar, ScalarValue};

use crate::{FSSTArray, FSSTEncoding};

impl ComputeVTable for FSSTEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

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
                Scalar::new(
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

impl ScalarAtFn<&FSSTArray> for FSSTEncoding {
    fn scalar_at(&self, array: &FSSTArray, index: usize) -> VortexResult<Scalar> {
        let compressed = scalar_at(array.codes(), index)?;
        let binary_datum = compressed
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("expected null to already be handled"))?;

        let decoded_buffer =
            ByteBuffer::from(array.decompressor().decompress(binary_datum.as_slice()));
        Ok(varbin_scalar(decoded_buffer, array.dtype()))
    }
}
