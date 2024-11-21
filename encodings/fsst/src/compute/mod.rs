mod compare;

use vortex_array::array::varbin_scalar;
use vortex_array::compute::unary::{scalar_at, ScalarAtFn};
use vortex_array::compute::{
    filter, slice, take, ArrayCompute, CompareFn, ComputeVTable, FilterFn, FilterMask, SliceFn,
    TakeFn, TakeOptions,
};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_buffer::Buffer;
use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::Scalar;

use crate::{FSSTArray, FSSTEncoding};

impl ArrayCompute for FSSTArray {
    fn compare(&self) -> Option<&dyn CompareFn> {
        Some(self)
    }
}

impl ComputeVTable for FSSTEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl SliceFn<FSSTArray> for FSSTEncoding {
    fn slice(&self, array: &FSSTArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        // Slicing an FSST array leaves the symbol table unmodified,
        // only slicing the `codes` array.
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols(),
            array.symbol_lengths(),
            slice(array.codes(), start, stop)?,
            slice(array.uncompressed_lengths(), start, stop)?,
        )?
        .into_array())
    }
}

impl TakeFn<FSSTArray> for FSSTEncoding {
    // Take on an FSSTArray is a simple take on the codes array.
    fn take(
        &self,
        array: &FSSTArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols(),
            array.symbol_lengths(),
            take(array.codes(), indices, options)?,
            take(array.uncompressed_lengths(), indices, options)?,
        )?
        .into_array())
    }
}

impl ScalarAtFn<FSSTArray> for FSSTEncoding {
    fn scalar_at(&self, array: &FSSTArray, index: usize) -> VortexResult<Scalar> {
        let compressed = scalar_at(array.codes(), index)?;
        let binary_datum = compressed
            .value()
            .as_buffer()?
            .ok_or_else(|| vortex_err!("Expected a binary scalar, found {}", compressed.dtype()))?;

        array.with_decompressor(|decompressor| {
            let decoded_buffer: Buffer = decompressor.decompress(binary_datum.as_slice()).into();
            Ok(varbin_scalar(decoded_buffer, array.dtype()))
        })
    }
}

impl FilterFn<FSSTArray> for FSSTEncoding {
    // Filtering an FSSTArray filters the codes array, leaving the symbols array untouched
    fn filter(&self, array: &FSSTArray, mask: FilterMask) -> VortexResult<ArrayData> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols(),
            array.symbol_lengths(),
            filter(&array.codes(), mask.clone())?,
            filter(&array.uncompressed_lengths(), mask)?,
        )?
        .into_array())
    }
}
