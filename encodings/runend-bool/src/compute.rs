use vortex_array::array::BoolArray;
use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{slice, ComputeVTable, SliceFn, TakeFn, TakeOptions};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::Scalar;

use crate::{value_at_index, RunEndBoolArray, RunEndBoolEncoding};

impl ComputeVTable for RunEndBoolEncoding {
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

impl ScalarAtFn<RunEndBoolArray> for RunEndBoolEncoding {
    fn scalar_at(&self, array: &RunEndBoolArray, index: usize) -> VortexResult<Scalar> {
        let start = array.start();
        Ok(Scalar::from(value_at_index(
            array.find_physical_index(index)?,
            start,
        )))
    }
}

impl TakeFn<RunEndBoolArray> for RunEndBoolEncoding {
    fn take(
        &self,
        array: &RunEndBoolArray,
        indices: &ArrayData,
        _options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        let primitive_indices = indices.clone().into_primitive()?;
        let physical_indices = match_each_integer_ptype!(primitive_indices.ptype(), |$P| {
            primitive_indices
                .into_maybe_null_slice::<$P>()
                .into_iter()
                .map(|idx| idx as usize)
                .map(|idx| {
                    if idx >= array.len() {
                        vortex_bail!(OutOfBounds: idx, 0, array.len())
                    }
                    array.find_physical_index(idx)
                })
                .collect::<VortexResult<Vec<_>>>()?
        });
        let start = array.start();
        Ok(
            BoolArray::from_iter(physical_indices.iter().map(|&it| value_at_index(it, start)))
                .to_array(),
        )
    }
}

impl SliceFn<RunEndBoolArray> for RunEndBoolEncoding {
    fn slice(&self, array: &RunEndBoolArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let slice_begin = array.find_physical_index(start)?;
        let slice_end = array.find_physical_index(stop)?;

        Ok(RunEndBoolArray::with_offset_and_size(
            slice(array.ends(), slice_begin, slice_end + 1)?,
            value_at_index(slice_begin, array.start()),
            array.validity().slice(slice_begin, slice_end + 1)?,
            stop - start,
            start + array.offset(),
        )?
        .into_array())
    }
}
