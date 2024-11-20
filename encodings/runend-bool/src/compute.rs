use vortex_array::array::BoolArray;
use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{slice, ArrayCompute, ComputeVTable, SliceFn, TakeFn, TakeOptions};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use crate::{value_at_index, RunEndBoolArray, RunEndBoolEncoding};

impl ArrayCompute for RunEndBoolArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl ComputeVTable for RunEndBoolEncoding {
    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn for RunEndBoolArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let start = self.start();
        Ok(Scalar::from(value_at_index(
            self.find_physical_index(index)?,
            start,
        )))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        let start = self.start();
        Scalar::from(value_at_index(
            self.find_physical_index(index)
                .vortex_expect("Search must be implemented for the underlying index array"),
            start,
        ))
    }
}

impl TakeFn for RunEndBoolArray {
    fn take(&self, indices: &ArrayData, _options: TakeOptions) -> VortexResult<ArrayData> {
        let primitive_indices = indices.clone().into_primitive()?;
        let physical_indices = match_each_integer_ptype!(primitive_indices.ptype(), |$P| {
            primitive_indices
                .maybe_null_slice::<$P>()
                .iter()
                .map(|idx| *idx as usize)
                .map(|idx| {
                    if idx >= self.len() {
                        vortex_bail!(OutOfBounds: idx, 0, self.len())
                    }
                    self.find_physical_index(idx)
                })
                .collect::<VortexResult<Vec<_>>>()?
        });
        let start = self.start();
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
            start,
        )?
        .into_array())
    }
}
