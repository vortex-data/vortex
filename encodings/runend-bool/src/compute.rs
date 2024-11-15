use vortex_array::array::BoolArray;
use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{slice, ArrayCompute, SliceFn, TakeFn};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant, ToArrayData};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use crate::compress::value_at_index;
use crate::RunEndBoolArray;

impl ArrayCompute for RunEndBoolArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
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
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
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
        Ok(BoolArray::from(
            physical_indices
                .iter()
                .map(|&it| value_at_index(it, start))
                .collect::<Vec<_>>(),
        )
        .to_array())
    }
}

impl SliceFn for RunEndBoolArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let slice_begin = self.find_physical_index(start)?;
        let slice_end = self.find_physical_index(stop)?;

        Ok(Self::with_offset_and_size(
            slice(self.ends(), slice_begin, slice_end + 1)?,
            value_at_index(slice_begin, self.start()),
            self.validity().slice(slice_begin, slice_end + 1)?,
            stop - start,
            start,
        )?
        .into_array())
    }
}
