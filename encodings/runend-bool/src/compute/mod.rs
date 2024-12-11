mod invert;

use vortex_array::array::BoolArray;
use vortex_array::compute::{slice, ComputeVTable, InvertFn, ScalarAtFn, SliceFn, TakeFn};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};
use vortex_dtype::match_each_integer_ptype;
use vortex_error::{vortex_bail, VortexResult};
use vortex_scalar::Scalar;

use crate::{value_at_index, RunEndBoolArray, RunEndBoolEncoding};

impl ComputeVTable for RunEndBoolEncoding {
    fn invert_fn(&self) -> Option<&dyn InvertFn<ArrayData>> {
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
    fn take(&self, array: &RunEndBoolArray, indices: &ArrayData) -> VortexResult<ArrayData> {
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
        let new_length = stop - start;

        let (slice_begin, slice_end) = if new_length == 0 {
            let ends_len = array.ends().len();
            (ends_len, ends_len)
        } else {
            let physical_begin = array.find_physical_index(start)?;
            let physical_end = array.find_physical_index(stop)?;
            (physical_begin, physical_end + 1)
        };

        Ok(RunEndBoolArray::with_offset_and_size(
            slice(array.ends(), slice_begin, slice_end)?,
            value_at_index(slice_begin, array.start()),
            array.validity().slice(start, stop)?,
            new_length,
            if new_length == 0 {
                0
            } else {
                start + array.offset()
            },
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::slice;
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayLen, IntoArrayData};

    use crate::RunEndBoolArray;

    #[test]
    fn slice_at_end() {
        let re_array =
            RunEndBoolArray::try_new(vec![7_u64, 10].into_array(), false, Validity::NonNullable)
                .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = slice(&re_array, re_array.len(), re_array.len()).unwrap();
        assert!(sliced_array.is_empty());

        let re_slice = RunEndBoolArray::try_from(sliced_array).unwrap();
        assert!(re_slice.ends().is_empty());
    }
}
