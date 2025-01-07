mod binary_numeric;
mod compare;
mod fill_null;
mod invert;
mod take;

use std::cmp::min;
use std::ops::AddAssign;

use num_traits::AsPrimitive;
use vortex_array::array::{BooleanBuffer, PrimitiveArray};
use vortex_array::compute::{
    filter, scalar_at, slice, BinaryNumericFn, CompareFn, ComputeVTable, FillNullFn, FilterFn,
    FilterMask, InvertFn, ScalarAtFn, SliceFn, TakeFn,
};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_buffer::buffer_mut;
use vortex_dtype::{match_each_unsigned_integer_ptype, NativePType};
use vortex_error::{VortexResult, VortexUnwrap};
use vortex_scalar::Scalar;

use crate::{RunEndArray, RunEndEncoding};

impl ComputeVTable for RunEndEncoding {
    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<ArrayData>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<ArrayData>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

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

impl ScalarAtFn<RunEndArray> for RunEndEncoding {
    fn scalar_at(&self, array: &RunEndArray, index: usize) -> VortexResult<Scalar> {
        scalar_at(array.values(), array.find_physical_index(index)?)
    }
}

impl SliceFn<RunEndArray> for RunEndEncoding {
    fn slice(&self, array: &RunEndArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let new_length = stop - start;

        let (slice_begin, slice_end) = if new_length == 0 {
            let values_len = array.values().len();
            (values_len, values_len)
        } else {
            let physical_start = array.find_physical_index(start)?;
            let physical_stop = array.find_physical_index(stop)?;

            (physical_start, physical_stop + 1)
        };

        Ok(RunEndArray::with_offset_and_length(
            slice(array.ends(), slice_begin, slice_end)?,
            slice(array.values(), slice_begin, slice_end)?,
            if new_length == 0 {
                0
            } else {
                start + array.offset()
            },
            new_length,
        )?
        .into_array())
    }
}

impl FilterFn<RunEndArray> for RunEndEncoding {
    fn filter(&self, array: &RunEndArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let primitive_run_ends = array.ends().into_primitive()?;
        let (run_ends, values_mask) = match_each_unsigned_integer_ptype!(primitive_run_ends.ptype(), |$P| {
            filter_run_ends(primitive_run_ends.as_slice::<$P>(), array.offset() as u64, array.len() as u64, mask)?
        });
        let values = filter(&array.values(), values_mask)?;

        RunEndArray::try_new(run_ends.into_array(), values).map(|a| a.into_array())
    }
}

// Code adapted from apache arrow-rs https://github.com/apache/arrow-rs/blob/b1f5c250ebb6c1252b4e7c51d15b8e77f4c361fa/arrow-select/src/filter.rs#L425
fn filter_run_ends<R: NativePType + AddAssign + From<bool> + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: FilterMask,
) -> VortexResult<(PrimitiveArray, FilterMask)> {
    let mut new_run_ends = buffer_mut![R::zero(); run_ends.len()];

    let mut start = 0u64;
    let mut j = 0;
    let mut count = R::zero();
    let filter_values = mask.to_boolean_buffer()?;

    let new_mask: FilterMask = BooleanBuffer::collect_bool(run_ends.len(), |i| {
        let mut keep = false;
        let end = min(run_ends[i].as_() - offset, length);

        // Safety: predicate must be the same length as the array the ends have been taken from
        for pred in (start..end)
            .map(|i| unsafe { filter_values.value_unchecked(i.try_into().vortex_unwrap()) })
        {
            count += <R as From<bool>>::from(pred);
            keep |= pred
        }
        // this is to avoid branching
        new_run_ends[j] = count;
        j += keep as usize;

        start = end;
        keep
    })
    .into();

    new_run_ends.truncate(j);
    Ok((
        PrimitiveArray::new(new_run_ends, Validity::NonNullable),
        new_mask,
    ))
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::test_harness::test_binary_numeric;
    use vortex_array::compute::{filter, scalar_at, slice, FilterMask};
    use vortex_array::{ArrayDType, ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::RunEndArray;

    pub(crate) fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).to_array(),
        )
        .unwrap()
    }

    #[test]
    fn ree_scalar_at_end() {
        let scalar = scalar_at(ree_array().as_ref(), 11).unwrap();
        assert_eq!(scalar, 5.into());
    }

    #[test]
    fn slice_array() {
        let arr = slice(
            RunEndArray::try_new(
                buffer![2u32, 5, 10].into_array(),
                buffer![1i32, 2, 3].into_array(),
            )
            .unwrap()
            .as_ref(),
            3,
            8,
        )
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 5);

        assert_eq!(
            arr.into_primitive().unwrap().as_slice::<i32>(),
            vec![2, 2, 3, 3, 3]
        );
    }

    #[test]
    fn double_slice() {
        let arr = slice(
            RunEndArray::try_new(
                buffer![2u32, 5, 10].into_array(),
                buffer![1i32, 2, 3].into_array(),
            )
            .unwrap()
            .as_ref(),
            3,
            8,
        )
        .unwrap();
        assert_eq!(arr.len(), 5);

        let doubly_sliced = slice(&arr, 0, 3).unwrap();

        assert_eq!(
            doubly_sliced.into_primitive().unwrap().as_slice::<i32>(),
            vec![2, 2, 3]
        );
    }

    #[test]
    fn slice_end_inclusive() {
        let arr = slice(
            RunEndArray::try_new(
                buffer![2u32, 5, 10].into_array(),
                buffer![1i32, 2, 3].into_array(),
            )
            .unwrap()
            .as_ref(),
            4,
            10,
        )
        .unwrap();
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(arr.len(), 6);

        assert_eq!(
            arr.into_primitive().unwrap().as_slice::<i32>(),
            vec![2, 3, 3, 3, 3, 3]
        );
    }

    #[test]
    fn decompress() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap();

        assert_eq!(
            arr.into_primitive().unwrap().as_slice::<i32>(),
            vec![1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
        );
    }

    #[test]
    fn slice_at_end() {
        let re_array = RunEndArray::try_new(
            buffer![7_u64, 10].into_array(),
            buffer![2_u64, 3].into_array(),
        )
        .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = slice(&re_array, re_array.len(), re_array.len()).unwrap();
        assert!(sliced_array.is_empty());

        let re_slice = RunEndArray::try_from(sliced_array).unwrap();
        assert!(re_slice.ends().is_empty());
        assert!(re_slice.values().is_empty())
    }

    #[test]
    fn filter_run_end() {
        let arr = ree_array();
        let filtered = filter(
            arr.as_ref(),
            FilterMask::from_iter([
                true, true, false, false, false, false, false, false, false, false, true, true,
            ]),
        )
        .unwrap();
        let filtered_run_end = RunEndArray::try_from(filtered).unwrap();

        assert_eq!(
            filtered_run_end
                .ends()
                .into_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [2, 4]
        );
        assert_eq!(
            filtered_run_end
                .values()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [1, 5]
        );
    }

    #[test]
    fn filter_sliced_run_end() {
        let arr = slice(ree_array(), 2, 7).unwrap();
        let filtered = filter(
            &arr,
            FilterMask::from_iter([true, false, false, true, true]),
        )
        .unwrap();
        let filtered_run_end = RunEndArray::try_from(filtered).unwrap();

        assert_eq!(
            filtered_run_end
                .ends()
                .into_primitive()
                .unwrap()
                .as_slice::<u64>(),
            [1, 2, 3]
        );
        assert_eq!(
            filtered_run_end
                .values()
                .into_primitive()
                .unwrap()
                .as_slice::<i32>(),
            [1, 4, 2]
        );
    }

    #[test]
    fn test_runend_binary_numeric() {
        let array = ree_array().into_array();
        test_binary_numeric::<i32>(array)
    }
}
