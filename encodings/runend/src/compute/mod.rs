mod compare;
mod invert;

use std::cmp::min;
use std::ops::AddAssign;

use num_traits::AsPrimitive;
use vortex_array::array::{BooleanBuffer, PrimitiveArray};
use vortex_array::compute::{
    filter, scalar_at, slice, take, CompareFn, ComputeVTable, FilterFn, FilterMask, InvertFn,
    ScalarAtFn, SliceFn, TakeFn,
};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{match_each_integer_ptype, match_each_unsigned_integer_ptype, NativePType};
use vortex_error::{VortexResult, VortexUnwrap};
use vortex_scalar::Scalar;

use crate::{RunEndArray, RunEndEncoding};

impl ComputeVTable for RunEndEncoding {
    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
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

impl TakeFn<RunEndArray> for RunEndEncoding {
    #[allow(deprecated)]
    fn take(&self, array: &RunEndArray, indices: &ArrayData) -> VortexResult<ArrayData> {
        let primitive_indices = indices.clone().into_primitive()?;
        let usize_indices = match_each_integer_ptype!(primitive_indices.ptype(), |$P| {
            primitive_indices
                .into_maybe_null_slice::<$P>()
                .into_iter()
                .map(|idx| {
                    let usize_idx = idx as usize;
                    if usize_idx >= array.len() {
                        vortex_error::vortex_bail!(OutOfBounds: usize_idx, 0, array.len());
                    }

                    Ok(usize_idx + array.offset())
                })
                .collect::<VortexResult<Vec<usize>>>()?
        });
        let physical_indices = array
            .find_physical_indices(&usize_indices)?
            .into_iter()
            .map(|idx| idx as u64)
            .collect::<Vec<_>>();
        let physical_indices_array = PrimitiveArray::from(physical_indices).into_array();
        take(array.values(), &physical_indices_array)
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
            filter_run_ends(primitive_run_ends.maybe_null_slice::<$P>(), array.offset() as u64, array.len() as u64, mask)?
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
    let mut new_run_ends = vec![R::zero(); run_ends.len()];

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
    Ok((PrimitiveArray::from(new_run_ends), new_mask))
}

#[cfg(test)]
mod test {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::{filter, scalar_at, slice, take, FilterMask};
    use vortex_array::{ArrayDType, ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};
    use vortex_dtype::{DType, Nullability, PType};

    use crate::RunEndArray;

    pub(crate) fn ree_array() -> RunEndArray {
        RunEndArray::encode(
            PrimitiveArray::from(vec![1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).to_array(),
        )
        .unwrap()
    }

    #[test]
    fn ree_take() {
        let taken = take(
            ree_array().as_ref(),
            PrimitiveArray::from(vec![9, 8, 1, 3]).as_ref(),
        )
        .unwrap();
        assert_eq!(
            taken.into_primitive().unwrap().maybe_null_slice::<i32>(),
            &[5, 5, 1, 4]
        );
    }

    #[test]
    fn ree_take_end() {
        let taken = take(
            ree_array().as_ref(),
            PrimitiveArray::from(vec![11]).as_ref(),
        )
        .unwrap();
        assert_eq!(
            taken.into_primitive().unwrap().maybe_null_slice::<i32>(),
            &[5]
        );
    }

    #[test]
    #[should_panic]
    fn ree_take_out_of_bounds() {
        take(
            ree_array().as_ref(),
            PrimitiveArray::from(vec![12]).as_ref(),
        )
        .unwrap();
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
                vec![2u32, 5, 10].into_array(),
                vec![1i32, 2, 3].into_array(),
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
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            vec![2, 2, 3, 3, 3]
        );
    }

    #[test]
    fn double_slice() {
        let arr = slice(
            RunEndArray::try_new(
                vec![2u32, 5, 10].into_array(),
                vec![1i32, 2, 3].into_array(),
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
            doubly_sliced
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            vec![2, 2, 3]
        );
    }

    #[test]
    fn slice_end_inclusive() {
        let arr = slice(
            RunEndArray::try_new(
                vec![2u32, 5, 10].into_array(),
                vec![1i32, 2, 3].into_array(),
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
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            vec![2, 3, 3, 3, 3, 3]
        );
    }

    #[test]
    fn decompress() {
        let arr = RunEndArray::try_new(
            vec![2u32, 5, 10].into_array(),
            vec![1i32, 2, 3].into_array(),
        )
        .unwrap();

        assert_eq!(
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            vec![1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
        );
    }

    #[test]
    fn slice_at_end() {
        let re_array =
            RunEndArray::try_new(vec![7_u64, 10].into_array(), vec![2_u64, 3].into_array())
                .unwrap();

        assert_eq!(re_array.len(), 10);

        let sliced_array = slice(&re_array, re_array.len(), re_array.len()).unwrap();
        assert!(sliced_array.is_empty());

        let re_slice = RunEndArray::try_from(sliced_array).unwrap();
        assert!(re_slice.ends().is_empty());
        assert!(re_slice.values().is_empty())
    }

    #[test]
    fn sliced_take() {
        let sliced = slice(ree_array().as_ref(), 4, 9).unwrap();
        let taken = take(
            sliced.as_ref(),
            PrimitiveArray::from(vec![1, 3, 4]).as_ref(),
        )
        .unwrap();

        assert_eq!(taken.len(), 3);
        assert_eq!(scalar_at(taken.as_ref(), 0).unwrap(), 4.into());
        assert_eq!(scalar_at(taken.as_ref(), 1).unwrap(), 2.into());
        assert_eq!(scalar_at(taken.as_ref(), 2).unwrap(), 5.into());
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
                .maybe_null_slice::<u64>(),
            [2, 4]
        );
        assert_eq!(
            filtered_run_end
                .values()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
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
                .maybe_null_slice::<u64>(),
            [1, 2, 3]
        );
        assert_eq!(
            filtered_run_end
                .values()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<i32>(),
            [1, 4, 2]
        );
    }
}
