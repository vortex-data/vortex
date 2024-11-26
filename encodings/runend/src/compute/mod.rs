mod compare;

use std::cmp::min;
use std::ops::AddAssign;

use num_traits::AsPrimitive;
use vortex_array::array::{BooleanBuffer, ConstantArray, PrimitiveArray, SparseArray};
use vortex_array::compute::unary::{scalar_at, ScalarAtFn};
use vortex_array::compute::{
    filter, slice, take, CompareFn, ComputeVTable, FilterFn, FilterMask, SliceFn, TakeFn,
    TakeOptions,
};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant};
use vortex_dtype::{match_each_integer_ptype, match_each_unsigned_integer_ptype, NativePType};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{RunEndArray, RunEndEncoding};

impl ComputeVTable for RunEndEncoding {
    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
        Some(self)
    }

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

impl ScalarAtFn<RunEndArray> for RunEndEncoding {
    fn scalar_at(&self, array: &RunEndArray, index: usize) -> VortexResult<Scalar> {
        scalar_at(array.values(), array.find_physical_index(index)?)
    }
}

impl TakeFn<RunEndArray> for RunEndEncoding {
    #[allow(deprecated)]
    fn take(
        &self,
        array: &RunEndArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
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
        let dense_values = take(array.values(), &physical_indices_array, options)?;

        Ok(match array.validity() {
            Validity::NonNullable => dense_values,
            Validity::AllValid => dense_values,
            Validity::AllInvalid => {
                ConstantArray::new(Scalar::null(array.dtype().clone()), indices.len()).into_array()
            }
            Validity::Array(original_validity) => {
                let dense_validity =
                    FilterMask::try_from(take(&original_validity, indices, options)?)?;
                let length = dense_validity.len();
                let dense_nonnull_indices = PrimitiveArray::from(
                    dense_validity
                        .iter_indices()?
                        .map(|idx| idx as u64)
                        .collect::<Vec<_>>(),
                )
                .into_array();
                let filtered_values = filter(&dense_values, dense_validity)?;
                let dtype = filtered_values.dtype().clone();

                SparseArray::try_new(
                    dense_nonnull_indices,
                    filtered_values,
                    length,
                    Scalar::null(dtype),
                )?
                .into_array()
            }
        })
    }
}

impl SliceFn<RunEndArray> for RunEndEncoding {
    fn slice(&self, array: &RunEndArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let slice_begin = array.find_physical_index(start)?;
        let slice_end = array.find_physical_index(stop)?;

        Ok(RunEndArray::with_offset_and_length(
            slice(array.ends(), slice_begin, slice_end + 1)?,
            slice(array.values(), slice_begin, slice_end + 1)?,
            array.validity().slice(start, stop)?,
            start + array.offset(),
            stop - start,
        )?
        .into_array())
    }
}

impl FilterFn<RunEndArray> for RunEndEncoding {
    fn filter(&self, array: &RunEndArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let validity = array.validity().filter(&mask)?;
        let primitive_run_ends = array.ends().into_primitive()?;
        let (run_ends, values_mask) = match_each_unsigned_integer_ptype!(primitive_run_ends.ptype(), |$P| {
            filter_run_ends(primitive_run_ends.maybe_null_slice::<$P>(), array.offset() as u64, array.len() as u64, mask)?
        });
        let values = filter(&array.values(), values_mask)?;

        RunEndArray::try_new(run_ends.into_array(), values, validity).map(|a| a.into_array())
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
        for pred in (start..end).map(|i| unsafe { filter_values.value_unchecked(i as usize) }) {
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
    use vortex_array::array::{BoolArray, PrimitiveArray};
    use vortex_array::compute::unary::{scalar_at, try_cast};
    use vortex_array::compute::{filter, slice, take, FilterMask, TakeOptions};
    use vortex_array::validity::{ArrayValidity, Validity};
    use vortex_array::{ArrayDType, ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_scalar::Scalar;

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
            TakeOptions::default(),
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
            TakeOptions::default(),
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
            TakeOptions::default(),
        )
        .unwrap();
    }

    #[test]
    fn ree_scalar_at_end() {
        let scalar = scalar_at(ree_array().as_ref(), 11).unwrap();
        assert_eq!(scalar, 5.into());
    }

    #[test]
    fn ree_null_scalar() {
        let array = ree_array();
        let null_ree = RunEndArray::try_new(
            array.ends(),
            try_cast(array.values(), &array.values().dtype().as_nullable()).unwrap(),
            Validity::AllInvalid,
        )
        .unwrap();
        let scalar = scalar_at(null_ree.as_ref(), 11).unwrap();
        assert_eq!(scalar, Scalar::null(null_ree.dtype().clone()));
    }

    #[test]
    fn slice_with_nulls() {
        let array = RunEndArray::try_new(
            PrimitiveArray::from(vec![3u32, 6, 8, 12]).into_array(),
            PrimitiveArray::from_vec(vec![1, 4, 2, 5], Validity::AllValid).into_array(),
            Validity::from_iter([
                false, false, false, false, true, true, false, false, false, false, true, true,
            ]),
        )
        .unwrap();
        let sliced = slice(array.as_ref(), 4, 10).unwrap();
        let sliced_primitive = sliced.into_primitive().unwrap();
        assert_eq!(
            sliced_primitive.maybe_null_slice::<i32>(),
            vec![4, 4, 2, 2, 5, 5]
        );
        assert_eq!(
            sliced_primitive
                .logical_validity()
                .into_array()
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, true, false, false, false, false]
        )
    }

    #[test]
    fn slice_array() {
        let arr = slice(
            RunEndArray::try_new(
                vec![2u32, 5, 10].into_array(),
                vec![1i32, 2, 3].into_array(),
                Validity::NonNullable,
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
                Validity::NonNullable,
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
                Validity::NonNullable,
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
            Validity::NonNullable,
        )
        .unwrap();

        assert_eq!(
            arr.into_primitive().unwrap().maybe_null_slice::<i32>(),
            vec![1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
        );
    }

    #[test]
    fn take_with_nulls() {
        let uncompressed = PrimitiveArray::from_vec(vec![1i32, 0, 3], Validity::AllValid);
        let validity = BoolArray::from_iter([
            true, true, false, false, false, true, true, true, true, true,
        ]);
        let arr = RunEndArray::try_new(
            vec![2u32, 5, 10].into_array(),
            uncompressed.into_array(),
            Validity::Array(validity.into_array()),
        )
        .unwrap();

        let test_indices = PrimitiveArray::from_vec(vec![0, 2, 4, 6], Validity::NonNullable);
        let taken = take(arr.as_ref(), test_indices.as_ref(), TakeOptions::default()).unwrap();

        assert_eq!(taken.len(), test_indices.len());

        let parray = taken.into_primitive().unwrap();
        assert_eq!(
            (0..4)
                .map(|idx| parray.is_valid(idx).then(|| parray.get_as_cast::<i32>(idx)))
                .collect::<Vec<Option<i32>>>(),
            vec![Some(1), None, None, Some(3),]
        );
    }

    #[test]
    fn sliced_take() {
        let sliced = slice(ree_array().as_ref(), 4, 9).unwrap();
        let taken = take(
            sliced.as_ref(),
            PrimitiveArray::from(vec![1, 3, 4]).as_ref(),
            TakeOptions::default(),
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
