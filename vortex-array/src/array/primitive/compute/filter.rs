use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::array::PrimitiveEncoding;
use crate::compute::{FilterFn, FilterIter, FilterMask};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData};

impl FilterFn<PrimitiveArray> for PrimitiveEncoding {
    fn filter(&self, array: &PrimitiveArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let validity = array.validity().filter(&mask)?;
        match_each_native_ptype!(array.ptype(), |$T| {
            let values = match mask.iter()? {
                FilterIter::Indices(indices) => filter_primitive_indices(array.maybe_null_slice::<$T>(), indices.iter().copied()),
                FilterIter::IndicesIter(iter) => filter_primitive_indices(array.maybe_null_slice::<$T>(), iter),
                FilterIter::Slices(slices) => filter_primitive_slices(array.maybe_null_slice::<$T>(), mask.true_count(), slices.iter().copied()),
                FilterIter::SlicesIter(iter) => filter_primitive_slices(array.maybe_null_slice::<$T>(), mask.true_count(), iter),
            };
            Ok(PrimitiveArray::from_vec(values, validity).into_array())
        })
    }
}

fn filter_primitive_indices<T: Copy>(values: &[T], indices: impl Iterator<Item = usize>) -> Vec<T> {
    indices
        .map(|idx| *unsafe { values.get_unchecked(idx) })
        .collect()
}

fn filter_primitive_slices<T: Clone>(
    values: &[T],
    indices_len: usize,
    indices: impl Iterator<Item = (usize, usize)>,
) -> Vec<T> {
    let mut output = Vec::with_capacity(indices_len);
    for (start, end) in indices {
        output.extend_from_slice(&values[start..end]);
    }
    output
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::array::primitive::PrimitiveArray;
    use crate::compute::{filter, FilterMask};
    use crate::{ArrayLen, IntoArrayVariant, ToArrayData};

    #[test]
    fn filter_run_variant_mixed_test() {
        let mask = [true, true, false, true, true, true, false, true];
        let arr = PrimitiveArray::from(vec![1u32, 24, 54, 2, 3, 2, 3, 2]);

        let filtered = filter(&arr.to_array(), FilterMask::from_iter(mask))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(
            filtered.len(),
            mask.iter().filter(|x| **x).collect_vec().len()
        );

        let rust_arr = arr.maybe_null_slice::<u32>();
        assert_eq!(
            filtered.maybe_null_slice::<u32>().to_vec(),
            mask.iter()
                .enumerate()
                .filter(|(_idx, b)| **b)
                .map(|m| rust_arr[m.0])
                .collect_vec()
        )
    }
}
