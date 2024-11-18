use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::compute::{FilterFn, FilterMask};
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData};

impl FilterFn for PrimitiveArray {
    fn filter(&self, mask: &FilterMask) -> VortexResult<ArrayData> {
        filter_select_primitive(self, mask).map(|a| a.into_array())
    }
}

fn filter_select_primitive(
    arr: &PrimitiveArray,
    mask: &FilterMask,
) -> VortexResult<PrimitiveArray> {
    let validity = arr.validity().filter(mask)?;
    let selection_count = mask.true_count();
    match_each_native_ptype!(arr.ptype(), |$T| {
        let slice = arr.maybe_null_slice::<$T>();
        Ok(PrimitiveArray::from_vec(filter_primitive_slice(slice, mask, selection_count)?, validity))
    })
}

pub fn filter_primitive_slice<T: NativePType>(
    arr: &[T],
    mask: &FilterMask,
    selection_count: usize,
) -> VortexResult<Vec<T>> {
    let mut chunks = Vec::with_capacity(selection_count);
    if selection_count * 2 > mask.len() {
        mask.iter_slices()?.for_each(|(start, end)| {
            chunks.extend_from_slice(&arr[start..end]);
        });
    } else {
        chunks.extend(mask.iter_indices()?.map(|idx| arr[idx]));
    }
    Ok(chunks)
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::array::primitive::compute::filter::filter_select_primitive;
    use crate::array::primitive::PrimitiveArray;
    use crate::compute::FilterMask;

    #[test]
    fn filter_run_variant_mixed_test() {
        let filter = [true, true, false, true, true, true, false, true];
        let arr = PrimitiveArray::from(vec![1u32, 24, 54, 2, 3, 2, 3, 2]);

        let filtered =
            filter_select_primitive(&arr, &FilterMask::from_iter(filter.iter().copied())).unwrap();
        assert_eq!(
            filtered.len(),
            filter.iter().filter(|x| **x).collect_vec().len()
        );

        let rust_arr = arr.maybe_null_slice::<u32>();
        assert_eq!(
            filtered.maybe_null_slice::<u32>().to_vec(),
            filter
                .iter()
                .enumerate()
                .filter(|(_idx, b)| **b)
                .map(|m| rust_arr[m.0])
                .collect_vec()
        )
    }
}
