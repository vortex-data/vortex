use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use vortex_error::VortexResult;

use crate::array::BoolArray;
use crate::compute::{FilterFn, FilterMask};
use crate::{ArrayData, IntoArrayData};

impl FilterFn for BoolArray {
    fn filter(&self, mask: &FilterMask) -> VortexResult<ArrayData> {
        filter_select_bool(self, mask).map(|a| a.into_array())
    }
}

fn filter_select_bool(arr: &BoolArray, mask: &FilterMask) -> VortexResult<BoolArray> {
    let validity = arr.validity().filter(mask)?;

    let selection_count = mask.true_count();
    let out = if selection_count * 2 > arr.len() {
        filter_select_bool_by_slice(&arr.boolean_buffer(), mask, selection_count)
    } else {
        filter_select_bool_by_index(&arr.boolean_buffer(), mask, selection_count)
    };
    BoolArray::try_new(out?, validity)
}

fn filter_select_bool_by_slice(
    values: &BooleanBuffer,
    mask: &FilterMask,
    selection_count: usize,
) -> VortexResult<BooleanBuffer> {
    let mut out_buf = BooleanBufferBuilder::new(selection_count);
    mask.iter_slices()?.for_each(|(start, end)| {
        out_buf.append_buffer(&values.slice(start, end - start));
    });
    Ok(out_buf.finish())
}

fn filter_select_bool_by_index(
    values: &BooleanBuffer,
    mask: &FilterMask,
    selection_count: usize,
) -> VortexResult<BooleanBuffer> {
    let mut out_buf = BooleanBufferBuilder::new(selection_count);
    mask.iter_indices()?
        .for_each(|idx| out_buf.append(values.value(idx)));
    Ok(out_buf.finish())
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::array::bool::compute::filter::{
        filter_select_bool, filter_select_bool_by_index, filter_select_bool_by_slice,
    };
    use crate::array::BoolArray;
    use crate::compute::FilterMask;

    #[test]
    fn filter_bool_test() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = FilterMask::from_iter([true, false, true]);

        let filtered = filter_select_bool(&arr, &mask).unwrap();
        assert_eq!(2, filtered.len());

        assert_eq!(
            vec![true, false],
            filtered.boolean_buffer().iter().collect_vec()
        )
    }

    #[test]
    fn filter_bool_by_slice_test() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = FilterMask::from_iter([true, false, true]);

        let filtered = filter_select_bool_by_slice(&arr.boolean_buffer(), &mask, 2).unwrap();
        assert_eq!(2, filtered.len());

        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }

    #[test]
    fn filter_bool_by_index_test() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = FilterMask::from_iter([true, false, true]);

        let filtered = filter_select_bool_by_index(&arr.boolean_buffer(), &mask, 2).unwrap();
        assert_eq!(2, filtered.len());

        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }
}
