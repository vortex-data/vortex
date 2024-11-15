use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use vortex_error::{vortex_err, VortexResult};

use crate::array::BoolArray;
use crate::compute::FilterFn;
use crate::variants::BoolArrayTrait;
use crate::{ArrayData, IntoArrayData};

impl FilterFn for BoolArray {
    fn filter(&self, predicate: &ArrayData) -> VortexResult<ArrayData> {
        filter_select_bool(self, predicate).map(|a| a.into_array())
    }
}

fn filter_select_bool(arr: &BoolArray, predicate: &ArrayData) -> VortexResult<BoolArray> {
    predicate.with_dyn(|b| {
        let validity = arr.validity().filter(predicate)?;
        let predicate = b.as_bool_array().ok_or(vortex_err!(
            NotImplemented: "as_bool_array",
            predicate.encoding().id()
        ))?;
        let selection_count = predicate.true_count();
        let out = if selection_count * 2 > arr.len() {
            filter_select_bool_by_slice(&arr.boolean_buffer(), predicate, selection_count)
        } else {
            filter_select_bool_by_index(&arr.boolean_buffer(), predicate, selection_count)
        };
        BoolArray::try_new(out, validity)
    })
}

fn filter_select_bool_by_slice(
    values: &BooleanBuffer,
    predicate: &dyn BoolArrayTrait,
    selection_count: usize,
) -> BooleanBuffer {
    let mut out_buf = BooleanBufferBuilder::new(selection_count);
    predicate.maybe_null_slices_iter().for_each(|(start, end)| {
        out_buf.append_buffer(&values.slice(start, end - start));
    });
    out_buf.finish()
}

fn filter_select_bool_by_index(
    values: &BooleanBuffer,
    predicate: &dyn BoolArrayTrait,
    selection_count: usize,
) -> BooleanBuffer {
    let mut out_buf = BooleanBufferBuilder::new(selection_count);
    predicate
        .maybe_null_indices_iter()
        .for_each(|idx| out_buf.append(values.value(idx)));
    out_buf.finish()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::array::bool::compute::filter::{
        filter_select_bool, filter_select_bool_by_index, filter_select_bool_by_slice,
    };
    use crate::array::BoolArray;
    use crate::ToArrayData;

    #[test]
    fn filter_bool_test() {
        let arr = BoolArray::from(vec![true, true, false]);
        let filter = BoolArray::from(vec![true, false, true]);

        let filtered = filter_select_bool(&arr, &filter.to_array()).unwrap();
        assert_eq!(2, filtered.len());

        assert_eq!(
            vec![true, false],
            filtered.boolean_buffer().iter().collect_vec()
        )
    }

    #[test]
    fn filter_bool_by_slice_test() {
        let arr = BoolArray::from(vec![true, true, false]);
        let filter = BoolArray::from(vec![true, false, true]);

        let filtered = filter_select_bool_by_slice(&arr.boolean_buffer(), &filter, 2);
        assert_eq!(2, filtered.len());

        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }

    #[test]
    fn filter_bool_by_index_test() {
        let arr = BoolArray::from(vec![true, true, false]);
        let filter = BoolArray::from(vec![true, false, true]);

        let filtered = filter_select_bool_by_index(&arr.boolean_buffer(), &filter, 2);
        assert_eq!(2, filtered.len());

        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }
}
