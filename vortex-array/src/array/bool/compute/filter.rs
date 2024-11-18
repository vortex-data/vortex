use arrow_buffer::{bit_util, BooleanBuffer, BooleanBufferBuilder};
use vortex_error::{VortexExpect, VortexResult};

use crate::array::BoolArray;
use crate::compute::{FilterFn, FilterIter, FilterMask};
use crate::{ArrayData, IntoArrayData};

impl FilterFn for BoolArray {
    fn filter(&self, mask: FilterMask) -> VortexResult<ArrayData> {
        let validity = self.validity().filter(&mask)?;

        let buffer = match mask.iter()? {
            FilterIter::Indices(iter) => {
                filter_indices(self.boolean_buffer(), mask.true_count(), iter)
            }
            FilterIter::LazyIndices(iter) => {
                filter_indices(self.boolean_buffer(), mask.true_count(), iter)
            }
            FilterIter::Slices(iter) => {
                filter_slices(self.boolean_buffer(), mask.true_count(), iter)
            }
            FilterIter::LazySlices(iter) => {
                filter_slices(self.boolean_buffer(), mask.true_count(), iter)
            }
        };

        Ok(Self::try_new(buffer, validity)?.into_array())
    }
}

fn filter_indices(
    buffer: BooleanBuffer,
    indices_len: usize,
    mut indices: impl Iterator<Item = usize>,
) -> BooleanBuffer {
    let src = buffer.values().as_ptr();
    let offset = buffer.offset();

    BooleanBuffer::collect_bool(indices_len, |_idx| {
        let idx = indices.next().vortex_expect("iterator length must match");
        unsafe { bit_util::get_bit_raw(src, idx + offset) }
    })
}

fn filter_slices(
    buffer: BooleanBuffer,
    indices_len: usize,
    slices: impl Iterator<Item = (usize, usize)>,
) -> BooleanBuffer {
    let src = buffer.values();
    let offset = buffer.offset();

    let mut builder = BooleanBufferBuilder::new(bit_util::ceil(indices_len, 8));
    for (start, end) in slices {
        builder.append_packed_range(start + offset..end + offset, src)
    }
    builder.into()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::array::bool::compute::filter::{filter_indices, filter_slices};
    use crate::array::BoolArray;
    use crate::compute::{filter, FilterMask};
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn filter_bool_test() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = FilterMask::from_iter([true, false, true]);

        let filtered = filter(&arr.into_array(), mask)
            .unwrap()
            .into_bool()
            .unwrap();
        assert_eq!(2, filtered.len());

        assert_eq!(
            vec![true, false],
            filtered.boolean_buffer().iter().collect_vec()
        )
    }

    #[test]
    fn filter_bool_by_slice_test() {
        let arr = BoolArray::from_iter([true, true, false]);

        let filtered = filter_slices(arr.boolean_buffer(), 2, [(0, 1), (2, 3)].into_iter());
        assert_eq!(2, filtered.len());

        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }

    #[test]
    fn filter_bool_by_index_test() {
        let arr = BoolArray::from_iter([true, true, false]);

        let filtered = filter_indices(arr.boolean_buffer(), 2, [0, 2].into_iter());
        assert_eq!(2, filtered.len());

        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }
}
