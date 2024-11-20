use arrow_buffer::{bit_util, BooleanBuffer, BooleanBufferBuilder};
use vortex_error::{VortexExpect, VortexResult};

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::{FilterFn, FilterIter, FilterMask};
use crate::{ArrayData, IntoArrayData};

impl FilterFn<BoolArray> for BoolEncoding {
    fn filter(&self, array: &BoolArray, mask: FilterMask) -> VortexResult<ArrayData> {
        let validity = array.validity().filter(&mask)?;

        let buffer = match mask.iter()? {
            FilterIter::Indices(indices) => filter_indices_slice(&array.boolean_buffer(), indices),
            FilterIter::IndicesIter(iter) => {
                filter_indices(&array.boolean_buffer(), mask.true_count(), iter)
            }
            FilterIter::Slices(slices) => filter_slices(
                &array.boolean_buffer(),
                mask.true_count(),
                slices.iter().copied(),
            ),
            FilterIter::SlicesIter(iter) => {
                filter_slices(&array.boolean_buffer(), mask.true_count(), iter)
            }
        };

        Ok(BoolArray::try_new(buffer, validity)?.into_array())
    }
}

/// Select indices from a boolean buffer.
/// NOTE: it was benchmarked to be faster using collect_bool to index into a slice than to
///  pass the indices as an iterator of usize. So we keep this alternate implementation.
fn filter_indices_slice(buffer: &BooleanBuffer, indices: &[usize]) -> BooleanBuffer {
    let src = buffer.values().as_ptr();
    let offset = buffer.offset();
    BooleanBuffer::collect_bool(indices.len(), |idx| unsafe {
        bit_util::get_bit_raw(src, *indices.get_unchecked(idx) + offset)
    })
}

pub fn filter_indices(
    buffer: &BooleanBuffer,
    indices_len: usize,
    mut indices: impl Iterator<Item = usize>,
) -> BooleanBuffer {
    let src = buffer.values().as_ptr();
    let offset = buffer.offset();

    BooleanBuffer::collect_bool(indices_len, |_idx| {
        let idx = indices
            .next()
            .vortex_expect("iterator is guaranteed to be within the length of the array.");
        unsafe { bit_util::get_bit_raw(src, idx + offset) }
    })
}

pub fn filter_slices(
    buffer: &BooleanBuffer,
    indices_len: usize,
    slices: impl Iterator<Item = (usize, usize)>,
) -> BooleanBuffer {
    let src = buffer.values();
    let offset = buffer.offset();

    let mut builder = BooleanBufferBuilder::new(indices_len);
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
    use crate::{ArrayLen, IntoArrayData, IntoArrayVariant};

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

        let filtered = filter_slices(&arr.boolean_buffer(), 2, [(0, 1), (2, 3)].into_iter());
        assert_eq!(2, filtered.len());

        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }

    #[test]
    fn filter_bool_by_index_test() {
        let arr = BoolArray::from_iter([true, true, false]);

        let filtered = filter_indices(&arr.boolean_buffer(), 2, [0, 2].into_iter());
        assert_eq!(2, filtered.len());

        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }
}
