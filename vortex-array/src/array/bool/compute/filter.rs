use std::sync::Arc;

use arrow_buffer::{bit_util, BooleanBuffer, BooleanBufferBuilder};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{AllOr, Mask, MaskIter, MaskValues};

use crate::array::{BoolArray, BoolEncoding};
use crate::compute::FilterFn;
use crate::{ArrayDType, ArrayData, Canonical, IntoArrayData};

/// If the filter density is above 80%, we use slices to filter the array instead of indices.
const FILTER_SLICES_DENSITY_THRESHOLD: f64 = 0.8;

impl FilterFn<BoolArray> for BoolEncoding {
    fn filter(&self, array: &BoolArray, mask: &Mask) -> VortexResult<ArrayData> {
        let validity = array.validity().filter(mask)?;

        let mask_values = mask
            .values()
            .vortex_expect("AllTrue and AllFalse are handled by filter fn");

        let buffer = match mask_values.threshold_iter(FILTER_SLICES_DENSITY_THRESHOLD) {
            MaskIter::Indices(indices) => filter_indices(
                &array.boolean_buffer(),
                mask.true_count(),
                indices.iter().copied(),
            ),
            MaskIter::Slices(slices) => filter_slices(
                &array.boolean_buffer(),
                mask.true_count(),
                slices.iter().copied(),
            ),
        };

        Ok(BoolArray::try_new(buffer, validity)?.into_array())
    }
}

/// Select indices from a boolean buffer.
/// NOTE: it was benchmarked to be faster using collect_bool to index into a slice than to
///  pass the indices as an iterator of usize. So we keep this alternate implementation.
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
    use vortex_mask::Mask;

    use crate::array::bool::compute::filter::{filter_indices, filter_slices};
    use crate::array::BoolArray;
    use crate::compute::filter;
    use crate::{ArrayLen, IntoArrayData, IntoArrayVariant};

    #[test]
    fn filter_bool_test() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = Mask::from_iter([true, false, true]);

        let filtered = filter(&arr.into_array(), &mask)
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
