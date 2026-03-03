// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::get_bit;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskIter;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::FilterReduce;
use crate::vtable::ValidityHelper;

/// If the filter density is above 80%, we use slices to filter the array instead of indices.
const FILTER_SLICES_DENSITY_THRESHOLD: f64 = 0.8;

impl FilterReduce for BoolVTable {
    fn filter(array: &BoolArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity().filter(mask)?;

        let mask_values = mask
            .values()
            .vortex_expect("AllTrue and AllFalse are handled by filter fn");

        let buffer = match mask_values.threshold_iter(FILTER_SLICES_DENSITY_THRESHOLD) {
            MaskIter::Indices(indices) => filter_indices(
                &array.to_bit_buffer(),
                mask.true_count(),
                indices.iter().copied(),
            ),
            MaskIter::Slices(slices) => filter_slices(
                &array.to_bit_buffer(),
                mask.true_count(),
                slices.iter().copied(),
            ),
        };

        Ok(Some(BoolArray::new(buffer, validity).into_array()))
    }
}

/// Select indices from a boolean buffer.
/// NOTE: it was benchmarked to be faster using collect_bool to index into a slice than to
///  pass the indices as an iterator of usize. So we keep this alternate implementation.
pub fn filter_indices(
    bools: &BitBuffer,
    indices_len: usize,
    mut indices: impl Iterator<Item = usize>,
) -> BitBuffer {
    let buffer = bools.inner().as_ref();
    BitBuffer::collect_bool(indices_len, |_idx| {
        let idx = indices
            .next()
            .vortex_expect("iterator is guaranteed to be within the length of the array.");
        get_bit(buffer, bools.offset() + idx)
    })
}

pub fn filter_slices(
    buffer: &BitBuffer,
    indices_len: usize,
    slices: impl Iterator<Item = (usize, usize)>,
) -> BitBuffer {
    let mut builder = BitBufferMut::with_capacity(indices_len);
    for (start, end) in slices {
        // TODO(ngates): we probably want a borrowed slice for things like this.
        builder.append_buffer(&buffer.slice(start..end));
    }
    builder.freeze()
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_mask::Mask;

    use crate::arrays::BoolArray;
    use crate::arrays::bool::compute::filter::filter_indices;
    use crate::arrays::bool::compute::filter::filter_slices;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn filter_bool_test() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = Mask::from_iter([true, false, true]);

        let filtered = arr.filter(mask).unwrap();
        assert_arrays_eq!(filtered, BoolArray::from_iter([true, false]));
    }

    #[test]
    fn filter_bool_by_slice_test() {
        let arr = BoolArray::from_iter([true, true, false]);

        let filtered = filter_slices(&arr.to_bit_buffer(), 2, [(0, 1), (2, 3)].into_iter());
        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }

    #[test]
    fn filter_bool_by_index_test() {
        let arr = BoolArray::from_iter([true, true, false]);

        let filtered = filter_indices(&arr.to_bit_buffer(), 2, [0, 2].into_iter());
        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }

    use rstest::rstest;

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true]))]
    #[case(BoolArray::from_iter([false, false]))]
    #[case(BoolArray::from_iter((0..100).map(|i| i % 2 == 0)))]
    #[case(BoolArray::from_iter((0..1024).map(|i| i % 3 != 0)))]
    fn test_filter_bool_conformance(#[case] array: BoolArray) {
        test_filter_conformance(&array.to_array());
    }
}
