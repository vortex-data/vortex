// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_mask::MaskIter;
use vortex_mask::MaskValues;

use crate::arrays::BoolArray;
use crate::arrays::filter::execute::filter_validity;

/// If the filter density is above 80%, we use slices to filter the array instead of indices.
const FILTER_SLICES_DENSITY_THRESHOLD: f64 = 0.8;

// TODO(connor): Use the optimized filters over bit buffers in `vortex-compute`.
pub fn filter_bool(array: &BoolArray, mask: &Arc<MaskValues>) -> BoolArray {
    let validity = array.validity().vortex_expect("missing BoolArray validity");
    let filtered_validity = filter_validity(validity, mask);

    let bit_buffer = array.to_bit_buffer();

    let filtered_buffer = match mask.threshold_iter(FILTER_SLICES_DENSITY_THRESHOLD) {
        MaskIter::Indices(indices) => {
            filter_indices(&bit_buffer, mask.true_count(), indices.iter().copied())
        }
        MaskIter::Slices(slices) => {
            filter_slices(&bit_buffer, mask.true_count(), slices.iter().copied())
        }
    };

    BoolArray::new(filtered_buffer, filtered_validity)
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
        vortex_buffer::get_bit(buffer, bools.offset() + idx)
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

    use super::*;
    use crate::arrays::BoolArray;
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn filter_bool_test() {
        let arr = BoolArray::from_iter([true, true, false]);
        let mask = Mask::from_iter([true, false, true]);

        let filtered = arr.filter(mask).unwrap().to_bool();
        assert_eq!(2, filtered.len());

        assert_eq!(
            vec![true, false],
            filtered.into_bit_buffer().iter().collect_vec()
        )
    }

    #[test]
    fn filter_bool_by_slice_test() {
        let arr = BoolArray::from_iter([true, true, false]);

        let filtered = filter_slices(&arr.into_bit_buffer(), 2, [(0, 1), (2, 3)].into_iter());
        assert_eq!(2, filtered.len());

        assert_eq!(vec![true, false], filtered.iter().collect_vec())
    }

    #[test]
    fn filter_bool_by_index_test() {
        let arr = BoolArray::from_iter([true, true, false]);

        let filtered = filter_indices(&arr.into_bit_buffer(), 2, [0, 2].into_iter());
        assert_eq!(2, filtered.len());

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
        test_filter_conformance(array.as_ref());
    }
}
