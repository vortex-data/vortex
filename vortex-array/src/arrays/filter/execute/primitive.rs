// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexExpect;
use vortex_mask::MaskIter;
use vortex_mask::MaskValues;

use crate::arrays::PrimitiveArray;
use crate::arrays::filter::execute::filter_validity;

/// Threshold for choosing between indices vs slices filtering strategy.
pub const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

// TODO(connor): Use the optimized filters over slices in `vortex-compute`.
pub fn filter_primitive(array: &PrimitiveArray, mask: &Arc<MaskValues>) -> PrimitiveArray {
    let validity = array
        .validity()
        .vortex_expect("missing PrimitiveArray validity");
    let filtered_validity = filter_validity(validity, mask);

    match_each_native_ptype!(array.ptype(), |T| {
        let filtered_buffer = filter_slice(
            array.as_slice::<T>(),
            mask,
            FILTER_SLICES_SELECTIVITY_THRESHOLD,
        );
        PrimitiveArray::new(filtered_buffer, filtered_validity)
    })
}

/// Filter a typed buffer by a mask, returning a new buffer with only the selected elements.
///
/// This is the core filtering operation used by both FilterArray execution and the
/// FilterKernel for primitive arrays.
///
/// # Arguments
/// * `values` - The source slice of values to filter
/// * `mask` - The mask indicating which elements to keep
/// * `selectivity_threshold` - Threshold for choosing between indices vs slices strategy
pub fn filter_slice<T: Copy>(
    values: &[T],
    mask: &MaskValues,
    selectivity_threshold: f64,
) -> Buffer<T> {
    match mask.threshold_iter(selectivity_threshold) {
        MaskIter::Indices(indices) => indices
            .iter()
            .copied()
            .map(|idx| *unsafe { values.get_unchecked(idx) })
            .collect(),
        MaskIter::Slices(slices) => {
            let mut output = BufferMut::with_capacity(mask.true_count());
            for (start, end) in slices.iter().copied() {
                output.extend_from_slice(&values[start..end]);
            }
            output.freeze()
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod test {
    use itertools::Itertools;
    use rstest::rstest;
    use vortex_mask::Mask;

    use crate::arrays::primitive::PrimitiveArray;
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::filter::LARGE_SIZE;
    use crate::compute::conformance::filter::MEDIUM_SIZE;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn filter_run_variant_mixed_test() {
        let mask = [true, true, false, true, true, true, false, true];
        let arr = PrimitiveArray::from_iter([1u32, 24, 54, 2, 3, 2, 3, 2]);

        let filtered = arr.filter(Mask::from_iter(mask)).unwrap().to_primitive();
        assert_eq!(
            filtered.len(),
            mask.iter().filter(|x| **x).collect_vec().len()
        );

        let rust_arr = arr.as_slice::<u32>();
        assert_eq!(
            filtered.as_slice::<u32>().to_vec(),
            mask.iter()
                .enumerate()
                .filter(|(_idx, b)| **b)
                .map(|m| rust_arr[m.0])
                .collect_vec()
        )
    }

    #[rstest]
    #[case(PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]))]
    #[case(PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]))]
    #[case(PrimitiveArray::from_iter([42u64]))]
    #[case(PrimitiveArray::from_iter(0..MEDIUM_SIZE as i32))]
    #[case(PrimitiveArray::from_option_iter(
        (0..MEDIUM_SIZE).map(|i| if i % 3 == 0 { None } else { Some(i as i64) }))
    )]
    #[case(PrimitiveArray::from_iter(0..LARGE_SIZE as u32))]
    #[case(PrimitiveArray::from_iter([0.1f32, 0.2, 0.3, 0.4, 0.5]))]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f64), None, Some(2.2), Some(3.3), None]))]
    fn test_filter_primitive_conformance(#[case] array: PrimitiveArray) {
        test_filter_conformance(array.as_ref());
    }
}
