// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::match_each_decimal_value_type;
use vortex_mask::MaskIter;
use vortex_mask::MaskValues;

use crate::arrays::DecimalArray;
use crate::arrays::filter::execute::filter_validity;
use crate::vtable::ValidityHelper;

/// Threshold for choosing between indices vs slices filtering strategy.
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

pub fn filter_decimal(array: &DecimalArray, mask: &Arc<MaskValues>) -> DecimalArray {
    let filtered_validity = filter_validity(array.validity().clone(), mask);

    match mask.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
        MaskIter::Indices(indices) => match_each_decimal_value_type!(array.values_type(), |T| {
            let filtered = filter_indices::<T>(array.buffer().as_slice(), indices.iter().copied());
            // SAFETY: Filter preserves the decimal dtype from the source array. The buffer is
            // correctly typed and sized based on the filtered indices.
            unsafe {
                DecimalArray::new_unchecked(filtered, array.decimal_dtype(), filtered_validity)
            }
        }),
        MaskIter::Slices(slices) => match_each_decimal_value_type!(array.values_type(), |T| {
            let filtered = filter_slices::<T>(
                array.buffer().as_slice(),
                mask.true_count(),
                slices.iter().copied(),
            );
            // SAFETY: Filter preserves the decimal dtype from the source array. The buffer is
            // correctly typed and sized based on the filtered slices.
            unsafe {
                DecimalArray::new_unchecked(filtered, array.decimal_dtype(), filtered_validity)
            }
        }),
    }
}

fn filter_indices<T: Copy>(values: &[T], indices: impl Iterator<Item = usize>) -> Buffer<T> {
    indices
        .map(|idx| {
            // SAFETY: The mask indices are guaranteed to be within bounds of the array.
            *unsafe { values.get_unchecked(idx) }
        })
        .collect()
}

fn filter_slices<T: Clone>(
    values: &[T],
    output_len: usize,
    slices: impl Iterator<Item = (usize, usize)>,
) -> Buffer<T> {
    let mut output = BufferMut::with_capacity(output_len);
    for (start, end) in slices {
        output.extend_from_slice(&values[start..end]);
    }
    output.freeze()
}

#[cfg(test)]
mod test {
    use vortex_dtype::DecimalDType;

    use crate::arrays::DecimalArray;
    use crate::compute::conformance::filter::test_filter_conformance;

    #[test]
    fn test_filter_decimal128_conformance() {
        let decimal_dtype = DecimalDType::new(38, 2);
        let values = vec![
            Some(12345i128),
            Some(67890),
            Some(-12345),
            Some(0),
            Some(99999),
        ];
        let array = DecimalArray::from_option_iter(values, decimal_dtype);
        test_filter_conformance(array.as_ref());
    }

    #[test]
    fn test_filter_decimal128_with_nulls_conformance() {
        let decimal_dtype = DecimalDType::new(38, 4);
        let values = vec![Some(12345i128), None, Some(-12345), Some(0), None];
        let array = DecimalArray::from_option_iter(values, decimal_dtype);
        test_filter_conformance(array.as_ref());
    }
}
