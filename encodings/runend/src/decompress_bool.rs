// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized run-end decoding for boolean arrays.
//!
//! Uses an adaptive strategy that pre-fills the buffer with the majority value
//! (0s or 1s) and only fills the minority runs, minimizing work for skewed distributions.

use itertools::Itertools;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_dtype::Nullability;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::iter::trimmed_ends_iter;

/// Decodes run-end encoded boolean values into a flat `BoolArray`.
pub fn runend_decode_bools(
    ends: PrimitiveArray,
    values: BoolArray,
    offset: usize,
    length: usize,
) -> VortexResult<BoolArray> {
    let validity = values.validity_mask()?;
    Ok(match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        runend_decode_typed_bool(
            trimmed_ends_iter(ends.as_slice::<E>(), offset, length),
            &values.to_bit_buffer(),
            validity,
            values.dtype().nullability(),
            length,
        )
    }))
}

/// Fills bits in range [start, end) to true using byte-level operations.
/// Assumes the buffer is pre-initialized to all zeros.
#[inline(always)]
fn fill_bits_true(slice: &mut [u8], start: usize, end: usize) {
    if start >= end {
        return;
    }

    let start_byte = start / 8;
    let start_bit = start % 8;
    let end_byte = end / 8;
    let end_bit = end % 8;

    if start_byte == end_byte {
        // All bits in same byte
        // Use u16 to avoid overflow, then truncate (guaranteed to fit in u8 since max is 0xFF)
        #[allow(clippy::cast_possible_truncation)]
        let mask = ((1u16 << (end_bit - start_bit)) - 1) as u8;
        slice[start_byte] |= mask << start_bit;
    } else {
        // First partial byte
        if start_bit != 0 {
            slice[start_byte] |= !((1u8 << start_bit) - 1);
        }

        // Middle bytes (bulk memset to 0xFF)
        let fill_start = if start_bit != 0 {
            start_byte + 1
        } else {
            start_byte
        };
        if fill_start < end_byte {
            slice[fill_start..end_byte].fill(0xFF);
        }

        // Last partial byte
        if end_bit != 0 {
            slice[end_byte] |= (1u8 << end_bit) - 1;
        }
    }
}

/// Clears bits in range [start, end) to false using byte-level operations.
/// Assumes the buffer is pre-initialized to all ones.
#[inline(always)]
fn fill_bits_false(slice: &mut [u8], start: usize, end: usize) {
    if start >= end {
        return;
    }

    let start_byte = start / 8;
    let start_bit = start % 8;
    let end_byte = end / 8;
    let end_bit = end % 8;

    if start_byte == end_byte {
        // All bits in same byte - create mask with 0s in the range we want to clear
        #[allow(clippy::cast_possible_truncation)]
        let mask = ((1u16 << (end_bit - start_bit)) - 1) as u8;
        slice[start_byte] &= !(mask << start_bit);
    } else {
        // First partial byte - clear high bits from start_bit
        if start_bit != 0 {
            slice[start_byte] &= (1u8 << start_bit) - 1;
        }

        // Middle bytes (bulk memset to 0x00)
        let fill_start = if start_bit != 0 {
            start_byte + 1
        } else {
            start_byte
        };
        if fill_start < end_byte {
            slice[fill_start..end_byte].fill(0x00);
        }

        // Last partial byte - clear low bits up to end_bit
        if end_bit != 0 {
            slice[end_byte] &= !((1u8 << end_bit) - 1);
        }
    }
}

/// Decodes run-end encoded boolean values using an adaptive strategy.
///
/// The strategy counts true vs false runs and chooses the optimal approach:
/// - If more true runs: pre-fill with 1s, clear false runs
/// - If more false runs: pre-fill with 0s, fill true runs
///
/// This minimizes work for skewed distributions (e.g., sparse validity masks).
pub fn runend_decode_typed_bool(
    run_ends: impl Iterator<Item = usize>,
    values: &BitBuffer,
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> BoolArray {
    match values_validity {
        Mask::AllTrue(_) => {
            // Adaptive strategy: choose based on which value is more common
            // If more runs have true values, pre-fill with 1s and clear false runs
            // If more runs have false values, pre-fill with 0s and fill true runs
            let true_count = values.true_count();
            let false_count = values.len() - true_count;

            if true_count > false_count {
                // More true runs - pre-fill with 1s and clear false runs
                let mut decoded = BitBufferMut::new_set(length);
                let decoded_bytes = decoded.as_mut_slice();
                let mut current_pos = 0usize;

                for (end, value) in run_ends.zip_eq(values.iter()) {
                    // Only clear when value is false (true is already 1)
                    if end > current_pos && !value {
                        fill_bits_false(decoded_bytes, current_pos, end);
                    }
                    current_pos = end;
                }
                BoolArray::new(decoded.freeze(), values_nullability.into())
            } else {
                // More or equal false runs - pre-fill with 0s and fill true runs
                let mut decoded = BitBufferMut::new_unset(length);
                let decoded_bytes = decoded.as_mut_slice();
                let mut current_pos = 0usize;

                for (end, value) in run_ends.zip_eq(values.iter()) {
                    // Only fill when value is true (false is already 0)
                    if end > current_pos && value {
                        fill_bits_true(decoded_bytes, current_pos, end);
                    }
                    current_pos = end;
                }
                BoolArray::new(decoded.freeze(), values_nullability.into())
            }
        }
        Mask::AllFalse(_) => BoolArray::new(BitBuffer::new_unset(length), Validity::AllInvalid),
        Mask::Values(mask) => {
            // For nullable values, adaptive strategy based on true count
            // (counting only valid values as true)
            let valid_true_count = values
                .iter()
                .zip(mask.bit_buffer().iter())
                .filter(|&(v, is_valid)| is_valid && v)
                .count();
            let valid_false_count = values
                .iter()
                .zip(mask.bit_buffer().iter())
                .filter(|&(v, is_valid)| is_valid && !v)
                .count();

            if valid_true_count > valid_false_count {
                // More true runs - pre-fill with 1s and clear false/null runs
                let mut decoded = BitBufferMut::new_set(length);
                let mut decoded_validity = BitBufferMut::new_unset(length);
                let decoded_bytes = decoded.as_mut_slice();
                let validity_bytes = decoded_validity.as_mut_slice();
                let mut current_pos = 0usize;

                for (end, value) in run_ends.zip_eq(
                    values
                        .iter()
                        .zip(mask.bit_buffer().iter())
                        .map(|(v, is_valid)| is_valid.then_some(v)),
                ) {
                    if end > current_pos {
                        match value {
                            None => {
                                // Null: clear decoded bits, validity stays false
                                fill_bits_false(decoded_bytes, current_pos, end);
                            }
                            Some(v) => {
                                // Valid: set validity bits to true
                                fill_bits_true(validity_bytes, current_pos, end);
                                // Clear decoded bits if value is false
                                if !v {
                                    fill_bits_false(decoded_bytes, current_pos, end);
                                }
                            }
                        }
                        current_pos = end;
                    }
                }
                BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze()))
            } else {
                // More or equal false runs - pre-fill with 0s and fill true runs
                let mut decoded = BitBufferMut::new_unset(length);
                let mut decoded_validity = BitBufferMut::new_unset(length);
                let decoded_bytes = decoded.as_mut_slice();
                let validity_bytes = decoded_validity.as_mut_slice();
                let mut current_pos = 0usize;

                for (end, value) in run_ends.zip_eq(
                    values
                        .iter()
                        .zip(mask.bit_buffer().iter())
                        .map(|(v, is_valid)| is_valid.then_some(v)),
                ) {
                    if end > current_pos {
                        match value {
                            None => {
                                // Validity stays false (already 0), decoded stays false
                            }
                            Some(v) => {
                                // Set validity bits to true
                                fill_bits_true(validity_bytes, current_pos, end);
                                // Set decoded bits if value is true
                                if v {
                                    fill_bits_true(decoded_bytes, current_pos, end);
                                }
                            }
                        }
                        current_pos = end;
                    }
                }
                BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::runend_decode_bools;

    #[test]
    fn decode_bools_alternating() -> VortexResult<()> {
        // Alternating true/false: [T, T, F, F, F, T, T, T, T, T]
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded = runend_decode_bools(ends, values, 0, 10)?;

        let expected = BoolArray::from(BitBuffer::from(vec![
            true, true, false, false, false, true, true, true, true, true,
        ]));
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn decode_bools_mostly_true() -> VortexResult<()> {
        // Mostly true: [T, T, T, T, T, F, T, T, T, T] - triggers true-heavy path
        let ends = PrimitiveArray::from_iter([5u32, 6, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded = runend_decode_bools(ends, values, 0, 10)?;

        let expected = BoolArray::from(BitBuffer::from(vec![
            true, true, true, true, true, false, true, true, true, true,
        ]));
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn decode_bools_mostly_false() -> VortexResult<()> {
        // Mostly false: [F, F, F, F, F, T, F, F, F, F] - triggers false-heavy path
        let ends = PrimitiveArray::from_iter([5u32, 6, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![false, true, false]));
        let decoded = runend_decode_bools(ends, values, 0, 10)?;

        let expected = BoolArray::from(BitBuffer::from(vec![
            false, false, false, false, false, true, false, false, false, false,
        ]));
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn decode_bools_all_true() -> VortexResult<()> {
        // All true: single run
        let ends = PrimitiveArray::from_iter([10u32]);
        let values = BoolArray::from(BitBuffer::from(vec![true]));
        let decoded = runend_decode_bools(ends, values, 0, 10)?;

        let expected = BoolArray::from(BitBuffer::from(vec![
            true, true, true, true, true, true, true, true, true, true,
        ]));
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn decode_bools_all_false() -> VortexResult<()> {
        // All false: single run
        let ends = PrimitiveArray::from_iter([10u32]);
        let values = BoolArray::from(BitBuffer::from(vec![false]));
        let decoded = runend_decode_bools(ends, values, 0, 10)?;

        let expected = BoolArray::from(BitBuffer::from(vec![
            false, false, false, false, false, false, false, false, false, false,
        ]));
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn decode_bools_with_offset() -> VortexResult<()> {
        // Test with offset: [T, T, F, F, F, T, T, T, T, T] -> slice [2..8] = [F, F, F, T, T, T]
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded = runend_decode_bools(ends, values, 2, 6)?;

        let expected =
            BoolArray::from(BitBuffer::from(vec![false, false, false, true, true, true]));
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }
}
