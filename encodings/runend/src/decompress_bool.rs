// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized run-end decoding for boolean arrays.
//!
//! Uses an adaptive strategy that pre-fills the buffer with the majority value
//! (0s or 1s) and only fills the minority runs, minimizing work for skewed distributions.

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::iter::trimmed_ends_iter;

/// Threshold for number of runs below which we use sequential append instead of prefill.
/// With few runs, the overhead of prefilling the entire buffer dominates.
const PREFILL_RUN_THRESHOLD: usize = 32;

/// Decodes run-end encoded boolean values into a flat `BoolArray`.
pub fn runend_decode_bools(
    ends: PrimitiveArray,
    values: BoolArray,
    offset: usize,
    length: usize,
) -> VortexResult<ArrayRef> {
    let validity = values.as_ref().validity()?.to_mask(
        values.as_ref().len(),
        &mut LEGACY_SESSION.create_execution_ctx(),
    )?;
    let values_buf = values.to_bit_buffer();
    let nullability = values.dtype().nullability();

    // Fast path for few runs with no offset - avoids iterator overhead
    let num_runs = values_buf.len();
    if offset == 0 && num_runs < PREFILL_RUN_THRESHOLD {
        return Ok(match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
            decode_few_runs_no_offset(
                ends.as_slice::<E>(),
                &values_buf,
                validity,
                nullability,
                length,
            )
        }));
    }

    Ok(match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        runend_decode_typed_bool(
            trimmed_ends_iter(ends.as_slice::<E>(), offset, length),
            &values_buf,
            validity,
            nullability,
            length,
        )
    }))
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
) -> ArrayRef {
    match values_validity {
        Mask::AllTrue(_) => {
            decode_bool_non_nullable(run_ends, values, values_nullability, length).into_array()
        }
        Mask::AllFalse(_) => {
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), length)
                .into_array()
        }
        Mask::Values(mask) => {
            decode_bool_nullable(run_ends, values, mask.bit_buffer(), length).into_array()
        }
    }
}

/// Fast path for few runs with no offset. Uses direct slice access to minimize overhead.
/// This avoids the `trimmed_ends_iter` iterator chain which adds significant overhead
/// for small numbers of runs.
#[inline(always)]
fn decode_few_runs_no_offset<E: vortex_array::dtype::IntegerPType>(
    ends: &[E],
    values: &BitBuffer,
    validity: Mask,
    nullability: Nullability,
    length: usize,
) -> ArrayRef {
    match validity {
        Mask::AllTrue(_) => {
            let mut decoded = BitBufferMut::with_capacity(length);
            let mut prev_end = 0usize;
            for (i, &end) in ends.iter().enumerate() {
                let end = end.as_().min(length);
                decoded.append_n(values.value(i), end - prev_end);
                prev_end = end;
            }
            BoolArray::new(decoded.freeze(), nullability.into()).into_array()
        }
        Mask::AllFalse(_) => {
            ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), length)
                .into_array()
        }
        Mask::Values(mask) => {
            let validity_buf = mask.bit_buffer();
            let mut decoded = BitBufferMut::with_capacity(length);
            let mut decoded_validity = BitBufferMut::with_capacity(length);
            let mut prev_end = 0usize;
            for (i, &end) in ends.iter().enumerate() {
                let end = end.as_().min(length);
                let run_len = end - prev_end;
                let is_valid = validity_buf.value(i);
                if is_valid {
                    decoded_validity.append_n(true, run_len);
                    decoded.append_n(values.value(i), run_len);
                } else {
                    decoded_validity.append_n(false, run_len);
                    decoded.append_n(false, run_len);
                }
                prev_end = end;
            }
            BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze())).into_array()
        }
    }
}

/// Decodes run-end encoded booleans when all values are valid (non-nullable).
fn decode_bool_non_nullable(
    run_ends: impl Iterator<Item = usize>,
    values: &BitBuffer,
    nullability: Nullability,
    length: usize,
) -> BoolArray {
    let num_runs = values.len();

    // For few runs, sequential append is faster than prefill + modify
    if num_runs < PREFILL_RUN_THRESHOLD {
        let mut decoded = BitBufferMut::with_capacity(length);
        for (end, value) in run_ends.zip(values.iter()) {
            decoded.append_n(value, end - decoded.len());
        }
        return BoolArray::new(decoded.freeze(), nullability.into());
    }

    // Adaptive strategy: prefill with majority value, only flip minority runs
    let prefill = values.true_count() > num_runs - values.true_count();
    let mut decoded = BitBufferMut::full(prefill, length);
    let mut current_pos = 0usize;

    for (end, value) in run_ends.zip_eq(values.iter()) {
        if end > current_pos && value != prefill {
            // SAFETY: current_pos < end <= length == decoded.len()
            unsafe { decoded.fill_range_unchecked(current_pos, end, value) };
        }
        current_pos = end;
    }
    BoolArray::new(decoded.freeze(), nullability.into())
}

/// Decodes run-end encoded booleans when values may be null (nullable).
fn decode_bool_nullable(
    run_ends: impl Iterator<Item = usize>,
    values: &BitBuffer,
    validity_mask: &BitBuffer,
    length: usize,
) -> BoolArray {
    let num_runs = values.len();

    // For few runs, sequential append is faster than prefill + modify
    if num_runs < PREFILL_RUN_THRESHOLD {
        return decode_nullable_sequential(run_ends, values, validity_mask, length);
    }

    // Adaptive strategy: prefill each buffer with its majority value
    let prefill_decoded = values.true_count() > num_runs - values.true_count();
    let prefill_valid = validity_mask.true_count() > num_runs - validity_mask.true_count();

    let mut decoded = BitBufferMut::full(prefill_decoded, length);
    let mut decoded_validity = BitBufferMut::full(prefill_valid, length);
    let mut current_pos = 0usize;

    for (end, (value, is_valid)) in run_ends.zip_eq(values.iter().zip(validity_mask.iter())) {
        if end > current_pos {
            // SAFETY: current_pos < end <= length == decoded.len() == decoded_validity.len()
            if is_valid != prefill_valid {
                unsafe { decoded_validity.fill_range_unchecked(current_pos, end, is_valid) };
            }
            // Decoded bit should be the actual value when valid, false when null.
            let want_decoded = is_valid && value;
            if want_decoded != prefill_decoded {
                unsafe { decoded.fill_range_unchecked(current_pos, end, want_decoded) };
            }
            current_pos = end;
        }
    }
    BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze()))
}

/// Sequential decode for few runs - avoids prefill overhead.
#[inline(always)]
fn decode_nullable_sequential(
    run_ends: impl Iterator<Item = usize>,
    values: &BitBuffer,
    validity_mask: &BitBuffer,
    length: usize,
) -> BoolArray {
    let mut decoded = BitBufferMut::with_capacity(length);
    let mut decoded_validity = BitBufferMut::with_capacity(length);

    for (end, (value, is_valid)) in run_ends.zip(values.iter().zip(validity_mask.iter())) {
        let run_len = end - decoded.len();
        if is_valid {
            decoded_validity.append_n(true, run_len);
            decoded.append_n(value, run_len);
        } else {
            decoded_validity.append_n(false, run_len);
            decoded.append_n(false, run_len);
        }
    }

    BoolArray::new(decoded.freeze(), Validity::from(decoded_validity.freeze()))
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::bool::BoolArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
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
        // Mostly true: [T, T, T, T, T, F, T, T, T, T]
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
        // Mostly false: [F, F, F, F, F, T, F, F, F, F]
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
    fn decode_bools_all_true_single_run() -> VortexResult<()> {
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
    fn decode_bools_all_false_single_run() -> VortexResult<()> {
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

    #[test]
    fn decode_bools_nullable() -> VortexResult<()> {
        use vortex_array::validity::Validity;

        // 3 runs: T (valid), F (null), T (valid) -> [T, T, null, null, null, T, T, T, T, T]
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = BoolArray::new(
            BitBuffer::from(vec![true, false, true]),
            Validity::from(BitBuffer::from(vec![true, false, true])),
        );
        let decoded = runend_decode_bools(ends, values, 0, 10)?;

        // Expected: values=[T, T, F, F, F, T, T, T, T, T], validity=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
        let expected = BoolArray::new(
            BitBuffer::from(vec![
                true, true, false, false, false, true, true, true, true, true,
            ]),
            Validity::from(BitBuffer::from(vec![
                true, true, false, false, false, true, true, true, true, true,
            ])),
        );
        assert_arrays_eq!(decoded, expected);
        Ok(())
    }

    #[test]
    fn decode_bools_nullable_few_runs() -> VortexResult<()> {
        // Test few runs (uses fast path): 5 runs of length 2000 each
        let ends = PrimitiveArray::from_iter([2000u32, 4000, 6000, 8000, 10000]);
        let values = BoolArray::new(
            BitBuffer::from(vec![true, false, true, false, true]),
            Validity::from(BitBuffer::from(vec![true, false, true, false, true])),
        );
        let decoded = runend_decode_bools(ends, values, 0, 10000)?.to_bool();

        // Check length and a few values
        assert_eq!(decoded.len(), 10000);
        // First run: valid true
        assert!(
            decoded
                .as_ref()
                .validity()?
                .to_mask(
                    decoded.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx()
                )
                .unwrap()
                .value(0)
        );
        assert!(decoded.to_bit_buffer().value(0));
        // Second run: null (validity false)
        assert!(
            !decoded
                .as_ref()
                .validity()?
                .to_mask(
                    decoded.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx()
                )
                .unwrap()
                .value(2000)
        );
        // Third run: valid true
        assert!(
            decoded
                .as_ref()
                .validity()?
                .to_mask(
                    decoded.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx()
                )
                .unwrap()
                .value(4000)
        );
        assert!(decoded.to_bit_buffer().value(4000));
        Ok(())
    }
}
