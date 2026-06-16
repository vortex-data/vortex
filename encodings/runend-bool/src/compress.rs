// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-end encoding and decoding routines specialized for boolean arrays.
//!
//! Boolean runs strictly alternate, so a run-end encoded bool array stores only the run `ends`
//! plus the value of the first run (`start`). The value of run `i` is then
//! [`value_at_index`]`(i, start)`.

use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::array::RunEndBool;
use crate::array::RunEndBoolArray;

/// Returns the boolean value of the run with index `idx` given the value of run 0 (`start`).
///
/// Runs strictly alternate, so even-indexed runs equal `start` and odd-indexed runs equal `!start`.
pub fn value_at_index(idx: usize, start: bool) -> bool {
    if idx.is_multiple_of(2) { start } else { !start }
}

/// Run-end encode a [`BitBuffer`], returning the run `ends` and the value of the first run.
///
/// `start` is the value of run 0. The returned `ends` are the exclusive end positions of each run.
pub fn runend_bool_encode_slice(elements: &BitBuffer) -> (Vec<u64>, bool) {
    let mut iter = elements.set_slices();
    let Some((start, end)) = iter.next() else {
        return (vec![elements.len() as u64], false);
    };
    let mut ends = Vec::new();
    let first_bool = start == 0;
    if !first_bool {
        ends.push(start as u64)
    }
    ends.push(end as u64);
    for (s, e) in iter {
        ends.push(s as u64);
        ends.push(e as u64);
    }
    let last_end = *ends.last().vortex_expect("ends is non-empty");
    if last_end != elements.len() as u64 {
        ends.push(elements.len() as u64)
    }
    (ends, first_bool)
}

/// Decode run-end encoded boolean values into a flat [`BitBuffer`].
pub fn runend_bool_decode_slice(
    run_ends_iter: impl Iterator<Item = usize>,
    start: bool,
    length: usize,
) -> BitBuffer {
    let mut decoded = BitBufferMut::with_capacity(length);
    for (idx, end) in run_ends_iter.enumerate() {
        decoded.append_n(value_at_index(idx, start), end - decoded.len());
    }
    decoded.freeze()
}

/// Run-end encode a [`BoolArray`] into a [`RunEndBoolArray`].
///
/// The run `ends` are narrowed to the smallest unsigned integer type that can hold them.
pub fn encode_runend_bool(
    array: &BoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RunEndBoolArray> {
    let length = array.as_ref().len();
    let validity = array.as_ref().validity()?;
    let bits = array.to_bit_buffer();
    let (ends, start) = runend_bool_encode_slice(&bits);

    let ends = PrimitiveArray::new(Buffer::from(ends), Validity::NonNullable)
        .narrow(ctx)
        .vortex_expect("ends must succeed downcasting");

    // SAFETY: runend_bool_encode_slice produces strictly-increasing ends with last == length.
    Ok(unsafe { RunEndBool::new_unchecked(ends.into_array(), start, 0, length, validity) })
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_runend::trimmed_ends_iter;

    use super::runend_bool_decode_slice;
    use super::runend_bool_encode_slice;
    use super::value_at_index;

    #[test]
    fn encode_decode_roundtrip() -> VortexResult<()> {
        let bits = BitBuffer::from(vec![
            true, true, false, false, false, true, true, true, true, true,
        ]);
        let (ends, start) = runend_bool_encode_slice(&bits);
        assert_eq!(ends, vec![2, 5, 10]);
        assert!(start);

        let decoded = runend_bool_decode_slice(trimmed_ends_iter(&ends, 0, 10), start, 10);
        assert_eq!(decoded, bits);
        Ok(())
    }

    #[test]
    fn encode_all_false() {
        let bits = BitBuffer::from(vec![false, false, false]);
        let (ends, start) = runend_bool_encode_slice(&bits);
        assert_eq!(ends, vec![3]);
        assert!(!start);
    }

    #[test]
    fn encode_leading_false() {
        let bits = BitBuffer::from(vec![false, true, true, false]);
        let (ends, start) = runend_bool_encode_slice(&bits);
        assert_eq!(ends, vec![1, 3, 4]);
        assert!(!start);
    }

    #[test]
    fn value_at_index_alternates() {
        assert!(value_at_index(0, true));
        assert!(!value_at_index(1, true));
        assert!(value_at_index(2, true));
        assert!(!value_at_index(0, false));
    }

    #[test]
    fn decode_with_offset() -> VortexResult<()> {
        // [T,T,F,F,F,T,T,T,T,T] sliced 2..8 => [F,F,F,T,T,T]
        let ends: Vec<u32> = vec![2, 5, 10];
        let decoded = runend_bool_decode_slice(trimmed_ends_iter(&ends, 2, 6), true, 6);
        assert_eq!(
            decoded,
            BitBuffer::from(vec![false, false, false, true, true, true])
        );
        Ok(())
    }

    #[test]
    fn encode_array_roundtrip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let array = BoolArray::from(BitBuffer::from(vec![
            true, true, false, false, false, true, true, true, true, true,
        ]));
        let encoded = super::encode_runend_bool(&array, &mut ctx)?;
        assert_eq!(encoded.as_ref().len(), 10);
        Ok(())
    }
}
