// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vgetq_lane_u64;
use std::arch::aarch64::vld1q_s8;
use std::arch::aarch64::vld1q_u8_x4;
use std::arch::aarch64::vmaxq_u8;
use std::arch::aarch64::vmaxvq_u8;
use std::arch::aarch64::vpaddq_u8;
use std::arch::aarch64::vqtbl4q_u8;
use std::arch::aarch64::vreinterpretq_u64_u8;
use std::arch::aarch64::vshlq_u8;

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferView;
use vortex_buffer::BufferMut;

/// Looks up a dictionary of at most 64 booleans in four NEON registers and packs the results.
pub(super) fn take(bools: BitBufferView<'_>, indices: &[u8]) -> BitBuffer {
    assert!(bools.len() <= 64);
    if indices.is_empty() {
        return BitBuffer::new_unset(0);
    }
    let mut values = [0u8; 64];
    for (dst, value) in values.iter_mut().zip(bools.iter()) {
        *dst = u8::from(value);
    }
    let mut words = BufferMut::<u64>::with_capacity(indices.len().div_ceil(64));
    const SHIFTS: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];

    // SAFETY: NEON is available on aarch64. Table and shift loads stay inside the fixed arrays;
    // each index load reads an exact 64-byte chunk. Table lookups outside 0..64 yield zero,
    // so invalid indices cannot read memory; the maximum-index check rejects them below.
    unsafe {
        let table = vld1q_u8_x4(values.as_ptr());
        let shifts = vld1q_s8(SHIFTS.as_ptr());
        let mut max_index = vdupq_n_u8(0);
        let mut chunks = indices.chunks_exact(64);
        for chunk in &mut chunks {
            let ids = vld1q_u8_x4(chunk.as_ptr());
            max_index = vmaxq_u8(
                max_index,
                vmaxq_u8(vmaxq_u8(ids.0, ids.1), vmaxq_u8(ids.2, ids.3)),
            );
            let a = vshlq_u8(vqtbl4q_u8(table, ids.0), shifts);
            let b = vshlq_u8(vqtbl4q_u8(table, ids.1), shifts);
            let c = vshlq_u8(vqtbl4q_u8(table, ids.2), shifts);
            let d = vshlq_u8(vqtbl4q_u8(table, ids.3), shifts);
            // Weight each boolean by its bit position, then sum each group of eight into a byte.
            let sum = vpaddq_u8(vpaddq_u8(a, b), vpaddq_u8(c, d));
            let packed = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vpaddq_u8(sum, sum)));
            words.push(packed.to_le());
        }
        assert!(usize::from(vmaxvq_u8(max_index)) < bools.len());
        let tail = chunks.remainder();
        if !tail.is_empty() {
            words.push(
                tail.iter()
                    .enumerate()
                    .fold(0u64, |word, (bit, &index)| {
                        word | (u64::from(bools.value(usize::from(index))) << bit)
                    })
                    .to_le(),
            );
        }
    }
    BitBuffer::new(words.freeze().into_byte_buffer(), indices.len())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::take;

    #[rstest]
    #[case(1)]
    #[case(4)]
    #[case(7)]
    #[case(25)]
    #[case(50)]
    #[case(64)]
    fn matches_scalar_lookup(#[case] values_len: usize) -> VortexResult<()> {
        for offset in [0, 1, 7, 8, 63] {
            let values = BitBuffer::from_iter((0..offset + values_len).map(|i| i % 3 == 0));
            let values = values.as_view().slice(offset..offset + values_len);
            for len in [0, 1, 63, 64, 65, 127, 129, 65_536] {
                let mut state = 91u64;
                let indices: Vec<u8> = (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        VortexResult::Ok(u8::try_from(state % values_len as u64)?)
                    })
                    .collect::<VortexResult<_>>()?;
                let expected = BitBuffer::from_iter(
                    indices.iter().map(|&index| values.value(usize::from(index))),
                );
                assert_eq!(take(values, &indices), expected);
            }
        }
        Ok(())
    }

    #[rstest]
    #[case(0)]
    #[case(63)]
    #[case(64)]
    #[should_panic]
    fn rejects_out_of_bounds_indices(
        #[case] position: usize,
        #[values(7, 64, 255)] invalid: u8,
    ) {
        let values = BitBuffer::new_set(7);
        let mut indices = [0u8; 65];
        indices[position] = invalid;
        take(values.as_view(), &indices);
    }
}
