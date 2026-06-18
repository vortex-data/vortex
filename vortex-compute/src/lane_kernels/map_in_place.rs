// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-place lane kernels: read from an [`IndexedSink`] and write back through
//! the same sink (no separate output buffer).
//!
//! [`IndexedSink`]: crate::lane_kernels::sink::IndexedSink

use vortex_buffer::BitBuffer;

use crate::lane_kernels::CHUNK_LEN;
use crate::lane_kernels::sink::IndexedSink;

/// Extension trait providing in-place lane-kernel methods on any [`IndexedSink`].
///
/// All methods have default implementations and are inherited via the blanket
/// `impl<S: IndexedSink> IndexedSinkExt for S` below. Bring the trait into scope
/// (`use vortex_compute::lane_kernels::IndexedSinkExt;`) to call them with
/// method syntax.
///
/// [`IndexedSink`]: crate::lane_kernels::sink::IndexedSink
pub trait IndexedSinkExt: IndexedSink + Sized {
    /// In-place counterpart of [`IndexedSourceExt::map_into`]. Each lane
    /// is replaced with `f(self[i])`.
    ///
    /// The closure reads `Self::Item` and returns `Self::Write`. For the common
    /// case `Self = &mut [T]` both are `T`; for `ReinterpretSink` the read and
    /// write types can differ (e.g. read `f32`, write `u32`) over the same
    /// backing memory when sizes and alignments match.
    ///
    /// As with [`IndexedSourceExt::map_into`], use this only when the
    /// input is known non-nullable.
    ///
    /// [`IndexedSourceExt::map_into`]: crate::lane_kernels::map_into::IndexedSourceExt::map_into
    #[inline]
    fn map_into_in_place<F>(self, mut f: F)
    where
        F: FnMut(Self::Item) -> Self::Write,
    {
        #[inline(always)]
        fn chunk<S, F>(values: &mut S, f: &mut F, base: usize, count: usize)
        where
            S: IndexedSink,
            F: FnMut(S::Item) -> S::Write,
        {
            for bit_idx in 0..count {
                let idx = base + bit_idx;
                // SAFETY: caller guarantees base + count <= len.
                let val = unsafe { values.get_unchecked(idx) };
                let result = f(val);
                // SAFETY: caller guarantees base + count <= len.
                unsafe { values.set_unchecked(idx, result) };
            }
        }

        let mut values = self;
        let len = values.len();
        let chunks_count = len / CHUNK_LEN;
        let remainder = len % CHUNK_LEN;

        for chunk_idx in 0..chunks_count {
            chunk(&mut values, &mut f, chunk_idx * CHUNK_LEN, CHUNK_LEN);
        }
        if remainder != 0 {
            chunk(&mut values, &mut f, chunks_count * CHUNK_LEN, remainder);
        }
    }

    /// In-place counterpart of [`IndexedSourceExt::try_map_into`]. Each
    /// lane is replaced with `f(self[i])`, or `Self::Write::default()` when `f`
    /// returns `None`. On failure returns `Err(first_failing_lane)`; the buffer
    /// state on `Err` is unspecified.
    ///
    /// ## Error attribution
    ///
    /// Per-lane `is_none()` flags are bit-packed into a `u64` at the lane's
    /// position — `fail_bits |= (opt.is_none() as u64) << bit_idx`. After the
    /// 64-lane loop, `trailing_zeros()` of `fail_bits` recovers the first
    /// failing lane index. `OR + shift` per lane is friendlier to the
    /// autovectorizer than `min`/`csel` — see [`try_map_masked_in_place`] for
    /// the same scheme over a masked variant.
    ///
    /// [`try_map_masked_in_place`]: IndexedSinkExt::try_map_masked_in_place
    /// [`IndexedSourceExt::try_map_into`]: crate::lane_kernels::map_into::IndexedSourceExt::try_map_into
    #[inline]
    fn try_map_in_place<F>(self, mut f: F) -> Result<(), usize>
    where
        Self::Write: Default,
        F: FnMut(Self::Item) -> Option<Self::Write>,
    {
        #[inline(always)]
        fn chunk<S, F>(values: &mut S, base: usize, count: usize, f: &mut F) -> Option<usize>
        where
            S: IndexedSink,
            S::Write: Default,
            F: FnMut(S::Item) -> Option<S::Write>,
        {
            let mut fail_bits: u64 = 0;
            for bit_idx in 0..count {
                let idx = base + bit_idx;
                // SAFETY: caller guarantees base + count <= len.
                let val = unsafe { values.get_unchecked(idx) };
                let opt = f(val);
                fail_bits |= (opt.is_none() as u64) << bit_idx;
                let result = opt.unwrap_or_default();
                // SAFETY: caller guarantees base + count <= len.
                unsafe { values.set_unchecked(idx, result) };
            }
            (fail_bits != 0).then_some(base + fail_bits.trailing_zeros() as usize)
        }

        let mut values = self;
        let len = values.len();
        let chunks_count = len / 64;
        let remainder = len % 64;

        for chunk_idx in 0..chunks_count {
            if let Some(failing) = chunk(&mut values, chunk_idx * 64, 64, &mut f) {
                return Err(failing);
            }
        }
        if remainder != 0
            && let Some(failing) = chunk(&mut values, chunks_count * 64, remainder, &mut f)
        {
            return Err(failing);
        }
        Ok(())
    }

    /// In-place counterpart of [`IndexedSourceExt::try_map_masked_into`]. Each
    /// lane of `self` is replaced with `f(self[i])`, or `Self::Write::default()`
    /// if `f` returned `None`. On failure returns `Err(first_failing_lane)`;
    /// lanes before that point have been written, and lanes within the failing
    /// chunk hold their unwrapped-or-default result. The buffer state on `Err`
    /// is intentionally unspecified.
    ///
    /// **Null-lane failures are filtered automatically** — same semantics as
    /// [`IndexedSourceExt::try_map_masked_into`]. The closure has no `valid`
    /// parameter; the kernel AND-combines `is_none()` with the chunk's validity
    /// bitmap before folding it into the attribution accumulator.
    ///
    /// ## Why in-place is slower at cache-resident sizes
    ///
    /// At sizes that fit in L1/L2 the in-place kernel is ~1.5× slower than the
    /// out-of-place kernel despite having half the memory traffic, because
    /// input and output share memory and the compiler must be conservative
    /// reordering loads/stores across iterations. At sizes that exceed L2 the
    /// in-place kernel wins back the gap by avoiding the second buffer's DRAM
    /// read+write traffic.
    ///
    /// [`IndexedSourceExt::try_map_masked_into`]: crate::lane_kernels::map_into::IndexedSourceExt::try_map_masked_into
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != mask.len()`.
    #[inline]
    fn try_map_masked_in_place<F>(self, mask: &BitBuffer, mut f: F) -> Result<(), usize>
    where
        Self::Write: Default,
        F: FnMut(Self::Item) -> Option<Self::Write>,
    {
        /// Bit-pack `is_none()` flags per lane, then AND with `src_chunk` post-loop to
        /// drop null-lane failures. The per-lane attribution work is `OR + shift`
        /// (no `min`/`csel`), giving LLVM more freedom to vectorize the value pipeline.
        #[inline(always)]
        fn chunk<S, F>(
            values: &mut S,
            src_chunk: u64,
            base: usize,
            count: usize,
            f: &mut F,
        ) -> Option<usize>
        where
            S: IndexedSink,
            S::Write: Default,
            F: FnMut(S::Item) -> Option<S::Write>,
        {
            let mut fail_bits: u64 = 0;
            for bit_idx in 0..count {
                let idx = base + bit_idx;
                // SAFETY: caller guarantees `base + count <= values.len()`.
                let val = unsafe { values.get_unchecked(idx) };
                let opt = f(val);
                fail_bits |= (opt.is_none() as u64) << bit_idx;
                let result = opt.unwrap_or_default();
                unsafe { values.set_unchecked(idx, result) };
            }
            let valid_failures = fail_bits & src_chunk;
            (valid_failures != 0).then_some(base + valid_failures.trailing_zeros() as usize)
        }

        let mut values = self;
        let len = values.len();
        assert_eq!(len, mask.len(), "values and mask must have the same length");

        let chunks = mask.chunks();
        let chunks_count = len / 64;
        let remainder = len % 64;

        for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
            if let Some(failing) = chunk(&mut values, src_chunk, chunk_idx * 64, 64, &mut f) {
                return Err(failing);
            }
        }
        if remainder != 0
            && let Some(failing) = chunk(
                &mut values,
                chunks.remainder_bits(),
                chunks_count * 64,
                remainder,
                &mut f,
            )
        {
            return Err(failing);
        }
        Ok(())
    }
}

impl<S: IndexedSink> IndexedSinkExt for S {}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::BitBufferMut;

    use super::*;
    use crate::lane_kernels::sink::ReinterpretSink;

    #[test]
    fn try_map_masked_in_place_all_ok() {
        let mut values: Vec<u32> = (0..200).collect();
        let mask = BitBuffer::new_set(200);
        let res = values
            .as_mut_slice()
            .try_map_masked_in_place(&mask, |v| v.checked_mul(2));
        assert!(res.is_ok());
        let expected: Vec<u32> = (0..200u32).map(|v| v * 2).collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn try_map_masked_in_place_first_failing_chunk_wins() {
        let mut values: Vec<u32> = (0..200).collect();
        values[83] = u32::MAX;
        values[150] = u32::MAX;
        let mask = BitBuffer::new_set(200);
        let res = values
            .as_mut_slice()
            .try_map_masked_in_place(&mask, |v| v.checked_mul(2));
        assert_eq!(res, Err(83));
    }

    #[test]
    fn try_map_masked_in_place_within_chunk_reports_lowest() {
        let mut values: Vec<u32> = (0..200).collect();
        values[80] = u32::MAX;
        values[100] = u32::MAX;
        let mask = BitBuffer::new_set(200);
        let res = values
            .as_mut_slice()
            .try_map_masked_in_place(&mask, |v| v.checked_mul(2));
        assert_eq!(res, Err(80));
    }

    #[test]
    fn try_map_masked_in_place_single_failure_lane_exact() {
        let mut values: Vec<u32> = (0..200).collect();
        values[42] = u32::MAX;
        let mask = BitBuffer::new_set(200);
        let res = values
            .as_mut_slice()
            .try_map_masked_in_place(&mask, |v| v.checked_mul(2));
        assert_eq!(res, Err(42));
    }

    #[test]
    fn try_map_masked_in_place_null_bypass() {
        let mut values: Vec<u32> = (0..200).collect();
        values[5] = u32::MAX;
        let mask = {
            let mut m = BitBufferMut::with_capacity(200);
            for i in 0..200 {
                m.append(i != 5);
            }
            m.freeze()
        };
        let res = values
            .as_mut_slice()
            .try_map_masked_in_place(&mask, |v| v.checked_mul(2));
        assert!(res.is_ok(), "null-lane overflow should be filtered");
        assert_eq!(values[5], 0);
        assert_eq!(values[6], 12);
    }

    #[test]
    fn try_map_masked_in_place_remainder_overflow() {
        let mut values: Vec<u32> = (0..130).collect();
        values[129] = u32::MAX;
        let mask = BitBuffer::new_set(130);
        let res = values
            .as_mut_slice()
            .try_map_masked_in_place(&mask, |v| v.checked_mul(2));
        assert_eq!(res, Err(129));
    }

    #[test]
    fn try_map_masked_in_place_sliced_mask() {
        let big = BitBuffer::new_set(256);
        let mask = big.slice(13..143);
        assert_eq!(mask.len(), 130);

        let mut values: Vec<u32> = (0..130).collect();
        values[77] = u32::MAX;
        let res = values
            .as_mut_slice()
            .try_map_masked_in_place(&mask, |v| v.checked_mul(2));
        assert_eq!(res, Err(77));
    }

    #[test]
    fn reinterpret_sink_same_width_f32_u32() {
        let mut buf: Vec<f32> = (0..130).map(|i| i as f32).collect();
        let mask = BitBuffer::new_set(130);
        ReinterpretSink::<f32, u32>::new(buf.as_mut_slice())
            .try_map_masked_in_place(&mask, |f| Some(f.to_bits().wrapping_add(1)))
            .unwrap();
        // SAFETY: same size + alignment for f32 and u32; every slot now holds a u32 written by
        // the closure.
        let as_u32: &[u32] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u32, buf.len()) };
        for (i, &got) in as_u32.iter().enumerate() {
            assert_eq!(got, (i as f32).to_bits().wrapping_add(1), "lane {i}");
        }
    }

    #[test]
    fn reinterpret_sink_failure_reports_lane() {
        let mut buf: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let mask = BitBuffer::new_set(200);
        let res = ReinterpretSink::<f32, u32>::new(buf.as_mut_slice()).try_map_masked_in_place(
            &mask,
            |f| {
                if f as u32 == 137 {
                    None
                } else {
                    Some(f as u32)
                }
            },
        );
        assert_eq!(res, Err(137));
    }

    #[test]
    fn try_map_masked_in_place_partial_chunk_success() {
        let mut values: Vec<u32> = (0..130).collect();
        let mask = BitBuffer::new_set(130);
        let res = values
            .as_mut_slice()
            .try_map_masked_in_place(&mask, |v| Some(v + 1));
        assert!(res.is_ok());
        assert_eq!(values[0], 1);
        assert_eq!(values[63], 64);
        assert_eq!(values[64], 65);
        assert_eq!(values[129], 130);
    }
}
