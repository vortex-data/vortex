// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Out-of-place lane kernels: read from an [`IndexedSource`] and write into a
//! caller-provided `&mut [MaybeUninit<R>]`.

use std::mem::MaybeUninit;

use vortex_buffer::BitBuffer;

use crate::lane_kernels::CHUNK_LEN;
use crate::lane_kernels::source::IndexedSource;

/// Extension trait providing out-of-place lane-kernel methods on any [`IndexedSource`].
///
/// All methods have default implementations and are inherited via the blanket
/// `impl<S: IndexedSource> IndexedSourceExt for S` below. Bring the trait into
/// scope (`use vortex_compute::lane_kernels::IndexedSourceExt;`) to call
/// them with method syntax: `values.try_map_masked_into(&mask, &mut out, f)`.
pub trait IndexedSourceExt: IndexedSource + Sized {
    /// Fallible map with mask-aware error attribution. `f` returns `Option<R>`;
    /// `None` indicates a per-lane failure (e.g. range overflow on a narrowing cast).
    ///
    /// **Null-lane failures are filtered automatically.** The closure is called on
    /// every lane regardless of validity; if a null lane's stored value causes `f(v)`
    /// to return `None`, the kernel does *not* propagate that as `Err`. The per-lane
    /// `is_none()` flags are bit-packed into a `u64` at the lane's position, then
    /// AND-combined with the chunk's validity bitmap — null-lane bits vanish.
    ///
    /// The closure shape is the same as [`try_map_into`] (`FnMut(Item) -> Option<R>`);
    /// the mask parameter is what makes this kernel mask-aware. Callers that need to
    /// distinguish null lanes inside the closure (e.g. to short-circuit an expensive
    /// computation) should construct their own per-lane validity check externally; for
    /// the common case, the kernel's automatic filter is sufficient.
    ///
    /// On failure returns `Err(failing_lane_index)`. Lanes whose `f` returned `None`
    /// write `R::default()` into `out`, but the contents of `out` must not be relied
    /// upon when this function returns `Err`.
    ///
    /// [`try_map_into`]: IndexedSourceExt::try_map_into
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != mask.len()` or `out.len() != self.len()`.
    #[inline]
    fn try_map_masked_into<R, F>(
        self,
        mask: &BitBuffer,
        out: &mut [MaybeUninit<R>],
        mut f: F,
    ) -> Result<(), usize>
    where
        R: Copy + Default,
        F: FnMut(Self::Item) -> Option<R>,
    {
        #[inline(always)]
        fn chunk<S, R, F>(
            values: &S,
            out: &mut [MaybeUninit<R>],
            f: &mut F,
            src_chunk: u64,
            base: usize,
            count: usize,
        ) -> Option<usize>
        where
            S: IndexedSource,
            R: Copy + Default,
            F: FnMut(S::Item) -> Option<R>,
        {
            let mut fail_bits: u64 = 0;
            for bit_idx in 0..count {
                let idx = base + bit_idx;
                // SAFETY: caller guarantees base + count <= len.
                let val = unsafe { values.get_unchecked(idx) };
                let opt = f(val);
                fail_bits |= (opt.is_none() as u64) << bit_idx;
                let result = opt.unwrap_or_default();
                unsafe { out.get_unchecked_mut(idx).write(result) };
            }
            let valid_failures = fail_bits & src_chunk;
            (valid_failures != 0).then_some(base + valid_failures.trailing_zeros() as usize)
        }

        let values = self;
        let len = values.len();
        assert_eq!(len, mask.len(), "values and mask must have the same length");
        assert_eq!(out.len(), len, "out must have the same length as values");

        let chunks = mask.chunks();
        let chunks_count = len / 64;
        let remainder = len % 64;

        for (chunk_idx, src_chunk) in chunks.iter().enumerate() {
            if let Some(idx) = chunk(&values, out, &mut f, src_chunk, chunk_idx * 64, 64) {
                return Err(idx);
            }
        }
        if remainder != 0
            && let Some(idx) = chunk(
                &values,
                out,
                &mut f,
                chunks.remainder_bits(),
                chunks_count * 64,
                remainder,
            )
        {
            return Err(idx);
        }
        Ok(())
    }

    /// Apply `f(value)` lane-by-lane with **no validity awareness at all** — every
    /// closure invocation is treated as "happened", regardless of whether the lane
    /// is null. Use this only when the input is known non-nullable.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != self.len()`.
    #[inline]
    fn map_into<R, F>(self, out: &mut [MaybeUninit<R>], mut f: F)
    where
        F: FnMut(Self::Item) -> R,
    {
        #[inline(always)]
        fn chunk<S, R, F>(
            values: &S,
            out: &mut [MaybeUninit<R>],
            f: &mut F,
            base: usize,
            count: usize,
        ) where
            S: IndexedSource,
            F: FnMut(S::Item) -> R,
        {
            for bit_idx in 0..count {
                let idx = base + bit_idx;
                // SAFETY: caller guarantees base + count <= len.
                let val = unsafe { values.get_unchecked(idx) };
                unsafe { out.get_unchecked_mut(idx).write(f(val)) };
            }
        }

        let values = self;
        let len = values.len();
        assert_eq!(out.len(), len, "out must have the same length as values");

        let chunks_count = len / CHUNK_LEN;
        let remainder = len % CHUNK_LEN;

        for chunk_idx in 0..chunks_count {
            chunk(&values, out, &mut f, chunk_idx * CHUNK_LEN, CHUNK_LEN);
        }
        if remainder != 0 {
            chunk(&values, out, &mut f, chunks_count * CHUNK_LEN, remainder);
        }
    }

    /// Fallible map with **no validity awareness at all** — every `None` returned
    /// by the closure is treated as a failure, even at null lanes.
    ///
    /// # Use this only for non-nullable inputs.
    ///
    /// For nullable inputs with a fallible closure, use [`try_map_masked_into`] —
    /// it has the same value-only closure shape (and the same perf win) but
    /// **correctly suppresses null-lane failures** via per-chunk
    /// `fail_bits & mask_chunk`.
    ///
    /// Using this kernel on a nullable input where a null lane's stored value
    /// would cause `f` to return `None` will produce a spurious `Err`. This is a
    /// correctness footgun on purpose — the name and this doc are how the API
    /// signals "you must know your input has no nulls."
    ///
    /// On failure returns `Err(failing_lane_index)`.
    ///
    /// [`try_map_masked_into`]: IndexedSourceExt::try_map_masked_into
    ///
    /// # Panics
    ///
    /// Panics if `out.len() != self.len()`.
    #[inline]
    fn try_map_into<R, F>(self, out: &mut [MaybeUninit<R>], mut f: F) -> Result<(), usize>
    where
        R: Copy + Default,
        F: FnMut(Self::Item) -> Option<R>,
    {
        /// Returns `true` if any lane in `[base, base+count)` failed (OR-reduced);
        /// the cold attribution path is called at the kernel level so it can be
        /// inlined separately for full vs remainder.
        #[inline(always)]
        fn chunk<S, R, F>(
            values: &S,
            out: &mut [MaybeUninit<R>],
            f: &mut F,
            base: usize,
            count: usize,
        ) -> bool
        where
            S: IndexedSource,
            R: Copy + Default,
            F: FnMut(S::Item) -> Option<R>,
        {
            let mut fail_acc: u64 = 0;
            for bit_idx in 0..count {
                let idx = base + bit_idx;
                // SAFETY: caller guarantees base + count <= len.
                let val = unsafe { values.get_unchecked(idx) };
                let opt = f(val);
                fail_acc |= opt.is_none() as u64;
                let result = opt.unwrap_or_default();
                unsafe { out.get_unchecked_mut(idx).write(result) };
            }
            fail_acc != 0
        }

        let values = self;
        let len = values.len();
        assert_eq!(out.len(), len, "out must have the same length as values");

        let chunks_count = len / CHUNK_LEN;
        let remainder = len % CHUNK_LEN;

        for chunk_idx in 0..chunks_count {
            let base = chunk_idx * CHUNK_LEN;
            if chunk(&values, out, &mut f, base, CHUNK_LEN) {
                return Err(attribute_failure_no_mask(&values, base, CHUNK_LEN, &mut f));
            }
        }
        if remainder != 0 {
            let base = chunks_count * CHUNK_LEN;
            if chunk(&values, out, &mut f, base, remainder) {
                return Err(attribute_failure_no_mask(&values, base, remainder, &mut f));
            }
        }
        Ok(())
    }
}

impl<S: IndexedSource> IndexedSourceExt for S {}

/// Shared cold scan: walks a chunk, returns the first lane index where
/// `lane_fails(bit_idx, value)` returns `true`. Used by
/// [`attribute_failure_no_mask`].
///
/// Caller guarantees `base + chunk_len <= values.len()`.
#[cold]
#[inline(never)]
fn cold_scan<S>(
    values: &S,
    base: usize,
    chunk_len: usize,
    mut lane_fails: impl FnMut(usize /* bit_idx */, S::Item) -> bool,
) -> usize
where
    S: IndexedSource,
{
    for bit_idx in 0..chunk_len {
        let idx = base + bit_idx;
        // SAFETY: caller guarantees idx < values.len().
        let val = unsafe { values.get_unchecked(idx) };
        if lane_fails(bit_idx, val) {
            return idx;
        }
    }
    unreachable!("cold_scan called without a failing lane")
}

/// Cold attribution for the no-mask variant. Replays `f` over the chunk to find
/// the first lane that returns `None`.
#[inline]
fn attribute_failure_no_mask<S, R, F>(values: &S, base: usize, chunk_len: usize, f: &mut F) -> usize
where
    S: IndexedSource,
    F: FnMut(S::Item) -> Option<R>,
{
    cold_scan(values, base, chunk_len, |_bit_idx, val| f(val).is_none())
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::BitBufferMut;

    use super::*;

    fn write_t<T: Copy>(out: Vec<MaybeUninit<T>>) -> Vec<T> {
        // SAFETY: tests always fully initialize the buffer.
        unsafe { std::mem::transmute(out) }
    }

    #[test]
    fn try_map_masked_into_all_ok() {
        let values: Vec<u64> = (0..200).collect();
        let mask = BitBuffer::new_set(200);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got, (0..200u32).collect::<Vec<_>>());
    }

    #[test]
    fn try_map_masked_into_overflow_fails() {
        let mut values: Vec<u64> = (0..200).collect();
        values[137] = (u32::MAX as u64) + 1;
        let mask = BitBuffer::new_set(200);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res, Err(137));
    }

    #[test]
    fn try_map_masked_into_overflow_reports_first_failing_lane() {
        let mut values: Vec<u64> = (0..200).collect();
        values[50] = u64::MAX;
        values[51] = u64::MAX;
        values[137] = u64::MAX;
        let mask = BitBuffer::new_set(200);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res, Err(50));
    }

    #[test]
    fn try_map_masked_into_value_only_closure_filters_null_overflow() {
        let mut values: Vec<u64> = (0..200).collect();
        values[5] = u64::MAX;
        values[42] = u64::MAX;
        let mask = {
            let mut m = BitBufferMut::with_capacity(200);
            for i in 0..200 {
                m.append(i != 5 && i != 42);
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(
            res.is_ok(),
            "null-lane overflow should be filtered by the cold path"
        );
    }

    #[test]
    fn try_map_masked_into_value_only_closure_reports_first_valid_failure() {
        let mut values: Vec<u64> = (0..200).collect();
        values[5] = u64::MAX;
        values[42] = u64::MAX;
        values[77] = u64::MAX;
        values[100] = u64::MAX;
        let mask = {
            let mut m = BitBufferMut::with_capacity(200);
            for i in 0..200 {
                m.append(i != 5 && i != 42);
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res, Err(77));
    }

    #[test]
    fn try_map_masked_into_null_lane_bypasses_check() {
        let mut values: Vec<u64> = (0..200).collect();
        values[5] = u64::MAX;
        let mask = {
            let mut m = BitBufferMut::with_capacity(200);
            for i in 0..200 {
                m.append(i != 5);
            }
            m.freeze()
        };
        let mut out = vec![MaybeUninit::<u32>::uninit(); 200];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got[5], 0);
        assert_eq!(got[6], 6);
    }

    #[test]
    fn try_map_masked_into_branchful_matches_branchless() {
        let mut values: Vec<u64> = (0..130).map(|i| i as u64 * 7).collect();
        values[2] = u64::MAX;
        values[65] = u32::MAX as u64;
        let mask = {
            let mut m = BitBufferMut::with_capacity(130);
            for i in 0..130 {
                m.append(!matches!(i, 2 | 17 | 99));
            }
            m.freeze()
        };

        let mut branchless = vec![MaybeUninit::<u32>::uninit(); 130];
        let mut branchful = vec![MaybeUninit::<u32>::uninit(); 130];
        values
            .as_slice()
            .try_map_masked_into(&mask, &mut branchless, |v| {
                (v <= u32::MAX as u64).then_some(v as u32)
            })
            .unwrap();
        values
            .as_slice()
            .try_map_masked_into(&mask, &mut branchful, |v| u32::try_from(v).ok())
            .unwrap();

        assert_eq!(write_t(branchful), write_t(branchless));
    }

    #[test]
    fn try_map_masked_into_partial_chunk() {
        let values: Vec<u64> = (0..130).collect();
        let mask = BitBuffer::new_set(130);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got.len(), 130);
        assert_eq!(got[129], 129);
    }

    #[test]
    fn try_map_masked_into_sliced_mask_unaligned_offset() {
        let big = BitBuffer::new_set(256);
        let mask = big.slice(13..143);
        assert_eq!(mask.len(), 130);

        let values: Vec<u64> = (0..130).collect();
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(res.is_ok());
        let got = write_t(out);
        assert_eq!(got, (0..130u32).collect::<Vec<_>>());
    }

    #[test]
    fn try_map_masked_into_sliced_mask_with_overflow() {
        let big = BitBuffer::new_set(256);
        let mask = big.slice(13..143);
        assert_eq!(mask.len(), 130);

        let mut values: Vec<u64> = (0..130).collect();
        values[77] = u64::MAX;
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res, Err(77));
    }

    #[test]
    fn try_map_masked_into_sliced_mask_null_lanes() {
        let mut m = BitBufferMut::with_capacity(256);
        for i in 0..256 {
            m.append(i % 3 != 0);
        }
        let big = m.freeze();
        let mask = big.slice(13..143);
        assert_eq!(mask.len(), 130);

        let mut values: Vec<u64> = (0..130).collect();
        values[2] = u64::MAX;
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert!(res.is_ok(), "null lane should bypass the range check");
    }

    #[test]
    fn try_map_masked_into_overflow_in_remainder() {
        let mut values: Vec<u64> = (0..130).collect();
        values[129] = (u32::MAX as u64) + 1;
        let mask = BitBuffer::new_set(130);
        let mut out = vec![MaybeUninit::<u32>::uninit(); 130];
        let res = values.as_slice().try_map_masked_into(&mask, &mut out, |v| {
            (v <= u32::MAX as u64).then_some(v as u32)
        });
        assert_eq!(res, Err(129));
    }
}
