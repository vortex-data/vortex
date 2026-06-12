// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;

pub use crate::arrays::varbinview::BinaryView;
use crate::dtype::NativePType;

/// Convert an offsets buffer to a buffer of element lengths.
#[inline]
pub fn offsets_to_lengths<P: NativePType>(offsets: &[P]) -> Buffer<P> {
    offsets
        .iter()
        .tuple_windows::<(_, _)>()
        .map(|(&start, &end)| end - start)
        .collect()
}

/// Maximum number of buffer bytes that can be referenced by a single `BinaryView`
pub const MAX_BUFFER_LEN: usize = i32::MAX as usize;

/// Split a large buffer of input `bytes` holding string data into `VarBinView` buffers and views.
///
/// `max_buffer_len` must not exceed [`MAX_BUFFER_LEN`], since every view offset is stored in a
/// `u32` and offsets are bounded by `max_buffer_len`.
pub fn build_views<P: NativePType + AsPrimitive<usize>>(
    start_buf_index: u32,
    max_buffer_len: usize,
    bytes: ByteBufferMut,
    lens: &[P],
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    assert!(
        max_buffer_len <= MAX_BUFFER_LEN,
        "max_buffer_len cannot exceed MAX_BUFFER_LEN, offsets must fit in u32"
    );

    if bytes.len() <= max_buffer_len {
        // Common case: the whole decoded heap fits within a single buffer, so no rollover can occur
        // (`bytes.len()` is the total decoded size and therefore an upper bound on every offset).
        build_views_single_buffer(start_buf_index, bytes, lens)
    } else {
        build_views_rolling(start_buf_index, max_buffer_len, bytes, lens)
    }
}

/// Build views when the whole heap fits in a single output buffer.
///
/// Because no rollover can occur, the hot loop drops the per-element rollover branch and constructs
/// reference views inline, avoiding the out-of-line `BinaryView::make_view` call for the common
/// long-string case. Every offset is bounded by `bytes.len()`, which the caller has guaranteed is
/// at most [`MAX_BUFFER_LEN`], so the `usize -> u32` conversions cannot truncate.
fn build_views_single_buffer<P: NativePType + AsPrimitive<usize>>(
    start_buf_index: u32,
    bytes: ByteBufferMut,
    lens: &[P],
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let mut views = BufferMut::<BinaryView>::with_capacity(lens.len());

    let data = bytes.as_slice();
    let mut offset = 0usize;
    // Write directly into the reserved spare capacity rather than `push_unchecked`. The latter
    // advances the backing buffer's length on every call, which the optimizer cannot prove is
    // loop-invariant, so it reloads and rewrites the output cursor through the stack each
    // iteration. Writing into the spare slice keeps the cursor in a register and the length is
    // set once after the loop.
    let spare = views.spare_capacity_mut();
    for (slot, &len) in spare.iter_mut().zip(lens) {
        let len = len.as_();
        let value = &data[offset..offset + len];
        let view = if len > BinaryView::MAX_INLINED_SIZE {
            let mut prefix = [0u8; 4];
            prefix.copy_from_slice(&value[..4]);
            BinaryView::new_ref(len.as_(), prefix, start_buf_index, offset.as_())
        } else {
            BinaryView::make_view(value, start_buf_index, offset.as_())
        };
        slot.write(view);
        offset += len;
    }
    // SAFETY: the loop initialized exactly `lens.len()` contiguous views (`spare` has at least
    //  `lens.len()` slots, and `zip` stops at the shorter operand).
    unsafe { views.set_len(lens.len()) };

    let buffers = if bytes.is_empty() {
        Vec::new()
    } else {
        vec![bytes.freeze()]
    };
    (buffers, views.freeze())
}

/// Build views when the heap exceeds `max_buffer_len` and must be split across multiple buffers.
///
/// The buffer is rolled over every `max_buffer_len` bytes so that no view offset overflows the
/// `u32` offset field.
fn build_views_rolling<P: NativePType + AsPrimitive<usize>>(
    start_buf_index: u32,
    max_buffer_len: usize,
    mut bytes: ByteBufferMut,
    lens: &[P],
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let mut views = BufferMut::<BinaryView>::with_capacity(lens.len());
    let mut buffers = Vec::new();
    let mut buf_index = start_buf_index;

    let mut offset = 0;
    for &len in lens {
        let len = len.as_();
        assert!(len <= max_buffer_len, "values cannot exceed max_buffer_len");

        if (offset + len) > max_buffer_len {
            // Roll the buffer every 2GiB, to avoid overflowing VarBinView offset field
            let rest = bytes.split_off(offset);

            buffers.push(bytes.freeze());
            buf_index += 1;
            offset = 0;

            bytes = rest;
        }
        let view = BinaryView::make_view(&bytes[offset..][..len], buf_index, offset.as_());
        // SAFETY: we reserved the right capacity beforehand
        unsafe { views.push_unchecked(view) };
        offset += len;
    }

    if !bytes.is_empty() {
        buffers.push(bytes.freeze());
    }

    (buffers, views.freeze())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;

    use crate::arrays::varbinview::BinaryView;
    use crate::arrays::varbinview::build_views::MAX_BUFFER_LEN;
    use crate::arrays::varbinview::build_views::build_views;

    /// Concatenate `values` into a single byte heap and return it alongside the per-element lengths,
    /// matching the `(bytes, lens)` inputs that `build_views` consumes.
    fn flatten(values: &[&[u8]]) -> (ByteBufferMut, Vec<u32>) {
        let mut bytes = ByteBufferMut::empty();
        let mut lens = Vec::with_capacity(values.len());
        for v in values {
            bytes.extend_from_slice(v);
            lens.push(u32::try_from(v.len()).unwrap());
        }
        (bytes, lens)
    }

    /// Reconstruct the logical value behind each view by dereferencing it through the output
    /// buffers. The first buffer corresponds to `start_buf_index`, so buffer indices are rebased by
    /// that amount. This is the core correctness invariant: regardless of which code path built the
    /// views, every view must point back at its original bytes.
    fn reconstruct(
        buffers: &[ByteBuffer],
        views: &[BinaryView],
        start_buf_index: u32,
    ) -> Vec<Vec<u8>> {
        views
            .iter()
            .map(|view| {
                if view.is_inlined() {
                    view.as_inlined().value().to_vec()
                } else {
                    let r = view.as_view();
                    let buf = &buffers[(r.buffer_index - start_buf_index) as usize];
                    buf[r.as_range()].to_vec()
                }
            })
            .collect()
    }

    /// The single-buffer fast path (`bytes.len() <= max_buffer_len`) must reproduce every input
    /// value exactly, emit a single output buffer holding the untouched heap, and reference only
    /// `start_buf_index`. We cover a spread of value sets that mix inlined (<= 12 bytes) and
    /// reference (> 12 bytes) lengths, including the 12/13 byte inline boundary, empty values, and a
    /// fully-inlined set.
    #[rstest]
    #[case::mixed(&[b"a".as_slice(), b"this is a long reference value", b"short", b"another long value here!!"])]
    #[case::inline_boundary(&[&[b'x'; 12] as &[u8], &[b'y'; 13], &[b'z'; 12], &[b'w'; 13]])]
    #[case::all_inlined(&[b"".as_slice(), b"a", b"bb", b"ccc", b"dddddddddddd"])]
    #[case::all_reference(&[&[b'a'; 100] as &[u8], &[b'b'; 50], &[b'c'; 4096]])]
    #[case::empty_values_interleaved(&[b"".as_slice(), b"a long value that is referenced", b"", b"", b"trailing long reference value"])]
    #[case::single_long(&[&[7u8; 1 << 16] as &[u8]])]
    fn fast_path_roundtrip(#[case] values: &[&[u8]]) {
        let (bytes, lens) = flatten(values);
        let total = bytes.len();
        let start_buf_index = 3;

        // `max_buffer_len` strictly greater than the heap forces the single-buffer fast path.
        let (buffers, views) = build_views(start_buf_index, total + 1, bytes, &lens);

        assert_eq!(views.len(), values.len());
        if total == 0 {
            assert!(buffers.is_empty(), "empty heap must not allocate a buffer");
        } else {
            assert_eq!(buffers.len(), 1, "whole heap must stay in one buffer");
            // The fast path freezes the input heap unchanged.
            let concatenated: Vec<u8> = values.concat();
            assert_eq!(buffers[0].as_slice(), concatenated.as_slice());
        }
        for view in views.iter() {
            if !view.is_inlined() {
                assert_eq!(view.as_view().buffer_index, start_buf_index);
            }
        }

        let expected: Vec<Vec<u8>> = values.iter().map(|v| v.to_vec()).collect();
        assert_eq!(reconstruct(&buffers, &views, start_buf_index), expected);
    }

    /// Offsets and sizes are written into the `u32` `Ref` fields via `as_` truncation, so we must
    /// confirm they stay correct once the running offset grows well past the 16-bit range (i.e. is
    /// not narrowed to a smaller width). A ~9 MiB heap pushes offsets above 2^23 while remaining far
    /// below `MAX_BUFFER_LEN`; each value encodes its index in its first bytes so a misplaced offset
    /// would reconstruct the wrong bytes.
    #[test]
    fn fast_path_large_offsets() {
        const N: usize = 9000;
        const LEN: usize = 1000;
        // The final offset is (N - 1) * LEN, which must exceed 2^23 to be a meaningful check.
        const _: () = assert!((N - 1) * LEN > (1 << 23));

        let values: Vec<Vec<u8>> = (0..N)
            .map(|i| {
                let mut v = vec![0u8; LEN];
                v[..4].copy_from_slice(&u32::try_from(i).unwrap().to_le_bytes());
                v
            })
            .collect();
        let refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();

        let (bytes, lens) = flatten(&refs);
        let total = bytes.len();

        let (buffers, views) = build_views(0, total + 1, bytes, &lens);

        assert_eq!(buffers.len(), 1);
        // The recorded offset must equal the cumulative byte position, exactly, for every view.
        for (i, view) in views.iter().enumerate() {
            let r = view.as_view();
            assert_eq!(r.offset as usize, i * LEN, "wrong offset for view {i}");
            assert_eq!(r.size as usize, LEN);
        }
        assert_eq!(reconstruct(&buffers, &views, 0), values);
    }

    /// The fast path is taken when `bytes.len() <= max_buffer_len`, so equality at the boundary must
    /// still produce a single buffer (not roll over to the slow path).
    #[test]
    fn fast_path_taken_at_exact_boundary() {
        let (bytes, lens) =
            flatten(&[b"this value is definitely long", b"and so is this one here"]);
        let total = bytes.len();

        let (buffers, views) = build_views(0, total, bytes, &lens);

        assert_eq!(
            buffers.len(),
            1,
            "len == max_buffer_len must stay on fast path"
        );
        assert_eq!(views.len(), 2);
    }

    /// For the same logical data, the fast path (single buffer) and the slow rollover path must
    /// reconstruct identical values. Driving the slow path with a small `max_buffer_len` forces
    /// buffer splitting while leaving the recovered values unchanged.
    #[test]
    fn fast_and_slow_paths_agree() {
        let values: &[&[u8]] = &[
            b"first long reference value",
            b"tiny",
            b"second long reference value!!",
            b"third looooong reference value",
        ];
        let expected: Vec<Vec<u8>> = values.iter().map(|v| v.to_vec()).collect();

        let (fast_bytes, lens) = flatten(values);
        let total = fast_bytes.len();
        let (fast_buffers, fast_views) = build_views(0, total + 1, fast_bytes, &lens);
        assert_eq!(fast_buffers.len(), 1);
        assert_eq!(reconstruct(&fast_buffers, &fast_views, 0), expected);

        // Force the rollover path: a small cap (>= the longest value) that the total heap exceeds.
        let longest = values.iter().map(|v| v.len()).max().unwrap();
        let (slow_bytes, _) = flatten(values);
        let (slow_buffers, slow_views) = build_views(0, longest, slow_bytes, &lens);
        assert!(
            slow_buffers.len() > 1,
            "small cap should split into many buffers"
        );
        assert_eq!(reconstruct(&slow_buffers, &slow_views, 0), expected);

        // Same logical contents regardless of how the heap was partitioned.
        assert_eq!(
            reconstruct(&fast_buffers, &fast_views, 0),
            reconstruct(&slow_buffers, &slow_views, 0)
        );
    }

    /// Empty input must yield no buffers and no views, exercising the `bytes.is_empty()` branch.
    #[test]
    fn fast_path_empty_input() {
        let lens: Vec<u32> = Vec::new();
        let (buffers, views) = build_views(0, 1024, ByteBufferMut::empty(), &lens);
        assert!(buffers.is_empty());
        assert!(views.is_empty());
    }

    /// The fast path must produce views byte-identical to the value-inspecting `make_view`, which is
    /// what the slow path uses. This pins the inline/reference decision and field layout.
    #[test]
    fn fast_path_matches_make_view() {
        let values: &[&[u8]] = &[b"inline", b"this is a long reference value", b""];
        let (bytes, lens) = flatten(values);
        let total = bytes.len();
        let (_buffers, views) = build_views(0, total + 1, bytes, &lens);

        let expected = [
            BinaryView::make_view(b"inline", 0, 0),
            BinaryView::make_view(b"this is a long reference value", 0, 6),
            BinaryView::make_view(b"", 0, 36),
        ];
        assert_eq!(views.as_slice(), &expected);
    }

    // TODO(someone): ideally CI would run this in release mode as well, since debug builds make the
    // ~2.25 GiB allocation and fill loop substantially slower.
    /// Slow regression for the single-buffer fast-path guard. The fast path is only valid when the
    /// whole heap fits in one buffer (`bytes.len() <= max_buffer_len`); once the heap exceeds
    /// [`MAX_BUFFER_LEN`] (`i32::MAX`, ~2.0 GiB) `build_views` must roll the heap into multiple
    /// buffers, resetting the per-buffer offset, so no view references an offset past the
    /// `i32`-bounded buffer limit.
    ///
    /// We build a heap just past `i32::MAX` and assert it rolls over into more than one buffer, that
    /// no buffer exceeds `MAX_BUFFER_LEN`, and that values straddling the rollover boundary (where
    /// the second buffer's offsets restart from zero) reconstruct exactly. If the guard regressed and
    /// the fast path swallowed the whole heap, it would emit a single >2 GiB buffer with offsets past
    /// `i32::MAX`, which the buffer-count and buffer-size assertions catch.
    ///
    /// Allocates ~2.25 GiB, so it is gated to CI and skipped when `VORTEX_SKIP_SLOW_TESTS` is set:
    ///
    /// ```text
    /// CI=1 cargo test --release -p vortex-array build_views_offsets_overflow
    /// ```
    ///
    /// [`MAX_BUFFER_LEN`]: super::MAX_BUFFER_LEN
    #[test_with::env(CI)]
    #[test_with::no_env(VORTEX_SKIP_SLOW_TESTS)]
    fn build_views_offsets_overflow_i32() {
        const STRING_LEN: usize = 64 * 1024;
        // Comfortably past MAX_BUFFER_LEN (`i32::MAX` ~= 2.0 GiB) so the heap must roll over.
        const TOTAL_BYTES: usize = (1usize << 31) + (256 << 20); // ~2.25 GiB
        const N: usize = TOTAL_BYTES / STRING_LEN;

        // Each value's first 8 bytes encode its row index, so a misrouted offset is detectable.
        let nth_string = |i: usize| {
            let mut s = vec![b'x'; STRING_LEN];
            s[..8].copy_from_slice(&(i as u64).to_le_bytes());
            s
        };

        let mut bytes = ByteBufferMut::with_capacity(N * STRING_LEN);
        let mut value = vec![b'x'; STRING_LEN];
        for i in 0..N {
            value[..8].copy_from_slice(&(i as u64).to_le_bytes());
            bytes.extend_from_slice(&value);
        }

        let lens = vec![u32::try_from(STRING_LEN).unwrap(); N];
        let (buffers, views) = build_views(0, MAX_BUFFER_LEN, bytes, &lens);

        assert_eq!(views.len(), N);
        assert!(
            buffers.len() >= 2,
            "heap exceeding MAX_BUFFER_LEN must roll over into multiple buffers, got {}",
            buffers.len()
        );
        for (i, b) in buffers.iter().enumerate() {
            assert!(
                b.len() <= MAX_BUFFER_LEN,
                "buffer {i} of {} bytes exceeds MAX_BUFFER_LEN",
                b.len()
            );
        }

        // The boundary row is the first whose offset would cross MAX_BUFFER_LEN on the fast path.
        let boundary = MAX_BUFFER_LEN / STRING_LEN;
        for i in [0, boundary - 1, boundary, boundary + 1, N / 2, N - 1] {
            let view = &views[i];
            let r = view.as_view();
            let got = &buffers[r.buffer_index as usize][r.as_range()];
            assert_eq!(got, nth_string(i).as_slice(), "value mismatch at row {i}");
            assert_eq!(r.size as usize, STRING_LEN);
        }
    }

    #[test]
    fn test_to_canonical_large() {
        // We are testing generating views for raw data that should look like
        //
        //    aaaaaaaaaaaaa ("a"*13)
        //    bbbbbbbbbbbbb ("b"*13)
        //    ccccccccccccc ("c"*13)
        //    ddddddddddddd ("d"*13)
        //
        // In real code, this would all fit in one buffer, but to unit test the splitting logic
        // we split buffers at length 26, which should result in two buffers for the output array.
        let raw_data =
            ByteBufferMut::copy_from("aaaaaaaaaaaaabbbbbbbbbbbbbcccccccccccccddddddddddddd");
        let lens = vec![13u8; 4];

        let (buffers, views) = build_views(0, 26, raw_data, &lens);

        assert_eq!(
            buffers,
            vec![
                ByteBuffer::copy_from("aaaaaaaaaaaaabbbbbbbbbbbbb"),
                ByteBuffer::copy_from("cccccccccccccddddddddddddd"),
            ]
        );

        assert_eq!(
            views.as_slice(),
            &[
                BinaryView::make_view(b"aaaaaaaaaaaaa", 0, 0),
                BinaryView::make_view(b"bbbbbbbbbbbbb", 0, 13),
                BinaryView::make_view(b"ccccccccccccc", 1, 0),
                BinaryView::make_view(b"ddddddddddddd", 1, 13),
            ]
        )
    }

    #[test]
    #[should_panic(expected = "max_buffer_len cannot exceed MAX_BUFFER_LEN")]
    fn test_max_buffer_len_too_large_panics() {
        build_views(
            0,
            MAX_BUFFER_LEN + 1,
            ByteBufferMut::copy_from("abc"),
            &[3u32],
        );
    }
}
