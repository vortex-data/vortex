// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;

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
/// The values must be laid end-to-end in `bytes`, one per entry of `lens`, describing the whole
/// buffer exactly. The returned buffers are zero-copy slices of `bytes`, numbered sequentially
/// from `start_buf_index`.
///
/// `max_buffer_len` must not exceed [`MAX_BUFFER_LEN`], since every view offset is stored in a
/// `u32` and offsets are bounded by `max_buffer_len`.
///
/// # Panics
///
/// Panics if the lengths do not describe `bytes` exactly, or if a single value exceeds
/// `max_buffer_len`.
pub fn build_views<P: NativePType + AsPrimitive<usize>>(
    start_buf_index: u32,
    max_buffer_len: usize,
    bytes: ByteBuffer,
    lens: &[P],
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let mut views = BufferMut::with_capacity(lens.len());
    let buffers = extend_views(
        &mut views,
        start_buf_index,
        max_buffer_len,
        &bytes,
        lens.len(),
        |i| lens[i].as_(),
    );
    (buffers, views.freeze())
}

/// [`build_views`] for values described by an offsets buffer instead of lengths.
///
/// `offsets` are absolute positions into `bytes` — the layout a `VarBinArray` stores — so there
/// is one more offset than there are values, and the values need not start at the beginning of
/// `bytes`: only `offsets[0]..offsets[last]` is referenced, and the returned buffers are zero-copy
/// slices of that range.
///
/// # Panics
///
/// Panics if `offsets` is empty, not monotonically non-decreasing within `bytes`, or if a single
/// value exceeds `max_buffer_len`.
pub fn build_views_from_offsets<P: NativePType + AsPrimitive<usize>>(
    start_buf_index: u32,
    max_buffer_len: usize,
    bytes: ByteBuffer,
    offsets: &[P],
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    assert!(!offsets.is_empty(), "offsets must hold at least one entry");
    let first: usize = offsets[0].as_();
    let last: usize = offsets[offsets.len() - 1].as_();
    let bytes = bytes.slice(first..last);

    let count = offsets.len() - 1;
    let mut views = BufferMut::with_capacity(count);
    // Wrapping keeps corrupt non-monotonic offsets from panicking on the subtraction itself; the
    // wrapped length then fails the in-bounds slicing (or `max_buffer_len`) checks in the loop.
    let buffers = extend_views(
        &mut views,
        start_buf_index,
        max_buffer_len,
        &bytes,
        count,
        |i| {
            AsPrimitive::<usize>::as_(offsets[i + 1])
                .wrapping_sub(AsPrimitive::<usize>::as_(offsets[i]))
        },
    );
    (buffers, views.freeze())
}

/// Appends one view per value straight into `views`, splitting `bytes` into buffers.
///
/// This is the core behind [`build_views`]: it writes into an existing views buffer so that a
/// [`VarBinViewBuilder`](crate::builders::VarBinViewBuilder) can build views directly into its
/// storage without an intermediate allocation. `len_at(i)` is the byte length of value `i`, and
/// the `count` lengths must describe `bytes` exactly. The returned buffers are zero-copy slices
/// of `bytes`, numbered sequentially from `start_buf_index`.
pub(crate) fn extend_views(
    views: &mut BufferMut<BinaryView>,
    start_buf_index: u32,
    max_buffer_len: usize,
    bytes: &ByteBuffer,
    count: usize,
    len_at: impl Fn(usize) -> usize,
) -> Vec<ByteBuffer> {
    assert!(
        max_buffer_len <= MAX_BUFFER_LEN,
        "max_buffer_len cannot exceed MAX_BUFFER_LEN, offsets must fit in u32"
    );

    if bytes.len() <= max_buffer_len {
        // Common case: the whole decoded heap fits within a single buffer, so no rollover can
        // occur (`bytes.len()` is the total decoded size and therefore an upper bound on every
        // offset).
        extend_views_single_buffer(views, start_buf_index, bytes, count, len_at);
        if bytes.is_empty() {
            Vec::new()
        } else {
            vec![bytes.clone()]
        }
    } else {
        extend_views_rolling(views, start_buf_index, max_buffer_len, bytes, count, len_at)
    }
}

/// Build views when the whole heap fits in a single output buffer.
///
/// Because no rollover can occur, the hot loop drops the per-element rollover branch and constructs
/// reference views inline, avoiding the out-of-line `BinaryView::make_view` call for the common
/// long-string case. Every offset is bounded by `bytes.len()`, which the caller has guaranteed is
/// at most [`MAX_BUFFER_LEN`], so the `usize -> u32` conversions cannot truncate.
fn extend_views_single_buffer(
    views: &mut BufferMut<BinaryView>,
    buf_index: u32,
    bytes: &ByteBuffer,
    count: usize,
    len_at: impl Fn(usize) -> usize,
) {
    views.reserve(count);
    let base = views.len();

    let data = bytes.as_slice();
    let mut offset = 0usize;
    // Write directly into the reserved spare capacity rather than `push_unchecked`. The latter
    // advances the backing buffer's length on every call, which the optimizer cannot prove is
    // loop-invariant, so it reloads and rewrites the output cursor through the stack each
    // iteration. Writing into the spare slice keeps the cursor in a register and the length is
    // set once after the loop.
    let spare = &mut views.spare_capacity_mut()[..count];
    for (i, slot) in spare.iter_mut().enumerate() {
        let len = len_at(i);
        let value = &data[offset..offset + len];
        let view = if len > BinaryView::MAX_INLINED_SIZE {
            let mut prefix = [0u8; 4];
            prefix.copy_from_slice(&value[..4]);
            BinaryView::new_ref(len.as_(), prefix, buf_index, offset.as_())
        } else {
            BinaryView::make_view(value, buf_index, offset.as_())
        };
        slot.write(view);
        offset += len;
    }
    assert_eq!(
        offset,
        data.len(),
        "value lengths must describe the byte heap exactly"
    );
    // SAFETY: the loop initialized exactly `count` contiguous views (`spare` has at least
    //  `count` slots).
    unsafe { views.set_len(base + count) };
}

/// Build views when the heap exceeds `max_buffer_len` and must be split across multiple buffers.
///
/// The buffer is rolled over every `max_buffer_len` bytes so that no view offset overflows the
/// `u32` offset field. Each output buffer is a zero-copy slice of `bytes`.
fn extend_views_rolling(
    views: &mut BufferMut<BinaryView>,
    start_buf_index: u32,
    max_buffer_len: usize,
    bytes: &ByteBuffer,
    count: usize,
    len_at: impl Fn(usize) -> usize,
) -> Vec<ByteBuffer> {
    views.reserve(count);
    let mut buffers = Vec::new();
    let mut buf_index = start_buf_index;

    let data = bytes.as_slice();
    // The absolute start of the current segment, and the offset of the next value within it.
    let mut segment_start = 0usize;
    let mut offset = 0usize;
    for i in 0..count {
        let len = len_at(i);
        assert!(len <= max_buffer_len, "values cannot exceed max_buffer_len");

        if offset + len > max_buffer_len {
            // Roll the buffer every 2GiB, to avoid overflowing VarBinView offset field
            buffers.push(bytes.slice(segment_start..segment_start + offset));
            buf_index += 1;
            segment_start += offset;
            offset = 0;
        }
        let start = segment_start + offset;
        let view = BinaryView::make_view(&data[start..start + len], buf_index, offset.as_());
        views.push(view);
        offset += len;
    }
    assert_eq!(
        segment_start + offset,
        data.len(),
        "value lengths must describe the byte heap exactly"
    );

    if segment_start < data.len() {
        buffers.push(bytes.slice(segment_start..data.len()));
    }

    buffers
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;

    use crate::arrays::varbinview::BinaryView;
    use crate::arrays::varbinview::build_views::MAX_BUFFER_LEN;
    use crate::arrays::varbinview::build_views::build_views;
    use crate::arrays::varbinview::build_views::build_views_from_offsets;

    /// Concatenate `values` into a single byte heap and return it alongside the per-element lengths,
    /// matching the `(bytes, lens)` inputs that `build_views` consumes.
    fn flatten(values: &[&[u8]]) -> (ByteBuffer, Vec<u32>) {
        let mut bytes = ByteBufferMut::empty();
        let mut lens = Vec::with_capacity(values.len());
        for v in values {
            bytes.extend_from_slice(v);
            lens.push(u32::try_from(v.len()).unwrap());
        }
        (bytes.freeze(), lens)
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
            // The fast path adopts the input heap unchanged.
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

    /// The output buffers must be zero-copy slices of the input heap, on both paths — a copy here
    /// silently doubles the memory cost of every decode that feeds views.
    #[test]
    fn output_buffers_are_zero_copy() {
        let values: &[&[u8]] = &[
            b"first long reference value",
            b"tiny",
            b"second long reference value!!",
            b"third looooong reference value",
        ];
        let (bytes, lens) = flatten(values);
        let base = bytes.as_ptr();

        // Fast path: the single output buffer is the input buffer.
        let (buffers, _views) = build_views(0, bytes.len() + 1, bytes.clone(), &lens);
        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].as_ptr(), base, "fast path must not copy");

        // Rolling path: every output buffer points into the input allocation.
        let longest = values.iter().map(|v| v.len()).max().unwrap();
        let (buffers, _views) = build_views(0, longest, bytes, &lens);
        assert!(buffers.len() > 1);
        let mut expected_ptr = base;
        for buffer in &buffers {
            assert_eq!(buffer.as_ptr(), expected_ptr, "rolling path must not copy");
            // SAFETY: the buffers partition the input heap, so the next one starts where
            // this one ends, still within (or one past) the original allocation.
            expected_ptr = unsafe { expected_ptr.add(buffer.len()) };
        }
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
        const { assert!((N - 1) * LEN > (1 << 23)) };

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
        let (buffers, views) = build_views(0, 1024, ByteBuffer::empty(), &lens);
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

    /// The offsets-driven variant must agree with the lengths-driven one, reference only the
    /// `offsets[0]..offsets[last]` range, and stay zero-copy — it exists so a `VarBinArray` heap
    /// can feed views without materializing a lengths buffer or copying its bytes.
    #[test]
    fn from_offsets_matches_lengths_and_is_zero_copy() {
        // A heap with a prefix and suffix outside the offsets range, as a sliced VarBin has.
        let heap = ByteBuffer::copy_from(b"..a long value that is referenced!tiny..".as_slice());
        let offsets: Vec<u32> = vec![2, 34, 38];

        let (buffers, views) = build_views_from_offsets(5, MAX_BUFFER_LEN, heap.clone(), &offsets);

        assert_eq!(buffers.len(), 1);
        // Zero-copy: the buffer points at offset 2 of the original allocation.
        // SAFETY: offset 2 is in bounds of the 40-byte heap.
        assert_eq!(buffers[0].as_ptr(), unsafe { heap.as_ptr().add(2) });
        assert_eq!(buffers[0].len(), 36);

        assert_eq!(
            reconstruct(&buffers, &views, 5),
            vec![
                b"a long value that is referenced!".to_vec(),
                b"tiny".to_vec()
            ]
        );
    }

    /// Lengths that do not cover the heap exactly are a caller bug and must be rejected rather
    /// than silently emitting views over a partially-covered buffer.
    #[test]
    #[should_panic(expected = "value lengths must describe the byte heap exactly")]
    fn short_lengths_panic() {
        let (bytes, _) = flatten(&[b"a long value that is referenced", b"tiny"]);
        build_views(0, MAX_BUFFER_LEN, bytes, &[31u32]);
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
        let (buffers, views) = build_views(0, MAX_BUFFER_LEN, bytes.freeze(), &lens);

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
            ByteBuffer::copy_from("aaaaaaaaaaaaabbbbbbbbbbbbbcccccccccccccddddddddddddd");
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
        build_views(0, MAX_BUFFER_LEN + 1, ByteBuffer::copy_from("abc"), &[3u32]);
    }
}
