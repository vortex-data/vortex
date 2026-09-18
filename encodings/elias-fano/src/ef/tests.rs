// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::convert::Infallible;

use rstest::rstest;

use super::*;

// ── Geometry ───────────────────────────────────────────────────────────

#[rstest]
// Four elements over a universe of 18: 2 low bits, 4 buckets.
#[case(17, 4, 2, 10)]
// A dense run 0..n: universe == n, so no low bits, and the upper array is 2n + 1.
#[case(999, 1000, 0, 2001)]
// Sparse: 1000 elements over a 2^20 universe wants 10 low bits, leaving 1024 buckets.
#[case((1 << 20) - 1, 1000, 10, 1000 + 1023 + 2)]
// All-equal input. Span 0 means one bucket and no low bits.
#[case(0, 100, 0, 102)]
// A single element at the very top of the u64 range: the clamp fires.
#[case(u64::MAX, 1, 63, 4)]
fn test_geometry(
    #[case] span: u64,
    #[case] n: usize,
    #[case] expected_width: u8,
    #[case] expected_upper_len: u64,
) -> Result<(), Error> {
    let width = lower_width(span, n);
    assert_eq!(width, expected_width, "lower_width");
    assert_eq!(upper_len(span, n, width)?, expected_upper_len, "upper_len");
    Ok(())
}

/// The upper array must always be long enough to hold the position that the highest element
/// claims, and to leave at least one zero rank above the highest one a query can name.
#[rstest]
#[case(0, 1)]
#[case(1, 1)]
#[case(u64::MAX, 1)]
#[case(u64::MAX, 1024)]
#[case(1_000_000, 100_000)]
#[case(7, 8)]
#[case(255, 256)]
fn test_upper_len_leaves_room(#[case] span: u64, #[case] n: usize) -> Result<(), Error> {
    let width = lower_width(span, n);
    let upper_len = upper_len(span, n, width)?;

    // The last element sits at `(span >> width) + (n - 1) + 1`, which must be in bounds.
    let last_position = (span >> width) + n as u64;
    assert!(last_position < upper_len, "last position {last_position}");

    // A reseat may name any zero rank up to the maximum element's high part.
    let max_zero_rank = span >> width;
    assert!(max_zero_rank < num_zeros(upper_len, n), "max zero rank");
    Ok(())
}

#[test]
fn test_lower_mask() {
    assert_eq!(lower_mask(0), 0);
    assert_eq!(lower_mask(1), 1);
    assert_eq!(lower_mask(8), 0xFF);
    assert_eq!(lower_mask(63), u64::MAX >> 1);
}

/// One sample per `1 << LOG_SAMPLING1` set bits above the first, and none for an empty sequence.
#[test]
fn test_num_samples1() {
    assert_eq!(num_samples1(0), 0);
    assert_eq!(num_samples1(1), 0);
    assert_eq!(num_samples1(1 << LOG_SAMPLING1), 0);
    assert_eq!(num_samples1((1 << LOG_SAMPLING1) + 1), 1);
    assert_eq!(num_samples1(1 << (LOG_SAMPLING1 + 1)), 1);
    assert_eq!(num_samples1((1 << (LOG_SAMPLING1 + 1)) + 1), 2);
}

// ── Select ─────────────────────────────────────────────────────────────

/// A pattern near 50% density, deliberately not byte-periodic.
fn mixed_bytes(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37) ^ 0x5A)
        .collect()
}

/// The bit at `index` of a window beginning `offset` bits into `bytes`, read one bit at a time.
fn bit_at(bytes: &[u8], offset: usize, index: usize) -> bool {
    let at = offset + index;
    (bytes[at / 8] >> (at % 8)) & 1 == 1
}

/// Positions of the bits equal to `want`, as the reference answer.
fn naive_positions(
    bytes: &[u8],
    offset: usize,
    start: usize,
    end: usize,
    want: bool,
) -> Vec<usize> {
    (start..end)
        .filter(|&i| bit_at(bytes, offset, i) == want)
        .map(|i| i - start)
        .collect()
}

/// Both variants must agree with the bit-at-a-time reference across the whole rank range, and
/// both must report `None` one past the last rank.
fn check_against_naive(bytes: &[u8], offset: usize, len: usize, start: usize, end: usize) {
    let bits = Bits::new(bytes, offset, len);
    for (want, select) in [
        (
            true,
            select_range as fn(Bits<'_>, usize, usize, usize) -> Option<usize>,
        ),
        (
            false,
            select_zero_range as fn(Bits<'_>, usize, usize, usize) -> Option<usize>,
        ),
    ] {
        let expected = naive_positions(bytes, offset, start, end, want);
        for (nth, &expected_pos) in expected.iter().enumerate() {
            assert_eq!(
                select(bits, start, end, nth),
                Some(expected_pos),
                "want={want} offset={offset} start={start} end={end} nth={nth}"
            );
        }
        assert_eq!(
            select(bits, start, end, expected.len()),
            None,
            "want={want} offset={offset} start={start} end={end} past-the-end rank"
        );
    }
}

#[rstest]
#[case(0, 0, 128)]
#[case(3, 0, 100)]
#[case(7, 0, 50)]
#[case(0, 0, 1)]
#[case(0, 0, 64)]
#[case(1, 0, 64)]
#[case(0, 0, 65)]
#[case(3, 0, 256)]
#[case(0, 0, 512)]
#[case(0, 0, 513)]
#[case(5, 0, 1024)]
// Windows that start partway in, which is the shape a sample lower bound produces.
#[case(0, 1, 128)]
#[case(0, 63, 128)]
#[case(0, 64, 200)]
#[case(0, 65, 200)]
#[case(3, 70, 300)]
#[case(5, 511, 1024)]
// Sub-word windows, where the only word's valid width is what the zero path must respect.
#[case(1, 0, 1)]
#[case(1, 2, 8)]
#[case(4, 3, 6)]
#[case(7, 0, 1)]
#[case(0, 9, 17)]
#[case(2, 71, 71)]
fn select_agrees_with_naive(#[case] offset: usize, #[case] start: usize, #[case] end: usize) {
    let bytes = mixed_bytes((offset + end).div_ceil(8) + 1);
    check_against_naive(&bytes, offset, end, start, end);
}

/// 50% density is exactly where confusing ones with zeros is least visible; these make it
/// obvious.
#[rstest]
#[case::all_zero(0x00)]
#[case::all_one(0xFF)]
#[case::sparse(0x01)]
#[case::dense(0xFE)]
fn select_uniform_density(#[case] fill: u8) {
    for (offset, start, end) in [
        (0usize, 0usize, 8usize),
        (0, 0, 128),
        (3, 2, 5),
        (5, 7, 130),
        (1, 200, 517),
    ] {
        let bytes = vec![fill; (offset + end).div_ceil(8) + 1];
        check_against_naive(&bytes, offset, end, start, end);
    }
}

#[test]
fn select_degenerate_buffers() {
    // All ones: no zero to find, at any rank.
    let ones = vec![0xFFu8; 17];
    let ones = Bits::new(&ones, 0, 128);
    assert_eq!(select_zero_range(ones, 0, 128, 0), None);
    assert_eq!(select_range(ones, 0, 128, 127), Some(127));

    // All zeros: the nth zero is at position n, and no set bit exists.
    let zeros = vec![0x00u8; 17];
    let zeros = Bits::new(&zeros, 0, 128);
    for nth in 0..128 {
        assert_eq!(
            select_zero_range(zeros, 0, 128, nth),
            Some(nth),
            "nth={nth}"
        );
    }
    assert_eq!(select_zero_range(zeros, 0, 128, 128), None);
    assert_eq!(select_range(zeros, 0, 128, 0), None);
}

/// `window_words` has to reproduce the window bit for bit, and zero-pad above it so that counting
/// set bits over the result counts exactly the window's — which is what the decoder relies on.
#[rstest]
#[case(0, 0, 128)]
#[case(3, 5, 130)]
#[case(7, 1, 64)]
#[case(0, 63, 65)]
#[case(2, 0, 1)]
#[case(5, 100, 100)]
#[case(1, 7, 1000)]
fn window_words_reproduces_the_window(
    #[case] offset: usize,
    #[case] start: usize,
    #[case] end: usize,
) {
    let bytes = mixed_bytes((offset + end).div_ceil(8) + 1);
    let bits = Bits::new(&bytes, offset, end);
    let words = window_words(bits, start, end);

    assert_eq!(words.len(), (end - start).div_ceil(64), "word count");
    for index in 0..end - start {
        let expected = bit_at(&bytes, offset, start + index);
        let actual = (words[index / 64] >> (index % 64)) & 1 == 1;
        assert_eq!(actual, expected, "bit {index}");
    }

    let ones: u32 = words.iter().map(|word| word.count_ones()).sum();
    let expected = naive_positions(&bytes, offset, start, end, true).len();
    assert_eq!(ones as usize, expected, "set bits");
}

/// An empty window holds nothing, whatever it is asked for.
#[test]
fn select_empty_window() {
    let bytes = mixed_bytes(16);
    let bits = Bits::new(&bytes, 0, 128);
    assert_eq!(select_range(bits, 64, 64, 0), None);
    assert_eq!(select_zero_range(bits, 64, 64, 0), None);
}

/// The counts a window reports must match a bit-at-a-time count, and the first absent rank is the
/// one at that count.
#[test]
fn select_count_agrees_with_naive_count() {
    let bytes = mixed_bytes(300);
    let bits = Bits::new(&bytes, 3, 2000);
    for (start, end) in [(0usize, 2000usize), (5, 1999), (7, 8), (64, 583)] {
        let ones = naive_positions(&bytes, 3, start, end, true).len();
        let zeros = (end - start) - ones;
        if let Some(last) = ones.checked_sub(1) {
            assert!(select_range(bits, start, end, last).is_some());
        }
        assert_eq!(select_range(bits, start, end, ones), None);
        if let Some(last) = zeros.checked_sub(1) {
            assert!(select_zero_range(bits, start, end, last).is_some());
        }
        assert_eq!(select_zero_range(bits, start, end, zeros), None);
    }
}

/// A window has to fit in the bytes backing it, whatever the offset.
#[test]
#[should_panic(expected = "cannot hold")]
fn bits_rejects_a_window_past_the_end() {
    let bytes = mixed_bytes(2);
    let _ = Bits::new(&bytes, 4, 13);
}

// ── The upper array ────────────────────────────────────────────────────

/// `position_of` and `high_of` have to invert each other for every rank and width, since the
/// encoder uses one and all three readers use the other.
#[rstest]
#[case(0, 0, 0)]
#[case(1, 0, 0)]
#[case(0, 7, 3)]
#[case(1_000_000, 4095, 10)]
#[case(u64::MAX, 0, 63)]
#[case(u32::MAX as u64, 1023, 17)]
fn position_and_high_invert(#[case] element: u64, #[case] rank: u64, #[case] lower_width: u8) {
    let position = position_of(element, rank, lower_width);
    assert_eq!(
        high_of(position, rank),
        Some(element >> lower_width),
        "element={element} rank={rank} lower_width={lower_width}"
    );
}

/// A position at or below its own rank is what a corrupt upper buffer looks like, and cannot be
/// inverted.
#[test]
fn high_of_rejects_a_position_below_its_rank() {
    assert_eq!(high_of(0, 0), None);
    assert_eq!(high_of(5, 5), None);
    assert_eq!(high_of(5, 9), None);
    assert_eq!(high_of(6, 5), Some(0));
}

/// Build the upper array for `elements` the way the encoder does.
///
/// Returns the bits, the shared sample buffer, and the array's length in bits.
fn build_upper(elements: &[u64], lower_width: u8) -> (Vec<u8>, Vec<u64>, usize) {
    let n = elements.len();
    let span = elements[n - 1];
    let upper_len = upper_len(span, n, lower_width).expect("representable");
    let mut builder = UpperBuilder::new(upper_len as usize);
    for (index, &element) in elements.iter().enumerate() {
        let rank = index as u64;
        builder.push(rank, position_of(element, rank, lower_width));
    }
    let (bits, samples) = builder.finish(n as u64, upper_len);
    (bits, samples, upper_len as usize)
}

/// A spread wide enough to fill both sample tables: 2000 elements over a span of ~6000 gives
/// `lower_width` 1, five zero-samples and seven one-samples.
fn spread() -> Vec<u64> {
    (0..2000u64).map(|i| i * 3).collect()
}

/// Both tables have to hold exactly what `num_samples0` and `num_samples1` predict, because a
/// reader splits the shared buffer at the seam those two imply rather than at a stored offset.
#[test]
fn upper_builder_fills_both_sample_tables() {
    let elements = spread();
    let (_, samples, _) = build_upper(&elements, 1);

    let span = elements[elements.len() - 1];
    let zeros = num_samples0(span, 1);
    let ones = num_samples1(elements.len());
    assert!(zeros > 0 && ones > 0, "the case must exercise both tables");
    assert_eq!(samples.len() as u64, zeros + ones, "total samples");
}

/// Every element's bit has to be recoverable through the one-sample table, and its high part
/// through `high_of` — the round trip a point lookup makes.
#[test]
fn sampled_select_recovers_every_element() {
    let elements = spread();
    let lower_width = 1u8;
    let (bytes, samples, upper_len) = build_upper(&elements, lower_width);
    let bits = Bits::new(&bytes, 0, upper_len);

    let span = elements[elements.len() - 1];
    let seam = num_samples0(span, lower_width) as usize;
    let samples1: Vec<u8> = samples[seam..]
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();

    for (index, &element) in elements.iter().enumerate() {
        let rank = index as u64;
        let position = sampled_select(bits, &samples1, LOG_SAMPLING1, rank, upper_len, false)
            .unwrap_or_else(|| panic!("no set bit for rank {rank}"));
        assert_eq!(position as u64, position_of(element, rank, lower_width));
        assert_eq!(high_of(position as u64, rank), Some(element >> lower_width));
    }
}

/// The zero-sample table answers the other query: `select0(high) - high` is the number of elements
/// whose high part is below `high`, with no rank directory stored.
#[test]
fn sampled_select_zero_counts_elements_below_a_bucket() {
    let elements = spread();
    let lower_width = 1u8;
    let (bytes, samples, upper_len) = build_upper(&elements, lower_width);
    let bits = Bits::new(&bytes, 0, upper_len);

    let span = elements[elements.len() - 1];
    let seam = num_samples0(span, lower_width) as usize;
    let samples0: Vec<u8> = samples[..seam]
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();

    for high in [0u64, 1, 2, 500, 1000, 2998] {
        let position = sampled_select(bits, &samples0, LOG_SAMPLING0, high, upper_len, true)
            .unwrap_or_else(|| panic!("no bucket boundary for high part {high}"));
        let expected = elements
            .iter()
            .take_while(|&&element| (element >> lower_width) < high)
            .count() as u64;
        assert_eq!(position as u64 - high, expected, "high={high}");
    }
}

/// `Ones` has to walk exactly the positions the builder set, in order.
#[test]
fn ones_walks_every_set_bit() {
    let elements = spread();
    let lower_width = 1u8;
    let (bytes, _, upper_len) = build_upper(&elements, lower_width);

    // The decoder materialises the window as whole words first; do the same by hand.
    let words: Vec<u64> = bytes
        .chunks(8)
        .map(|chunk| {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(buf)
        })
        .collect();

    let mut ones = Ones::new(&words);
    for (index, &element) in elements.iter().enumerate() {
        let rank = index as u64;
        assert_eq!(
            ones.next().map(|p| p as u64),
            Some(position_of(element, rank, lower_width)),
            "rank {rank}"
        );
    }
    assert_eq!(ones.next(), None, "no set bits past the last element");
    assert!(upper_len > 0);
}

// ── The codec, end to end ──────────────────────────────────────────────

/// A low-bits source backed by a plain slice.
///
/// Six lines, and the only thing an embedder has to supply. A host reading its own bit-packed
/// column is not much longer.
struct VecLows(Vec<u64>);

impl LowBits for VecLows {
    type Error = Infallible;

    fn get(&mut self, rank: u64) -> Result<u64, Infallible> {
        Ok(self.0[rank as usize])
    }
}

/// Everything a reader needs, held together so a test can keep the borrows alive.
struct RoundTrip {
    upper: Vec<u8>,
    samples: Vec<u8>,
    seam: usize,
    lows: VecLows,
    lower_width: u8,
    upper_len: usize,
    len: usize,
    span: u64,
}

impl RoundTrip {
    fn encode(elements: &[u64]) -> Self {
        let span = elements[elements.len() - 1];
        let encoded = encode(elements.iter().copied(), span).expect("representable");

        // The two sample tables share one buffer and the seam is derived, never stored.
        let samples: Vec<u8> = encoded
            .samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let seam = num_samples0(span, encoded.lower_width) as usize * size_of::<u64>();

        Self {
            upper: encoded.upper,
            samples,
            seam,
            lows: VecLows(encoded.lower),
            lower_width: encoded.lower_width,
            upper_len: encoded.upper_len as usize,
            len: elements.len(),
            span,
        }
    }

    fn layout(&self) -> Layout<'_> {
        let (_, samples1) = self.samples.split_at(self.seam);
        Layout::new(
            Bits::new(&self.upper, 0, self.upper_len),
            samples1,
            self.lower_width,
            0,
            self.len,
        )
    }
}

fn sparse() -> Vec<u64> {
    (0..200u64).map(|i| i * 977).collect()
}

fn dense() -> Vec<u64> {
    (0..500u64).collect()
}

/// Fifty distinct values spread over a wide universe, each repeated twenty times, so `lower_width`
/// stays positive and every occupied bucket is deep enough to send `next_geq` into its bisection.
fn duplicates() -> Vec<u64> {
    (0..50u64)
        .flat_map(|value| std::iter::repeat_n(value * 4096, 20))
        .collect()
}

/// Encode, then read every element back through the public API alone.
///
/// This is the test the module exists to make possible. If it cannot be written without reaching
/// past that API, this is a bag of helpers rather than a codec.
#[rstest]
#[case::sparse(sparse())]
#[case::dense(dense())]
#[case::duplicates(duplicates())]
#[case::all_equal(vec![7u64; 64])]
#[case::single(vec![0u64])]
#[case::single_wide(vec![u64::MAX])]
fn codec_reads_back_every_element(#[case] elements: Vec<u64>) {
    let round_trip = RoundTrip::encode(&elements);
    let mut lows = VecLows(round_trip.lows.0.clone());
    let layout = round_trip.layout();

    for (index, &expected) in elements.iter().enumerate() {
        assert_eq!(
            element_at(layout, index, &mut lows),
            Ok(expected),
            "index {index}"
        );
    }

    // And once more in reverse. A read holds no state, so the order cannot matter — which is the
    // claim being pinned.
    for (index, &expected) in elements.iter().enumerate().rev() {
        assert_eq!(
            element_at(layout, index, &mut lows),
            Ok(expected),
            "reverse index {index}"
        );
    }
}

/// What the encoder writes, the validator must accept.
#[rstest]
#[case::sparse(sparse())]
#[case::dense(dense())]
#[case::duplicates(duplicates())]
#[case::single_wide(vec![u64::MAX])]
fn encoder_output_validates(#[case] elements: Vec<u64>) {
    let round_trip = RoundTrip::encode(&elements);
    assert_eq!(
        validate_layout(
            round_trip.span,
            round_trip.len,
            round_trip.lower_width,
            round_trip.upper_len as u64,
            &round_trip.samples,
        ),
        Ok(())
    );
}

/// The bulk decoder has to reproduce the sequence from the same buffers the reader walks.
#[rstest]
#[case::sparse(sparse())]
#[case::dense(dense())]
#[case::duplicates(duplicates())]
fn decoder_reproduces_the_sequence(#[case] elements: Vec<u64>) {
    let round_trip = RoundTrip::encode(&elements);
    let (_, samples1) = round_trip.samples.split_at(round_trip.seam);
    let bits = Bits::new(&round_trip.upper, 0, round_trip.upper_len);
    let last = elements.len() as u64 - 1;

    // Trim to the window holding exactly these elements' set bits, which is what a caller does and
    // what lets the walk below run without a per-element bound check.
    let start = position_of_rank(bits, samples1, 0).expect("first element");
    let end = position_of_rank(bits, samples1, last).expect("last element") + 1;
    let words = window_words(bits, start, end);

    let mut decoder = Decoder::new(&words, start, 0, elements.len(), round_trip.lower_width)
        .expect("well-formed");
    let mut decoded = vec![0u64; elements.len()];
    decoder.segment(Some(&round_trip.lows.0), |index, element| {
        decoded[index] = element;
    });
    assert_eq!(decoder.finish(), Ok(()));
    assert_eq!(decoded, elements);
}
