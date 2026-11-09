// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use hegel::TestCase;
use hegel::generators as gs;
use rstest::rstest;
use vortex_buffer::BitBuffer;

use crate::AllOr;
use crate::Mask;
use crate::MaskIter;

const MAX_LEN: usize = 200;

fn draw_len(tc: &TestCase) -> usize {
    tc.draw(gs::integers::<usize>().max_value(MAX_LEN))
}

fn draw_bits(tc: &TestCase, len: usize) -> Vec<bool> {
    match tc.draw(gs::integers::<u8>().max_value(2)) {
        0 => vec![true; len],
        1 => vec![false; len],
        _ => tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len)),
    }
}

fn build_mask(tc: &TestCase, bits: &[bool]) -> Mask {
    let len = bits.len();
    match tc.draw(gs::integers::<u8>().max_value(4)) {
        0 if bits.iter().all(|&b| b) => Mask::new_true(len),
        0 if !bits.iter().any(|&b| b) => Mask::new_false(len),
        0 | 1 => Mask::from_iter(bits.iter().copied()),
        2 => Mask::from_indices(len, model_indices(bits)),
        3 => Mask::from_slices(len, model_slices(bits)),
        _ => {
            let offset = tc.draw(gs::integers::<usize>().max_value(63));
            let mut source = vec![true; offset];
            source.extend_from_slice(bits);
            source.extend([false, true, false]);
            Mask::from_buffer(BitBuffer::from(source).slice(offset..offset + len))
        }
    }
}

fn draw_mask_of_len(tc: &TestCase, len: usize) -> (Mask, Vec<bool>) {
    let bits = draw_bits(tc, len);
    (build_mask(tc, &bits), bits)
}

fn draw_mask(tc: &TestCase) -> (Mask, Vec<bool>) {
    let len = draw_len(tc);
    draw_mask_of_len(tc, len)
}

fn model_indices(bits: &[bool]) -> Vec<usize> {
    bits.iter()
        .enumerate()
        .filter_map(|(i, &b)| b.then_some(i))
        .collect()
}

fn model_slices(bits: &[bool]) -> Vec<(usize, usize)> {
    let mut slices: Vec<(usize, usize)> = Vec::new();
    for i in model_indices(bits) {
        match slices.last_mut() {
            Some((_, end)) if *end == i => *end += 1,
            _ => slices.push((i, i + 1)),
        }
    }
    slices
}

fn assert_matches_model(mask: &Mask, bits: &[bool]) {
    let indices = model_indices(bits);
    let slices = model_slices(bits);
    assert_counts_match(mask, bits.len(), &indices);
    assert_elements_match(mask, bits, &indices);
    assert_representations_match(mask, bits, &indices, &slices);
    assert_values_match(mask, bits, &indices, &slices);
}

fn assert_counts_match(mask: &Mask, len: usize, indices: &[usize]) {
    assert_eq!(mask.len(), len);
    assert_eq!(mask.is_empty(), len == 0);
    assert_eq!(mask.true_count(), indices.len());
    assert_eq!(mask.false_count(), len - indices.len());
    assert_eq!(mask.all_true(), indices.len() == len);
    assert_eq!(mask.all_false(), indices.is_empty());
    if len > 0 {
        #[expect(clippy::cast_precision_loss)]
        let density = indices.len() as f64 / len as f64;
        assert!((mask.density() - density).abs() < 1e-10);
    }
}

fn assert_elements_match(mask: &Mask, bits: &[bool], indices: &[usize]) {
    let len = bits.len();
    assert_eq!(mask.iter().collect::<Vec<_>>(), bits);
    assert_eq!(mask.iter().len(), len);
    assert_eq!((0..len).map(|i| mask.value(i)).collect::<Vec<_>>(), bits);
    assert_eq!(mask.first(), indices.first().copied());
    assert_eq!(mask.last(), indices.last().copied());
    for (n, &idx) in indices.iter().enumerate() {
        assert_eq!(mask.rank(n), idx);
    }
}

fn assert_representations_match(
    mask: &Mask,
    bits: &[bool],
    indices: &[usize],
    slices: &[(usize, usize)],
) {
    let len = bits.len();
    let mask_indices = match mask.indices() {
        AllOr::All => (0..len).collect(),
        AllOr::None => Vec::new(),
        AllOr::Some(indices) => indices.to_vec(),
    };
    assert_eq!(mask_indices, indices);

    let mask_slices = match mask.slices() {
        AllOr::All if len > 0 => vec![(0, len)],
        AllOr::All | AllOr::None => Vec::new(),
        AllOr::Some(slices) => slices.to_vec(),
    };
    assert_eq!(mask_slices, slices);

    let buffer_bits = match mask.bit_buffer() {
        AllOr::All => vec![true; len],
        AllOr::None => vec![false; len],
        AllOr::Some(buffer) => buffer.iter().collect(),
    };
    assert_eq!(buffer_bits, bits);
    assert_eq!(mask.to_bit_buffer().iter().collect::<Vec<_>>(), bits);
}

fn assert_values_match(mask: &Mask, bits: &[bool], indices: &[usize], slices: &[(usize, usize)]) {
    let len = bits.len();
    let Some(values) = mask.values() else {
        return;
    };
    assert!(!values.is_empty());
    assert_eq!(values.len(), len);
    assert_eq!(values.true_count(), indices.len());
    assert_eq!(values.indices(), indices);
    assert_eq!(values.slices(), slices);
    assert_eq!(values.bit_buffer().iter().collect::<Vec<_>>(), bits);
    assert_eq!((0..len).map(|i| values.value(i)).collect::<Vec<_>>(), bits);
}

#[hegel::test]
fn mask(tc: TestCase) {
    let (mask, bits) = draw_mask(&tc);
    assert_matches_model(&mask, &bits);
}

#[hegel::test]
fn mask_eq(tc: TestCase) {
    let (lhs, lhs_bits) = draw_mask(&tc);
    let rhs_bits = if tc.draw(gs::booleans()) {
        lhs_bits.clone()
    } else {
        let len = if tc.draw(gs::booleans()) {
            lhs_bits.len()
        } else {
            draw_len(&tc)
        };
        draw_bits(&tc, len)
    };
    let rhs = build_mask(&tc, &rhs_bits);

    assert_eq!(lhs == rhs, lhs_bits == rhs_bits);
    assert_eq!(rhs == lhs, lhs_bits == rhs_bits);
}

#[hegel::test]
fn mask_slice(tc: TestCase) {
    let (mask, bits) = draw_mask(&tc);
    let end = tc.draw(gs::integers::<usize>().max_value(bits.len()));
    let start = tc.draw(gs::integers::<usize>().max_value(end));
    assert_matches_model(&mask.slice(start..end), &bits[start..end]);
}

#[hegel::test]
fn mask_limit(tc: TestCase) {
    let (mask, bits) = draw_mask(&tc);
    let limit = tc.draw(gs::integers::<usize>().max_value(bits.len() + 2));
    let mut remaining = limit;
    let expected: Vec<bool> = bits
        .iter()
        .map(|&b| {
            let keep = b && remaining > 0;
            if keep {
                remaining -= 1;
            }
            keep
        })
        .collect();
    assert_matches_model(&mask.limit(limit), &expected);
}

#[hegel::test]
fn mask_bitops(tc: TestCase) {
    let len = draw_len(&tc);
    let (lhs, lhs_bits) = draw_mask_of_len(&tc, len);
    let (rhs, rhs_bits) = draw_mask_of_len(&tc, len);
    let zip = |f: fn(bool, bool) -> bool| -> Vec<bool> {
        lhs_bits
            .iter()
            .zip(&rhs_bits)
            .map(|(&l, &r)| f(l, r))
            .collect()
    };

    let and = zip(|l, r| l & r);
    let or = zip(|l, r| l | r);
    let and_not = zip(|l, r| l & !r);
    let not: Vec<bool> = lhs_bits.iter().map(|&b| !b).collect();

    assert_matches_model(&(&lhs & &rhs), &and);
    assert_matches_model(&(lhs.clone() & &rhs), &and);
    assert_matches_model(&(&lhs | &rhs), &or);
    assert_matches_model(&(lhs.clone() | &rhs), &or);
    assert_matches_model(&lhs.clone().bitand_not(&rhs), &and_not);
    assert_matches_model(&!&lhs, &not);
    assert_matches_model(&!lhs, &not);
}

#[hegel::test]
fn mask_concat(tc: TestCase) {
    let count = tc.draw(gs::integers::<usize>().max_value(4));
    let (masks, bits): (Vec<Mask>, Vec<Vec<bool>>) = (0..count)
        .map(|_| {
            let len = tc.draw(gs::integers::<usize>().max_value(MAX_LEN / 4));
            draw_mask_of_len(&tc, len)
        })
        .unzip();
    let expected = bits.concat();

    assert_matches_model(&Mask::concat(masks.iter()).unwrap(), &expected);
    assert_matches_model(&Mask::from_iter(masks), &expected);
}

#[hegel::test]
fn mask_from_excluded(tc: TestCase) {
    let len = draw_len(&tc);
    let bits = draw_bits(&tc, len);
    let mut excluded = Vec::new();
    for (i, _) in bits.iter().enumerate().filter(|&(_, &b)| !b) {
        let copies = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        for _ in 0..copies {
            let pos = tc.draw(gs::integers::<usize>().max_value(excluded.len()));
            excluded.insert(pos, i);
        }
    }
    assert_matches_model(&Mask::from_excluded_indices(len, excluded), &bits);
}

#[hegel::test]
fn mask_from_intersection(tc: TestCase) {
    let len = draw_len(&tc);
    let lhs = draw_bits(&tc, len);
    let rhs = draw_bits(&tc, len);
    let expected: Vec<bool> = lhs.iter().zip(&rhs).map(|(&l, &r)| l && r).collect();
    let mask = Mask::from_intersection_indices(
        len,
        model_indices(&lhs).into_iter(),
        model_indices(&rhs).into_iter(),
    );
    assert_matches_model(&mask, &expected);
}

#[hegel::test]
fn mask_valid_counts(tc: TestCase) {
    let (mask, bits) = draw_mask(&tc);
    let mut indices: Vec<usize> =
        tc.draw(gs::vecs(gs::integers::<usize>().max_value(bits.len())).max_size(MAX_LEN));
    indices.sort_unstable();
    let expected: Vec<usize> = indices
        .iter()
        .map(|&idx| bits[..idx].iter().filter(|&&b| b).count())
        .collect();
    assert_eq!(mask.valid_counts_for_indices(&indices), expected);
}

#[hegel::test]
fn mask_intersect_by_rank(tc: TestCase) {
    let (mask, bits) = draw_mask(&tc);
    let (rank, rank_bits) = draw_mask_of_len(&tc, mask.true_count());
    let mut ranks = rank_bits.iter();
    let expected: Vec<bool> = bits
        .iter()
        .map(|&b| b && *ranks.next().unwrap_or(&false))
        .collect();
    assert_matches_model(&mask.intersect_by_rank(&rank), &expected);
}

#[rstest]
#[case::all_true(Mask::new_true(4), vec![true, true, true, true])]
#[case::all_false(Mask::new_false(3), vec![false, false, false])]
#[case::values(
    Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false])),
    vec![true, false, true, true, false]
)]
#[case::empty(Mask::new_true(0), vec![])]
fn mask_iter_matches_value(#[case] mask: Mask, #[case] expected: Vec<bool>) {
    let collected: Vec<bool> = mask.iter().collect();
    assert_eq!(collected, expected);

    let by_value: Vec<bool> = (0..mask.len()).map(|i| mask.value(i)).collect();
    assert_eq!(collected, by_value);

    let mut it = mask.iter();
    assert_eq!(it.len(), mask.len());
    if it.next().is_some() {
        assert_eq!(it.len(), mask.len() - 1);
    }
}

#[test]
#[should_panic]
fn test_mask_slice_out_of_bounds() {
    let mask = Mask::new_true(5);
    mask.slice(3..8); // offset + length > len
}

#[test]
#[should_panic(expected = "Masks must have the same length")]
fn test_bitand_different_lengths() {
    let mask1 = Mask::new_true(5);
    let mask2 = Mask::new_true(3);
    let _unused = &mask1 & &mask2;
}

#[test]
fn test_mask_values_threshold_iter() {
    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false]));
    let values = mask.values().unwrap();

    // With low threshold, should prefer indices
    match values.threshold_iter(0.7) {
        MaskIter::Indices(indices) => {
            assert_eq!(indices, &[0, 2, 3]);
        }
        _ => panic!("Expected indices iterator"),
    }

    // With high threshold, should prefer slices
    match values.threshold_iter(0.5) {
        MaskIter::Slices(slices) => {
            assert_eq!(slices, &[(0, 1), (2, 4)]);
        }
        _ => panic!("Expected slices iterator"),
    }
}

#[test]
fn test_mask_values_cached_representations() {
    let from_buffer = Mask::from_buffer(BitBuffer::from_iter([true, false, true]));
    let values = from_buffer.values().unwrap();
    assert!(values.cached_indices().is_none());
    assert!(values.cached_slices().is_none());

    let from_indices = Mask::from_indices(5, [1, 3]);
    let values = from_indices.values().unwrap();
    assert_eq!(values.cached_indices(), Some([1, 3].as_slice()));
    assert!(values.cached_slices().is_none());

    let from_slices = Mask::from_slices(6, vec![(1, 3), (5, 6)]);
    let values = from_slices.values().unwrap();
    assert!(values.cached_indices().is_none());
    assert_eq!(values.cached_slices(), Some([(1, 3), (5, 6)].as_slice()));
}

#[rstest]
#[case::adjacent_runs(vec![(1, 3), (3, 4), (6, 8)])]
#[case::all_false(vec![])]
#[case::all_true(vec![(0, 8)])]
fn mask_from_buffer_with_slices(#[case] slices: Vec<(usize, usize)>) {
    let buffer =
        BitBuffer::from_iter((0..8).map(|i| slices.iter().any(|&(s, e)| (s..e).contains(&i))));
    let mask = Mask::from_buffer_with_slices(buffer.clone(), slices.clone());
    assert_eq!(mask, Mask::from_buffer(buffer));
    if let Some(values) = mask.values() {
        assert_eq!(values.cached_slices(), Some(slices.as_slice()));
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic]
fn mask_from_buffer_with_slices_missing_set_bit() {
    Mask::from_buffer_with_slices(
        BitBuffer::from_iter([true, true, false, true]),
        vec![(0, 2)],
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic]
fn mask_from_buffer_with_slices_unset_bit() {
    Mask::from_buffer_with_slices(
        BitBuffer::from_iter([true, false, false, true]),
        vec![(0, 2)],
    );
}

#[test]
#[should_panic]
fn test_mask_valid_counts_for_indices_error() {
    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true]));
    let indices = vec![0, 2, 5]; // 5 is out of bounds
    mask.valid_counts_for_indices(&indices);
}

#[test]
#[should_panic]
fn test_mask_from_indices_unsorted() {
    Mask::from_indices(5, vec![2, 0, 3]); // Not sorted
}

#[test]
#[should_panic]
fn test_mask_from_indices_duplicate() {
    Mask::from_indices(5, vec![0, 2, 2]); // Not unique
}

#[test]
#[should_panic]
fn test_mask_from_indices_out_of_bounds() {
    Mask::from_indices(5, vec![0, 2, 5]); // 5 is out of bounds
}

#[test]
#[should_panic]
fn test_mask_from_slices_invalid_range() {
    Mask::from_slices(5, vec![(2, 2)]); // Invalid range where start == end
}

#[test]
#[should_panic]
fn test_mask_from_slices_out_of_bounds() {
    Mask::from_slices(5, vec![(0, 6)]); // end > len
}

#[test]
#[should_panic]
fn test_mask_from_slices_unsorted() {
    Mask::from_slices(5, vec![(2, 3), (0, 1)]); // Not sorted
}

#[test]
#[should_panic]
fn test_mask_from_slices_overlapping() {
    Mask::from_slices(5, vec![(0, 3), (2, 4)]); // Overlapping ranges
}

#[test]
fn test_mask_threshold_iter() {
    let all_true = Mask::new_true(5);
    assert!(matches!(all_true.threshold_iter(0.5), AllOr::All));

    let all_false = Mask::new_false(5);
    assert!(matches!(all_false.threshold_iter(0.5), AllOr::None));

    let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false]));
    if let AllOr::Some(MaskIter::Indices(indices)) = mask.threshold_iter(0.7) {
        assert_eq!(indices, &[0, 2, 3]);
    } else {
        panic!("Expected indices iterator");
    }
}

#[test]
fn test_mask_indices_caching() {
    // Test that indices are properly cached
    let mask = Mask::from_slices(10, vec![(0, 3), (5, 7), (9, 10)]);

    // First call should compute indices
    let indices1 = mask.indices();
    // Second call should return cached value
    let indices2 = mask.indices();

    match (indices1, indices2) {
        (AllOr::Some(i1), AllOr::Some(i2)) => {
            assert_eq!(i1, i2);
            assert_eq!(i1, &[0, 1, 2, 5, 6, 9]);
            // Verify they're the same reference (cached)
            assert!(std::ptr::eq(i1, i2));
        }
        _ => panic!("Expected Some variant"),
    }
}

#[test]
fn test_mask_slices_caching() {
    // Test that slices are properly cached
    let mask = Mask::from_indices(10, vec![0, 1, 2, 5, 6, 9]);

    // First call should compute slices
    let slices1 = mask.slices();
    // Second call should return cached value
    let slices2 = mask.slices();

    match (slices1, slices2) {
        (AllOr::Some(s1), AllOr::Some(s2)) => {
            assert_eq!(s1, s2);
            assert_eq!(s1, &[(0, 3), (5, 7), (9, 10)]);
            // Verify they're the same reference (cached)
            assert!(std::ptr::eq(s1, s2));
        }
        _ => panic!("Expected Some variant"),
    }
}

#[test]
fn test_allor_unwrap_or_else() {
    let all: AllOr<i32> = AllOr::All;
    assert_eq!(all.unwrap_or_else(|| 42, || 0), 42);

    let none: AllOr<i32> = AllOr::None;
    assert_eq!(none.unwrap_or_else(|| 42, || 0), 0);

    let some: AllOr<i32> = AllOr::Some(10);
    assert_eq!(some.unwrap_or_else(|| 42, || 0), 10);
}

#[test]
fn test_allor_cloned() {
    let original = vec![1, 2, 3];
    let all: AllOr<&Vec<i32>> = AllOr::All;
    assert!(matches!(all.cloned(), AllOr::All));

    let none: AllOr<&Vec<i32>> = AllOr::None;
    assert!(matches!(none.cloned(), AllOr::None));

    let some: AllOr<&Vec<i32>> = AllOr::Some(&original);
    if let AllOr::Some(cloned) = some.cloned() {
        assert_eq!(cloned, original);
    } else {
        panic!("Expected Some variant");
    }
}

#[test]
fn test_allor_eq() {
    let all1: AllOr<i32> = AllOr::All;
    let all2: AllOr<i32> = AllOr::All;
    assert_eq!(all1, all2);

    let none1: AllOr<i32> = AllOr::None;
    let none2: AllOr<i32> = AllOr::None;
    assert_eq!(none1, none2);

    let some1: AllOr<i32> = AllOr::Some(42);
    let some2: AllOr<i32> = AllOr::Some(42);
    let some3: AllOr<i32> = AllOr::Some(43);
    assert_eq!(some1, some2);
    assert_ne!(some1, some3);

    assert_ne!(all1, none1);
    assert_ne!(all1, some1);
    assert_ne!(none1, some1);
}
