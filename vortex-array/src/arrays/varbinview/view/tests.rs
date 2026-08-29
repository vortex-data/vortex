// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::*;

#[rstest::rstest]
// Just past the inline boundary, typical, and large values.
#[case(13, 7, 42)]
#[case(20, 7, 42)]
#[case(255, 7, 42)]
#[case(4096, 7, 42)]
// Zero buffer index/offset and the `u32` extremes, to confirm the `u128` field assembly does
// not overflow into neighbouring fields.
#[case(13, 0, 0)]
#[case(13, u32::MAX, u32::MAX)]
fn new_ref_matches_make_view(#[case] len: u32, #[case] buffer_index: u32, #[case] offset: u32) {
    // `new_ref` assembles the reference view as a `u128`; it must be byte-identical to the
    // value-inspecting `make_view` for any value longer than the inline limit.
    let value: Vec<u8> = (0..len)
        .map(|i| u8::try_from(i % 251).vortex_expect("i % 251 fits in u8"))
        .collect();
    let prefix = [value[0], value[1], value[2], value[3]];
    let made = BinaryView::make_view(&value, buffer_index, offset);
    let built = BinaryView::new_ref(len, prefix, buffer_index, offset);
    assert_eq!(made.as_u128(), built.as_u128(), "mismatch at len {len}");
    assert!(!built.is_inlined());
    let r = built.as_view();
    assert_eq!(r.size, len);
    assert_eq!(r.prefix, prefix);
    assert_eq!(r.buffer_index, buffer_index);
    assert_eq!(r.offset, offset);
}

/// Byte strings over a three-symbol alphabet, up to `max_len`. `\x00` is included so that a
/// value's real bytes collide with the zero padding of a shorter one.
fn values(max_len: usize) -> Vec<Vec<u8>> {
    let mut values = vec![Vec::new()];
    let mut frontier = vec![Vec::new()];
    for _ in 0..max_len {
        frontier = frontier
            .iter()
            .flat_map(|value| {
                b"\x00a\xff".iter().map(|&byte| {
                    let mut next = value.clone();
                    next.push(byte);
                    next
                })
            })
            .collect();
        values.extend(frontier.iter().cloned());
    }
    values
}

/// A view holding `value`, inlined when it fits and otherwise referencing buffer 0 at offset 0.
fn view_of(value: &[u8]) -> BinaryView {
    BinaryView::make_view(value, 0, 0)
}

/// The comparison words a view carries must agree with the ones computed from its value; the
/// `_of` constructors exist so a scalar can be compared against views without building one.
#[test]
fn words_match_between_view_and_value() {
    for value in values(5).iter().chain(&[vec![b'z'; 40]]) {
        let view = view_of(value);
        assert_eq!(view.prefix(), BinaryView::prefix_of(value), "{value:?}");
        assert_eq!(view.head(), BinaryView::head_of(value), "{value:?}");
        assert_eq!(
            view.order_prefix(),
            BinaryView::order_prefix_of(value),
            "{value:?}"
        );
        if view.is_inlined() {
            assert_eq!(
                view.order_tail(),
                BinaryView::order_tail_of(value),
                "{value:?}"
            );
        }
    }
}

/// The order prefix refines the value order: a strictly lower prefix proves a strictly lower
/// value, and equal values always agree on it. Equal prefixes decide nothing, which is why every
/// caller falls through to a further comparison.
#[test]
fn order_prefix_refines_value_order() {
    let values = values(6);
    for a in &values {
        for b in &values {
            let (a_prefix, b_prefix) = (
                BinaryView::order_prefix_of(a),
                BinaryView::order_prefix_of(b),
            );
            if a_prefix < b_prefix {
                assert!(a < b, "prefix ordered {a:?} below {b:?}");
            }
            if a == b {
                assert_eq!(a_prefix, b_prefix, "{a:?} disagreed with itself");
            }
        }
    }
}

/// For values short enough to inline, the order prefix, order tail and length together decide the
/// order outright — no data buffer, and no comparison of the values themselves.
#[test]
fn order_words_fully_order_inlined_values() {
    let values = values(4)
        .into_iter()
        .filter(|value| value.len() <= BinaryView::MAX_INLINED_SIZE)
        .collect::<Vec<_>>();
    for a in &values {
        let a_view = view_of(a);
        for b in &values {
            let b_view = view_of(b);
            let by_words = (a_view.order_prefix(), a_view.order_tail(), a_view.len()).cmp(&(
                b_view.order_prefix(),
                b_view.order_tail(),
                b_view.len(),
            ));
            assert_eq!(by_words, a.cmp(b), "words misordered {a:?} against {b:?}");
        }
    }
}

/// The head rules equality out but never in: values that share a length and a four-byte prefix
/// collide, which is why callers still compare the bytes past the prefix.
#[test]
fn head_rules_out_inequality() {
    let values = values(6);
    for a in &values {
        for b in &values {
            if a == b {
                assert_eq!(BinaryView::head_of(a), BinaryView::head_of(b));
            } else if BinaryView::head_of(a) == BinaryView::head_of(b) {
                assert_eq!(a.len(), b.len(), "head collided across lengths");
                assert_eq!(a[..4.min(a.len())], b[..4.min(b.len())]);
            }
        }
    }
}
