// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ExecutionCtx;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::scalar::Scalar;

pub(super) fn accumulate_varbinview(
    partial: &mut MinMaxPartial,
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    partial.merge(compute_min_max_with_validity(array, ctx)?);
    Ok(())
}

fn compute_min_max_with_validity(
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<MinMaxResult>> {
    let views = array.views();
    let buffers = (0..array.data_buffers().len())
        .map(|idx| array.buffer(idx).as_slice())
        .collect::<Vec<_>>();

    let mut extrema = None;
    match array.validity()?.execute_mask(array.len(), ctx)? {
        Mask::AllTrue(_) => extrema = scan(extrema, views, &buffers),
        Mask::AllFalse(_) => {}
        // Each `[start, end)` run is fully valid, so the runs chain through a single scan
        // state instead of materializing the validity bits and testing them per element.
        Mask::Values(v) => {
            for &(start, end) in v.slices() {
                extrema = scan(extrema, &views[start..end], &buffers);
            }
        }
    }

    let dtype = array.dtype();
    Ok(extrema.map(|extrema| MinMaxResult {
        min: make_scalar(dtype, extrema.min_bytes),
        max: make_scalar(dtype, extrema.max_bytes),
    }))
}

/// The running extrema of a scan: the smallest and largest values seen so far, each held as its
/// [ordering prefix](order_prefix) alongside the bytes the prefix was taken from.
struct Extrema<'a> {
    min_prefix: u32,
    min_bytes: &'a [u8],
    max_prefix: u32,
    max_bytes: &'a [u8],
}

/// Fold the fully-valid `views` into `extrema`, seeding it from the first view if it is empty.
///
/// Every view stores the first four bytes of its value inline, and comparing those four bytes as
/// a big-endian integer agrees with the lexicographic order of the values themselves (see
/// [`order_prefix`]). A value whose prefix falls strictly inside the running `[min, max]`
/// interval is therefore settled by its 16-byte view alone, without reading the value — which
/// for a value longer than twelve bytes means a read into a data buffer.
///
/// Views are folded two at a time. Ordering a pair against itself first means only its smaller
/// value can lower the minimum and only its larger value can raise the maximum, so a pair that
/// has to be resolved costs three value comparisons instead of four. That is what keeps the scan
/// competitive on columns whose values all share their first four bytes, where the prefix never
/// rejects anything.
fn scan<'a>(
    extrema: Option<Extrema<'a>>,
    views: &'a [BinaryView],
    buffers: &[&'a [u8]],
) -> Option<Extrema<'a>> {
    let (seed, rest) = match extrema {
        Some(extrema) => (extrema, views),
        None => {
            let (first, rest) = views.split_first()?;
            let prefix = order_prefix(first);
            let bytes = first.bytes(buffers);
            let seed = Extrema {
                min_prefix: prefix,
                min_bytes: bytes,
                max_prefix: prefix,
                max_bytes: bytes,
            };
            (seed, rest)
        }
    };
    let Extrema {
        mut min_prefix,
        mut min_bytes,
        mut max_prefix,
        mut max_bytes,
    } = seed;

    let (pairs, remainder) = rest.as_chunks::<2>();
    for [a, b] in pairs {
        let (a_prefix, b_prefix) = (order_prefix(a), order_prefix(b));
        if a_prefix > min_prefix
            && a_prefix < max_prefix
            && b_prefix > min_prefix
            && b_prefix < max_prefix
        {
            continue;
        }

        let (a_bytes, b_bytes) = (a.bytes(buffers), b.bytes(buffers));
        let ((lo_prefix, lo), (hi_prefix, hi)) =
            if a_prefix < b_prefix || (a_prefix == b_prefix && a_bytes <= b_bytes) {
                ((a_prefix, a_bytes), (b_prefix, b_bytes))
            } else {
                ((b_prefix, b_bytes), (a_prefix, a_bytes))
            };
        if lo_prefix < min_prefix || (lo_prefix == min_prefix && lo < min_bytes) {
            min_prefix = lo_prefix;
            min_bytes = lo;
        }
        if hi_prefix > max_prefix || (hi_prefix == max_prefix && hi > max_bytes) {
            max_prefix = hi_prefix;
            max_bytes = hi;
        }
    }

    // An odd view is folded on its own. `min_bytes <= max_bytes` holds by construction, so a
    // value that lowers the minimum cannot also raise the maximum.
    for view in remainder {
        let prefix = order_prefix(view);
        if prefix > min_prefix && prefix < max_prefix {
            continue;
        }
        let bytes = view.bytes(buffers);
        if prefix < min_prefix || (prefix == min_prefix && bytes < min_bytes) {
            min_prefix = prefix;
            min_bytes = bytes;
        } else if prefix > max_prefix || (prefix == max_prefix && bytes > max_bytes) {
            max_prefix = prefix;
            max_bytes = bytes;
        }
    }

    Some(Extrema {
        min_prefix,
        min_bytes,
        max_prefix,
        max_bytes,
    })
}

/// The first four bytes of a view's value as a big-endian `u32`, so that comparing two prefixes
/// agrees with the lexicographic order of the values they came from.
///
/// Both view variants store those bytes at the same offset — the inlined value for short values,
/// the `prefix` field for values held in a data buffer — zero-padded for values shorter than four
/// bytes. Zero padding is what makes the comparison an order refinement rather than an
/// approximation: if `a`'s prefix is below `b`'s then either they first differ at a byte both
/// values have, or `a` ran out of bytes first and is a proper prefix of `b`; either way `a < b`.
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "intentional bit slicing")]
fn order_prefix(view: &BinaryView) -> u32 {
    // Bits 32..64 hold the prefix bytes in value order, which `swap_bytes` turns into the
    // big-endian integer whose comparison matches a byte-wise comparison of those bytes.
    ((view.as_u128() >> 32) as u32).swap_bytes()
}

fn make_scalar(dtype: &DType, value: &[u8]) -> Scalar {
    match dtype {
        DType::Binary(_) => Scalar::binary(value.to_vec(), NonNullable),
        DType::Utf8(_) => {
            // SAFETY: a `VarBinViewArray` with a Utf8 dtype validates every view's bytes as
            // UTF-8 on construction.
            let value = unsafe { str::from_utf8_unchecked(value) };
            Scalar::utf8(value, NonNullable)
        }
        _ => vortex_panic!("cannot make Scalar from bytes with dtype {dtype}"),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::min_max::min_max;
    use crate::arrays::VarBinViewArray;
    use crate::dtype::Nullability;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    fn utf8_min_max(values: &[Option<&str>]) -> VortexResult<Option<(String, String)>> {
        let array =
            VarBinViewArray::from_iter(values.iter().copied(), DType::Utf8(Nullability::Nullable))
                .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        Ok(
            min_max(&array, &mut ctx, NumericalAggregateOpts::default())?.map(|result| {
                (
                    result.min.as_utf8().value().unwrap().to_string(),
                    result.max.as_utf8().value().unwrap().to_string(),
                )
            }),
        )
    }

    // Values are compared by their inline four-byte prefix first, so the cases that matter are
    // the ones where that prefix cannot decide the order.
    #[rstest]
    // Prefixes tie and the order is settled past the inline prefix, on both sides of the
    // twelve-byte boundary between inlined and buffer-backed values.
    #[case::tied_prefix_inlined(&[Some("abcdZ"), Some("abcdA"), Some("abcdM")], "abcdA", "abcdZ")]
    #[case::tied_prefix_outlined(
        &[Some("abcd_long_value_zzz"), Some("abcd_long_value_aaa")],
        "abcd_long_value_aaa",
        "abcd_long_value_zzz"
    )]
    // One value is a proper prefix of another, so the shorter one sorts first even though its
    // inline prefix is zero-padded to the same four bytes.
    #[case::proper_prefix(&[Some("abcd"), Some("abc"), Some("abcde")], "abc", "abcde")]
    #[case::empty_value(&[Some("a"), Some(""), Some("b")], "", "b")]
    // Values shorter than the four-byte prefix, mixed with longer ones sharing their bytes.
    #[case::short_values(&[Some("b"), Some("ab"), Some("abc"), Some("a")], "a", "b")]
    // A single inlined value and a single buffer-backed value are both min and max.
    #[case::single_inlined(&[Some("xy")], "xy", "xy")]
    #[case::single_outlined(&[Some("a value longer than twelve bytes")],
        "a value longer than twelve bytes", "a value longer than twelve bytes")]
    // Extrema fall in different valid runs, and nulls never contribute.
    #[case::extrema_in_separate_runs(
        &[Some("m"), None, None, Some("z"), None, Some("a"), Some("q")],
        "a",
        "z"
    )]
    #[case::nulls_around_extrema(&[None, Some("b"), None], "b", "b")]
    fn utf8_extrema(
        #[case] values: &[Option<&str>],
        #[case] min: &str,
        #[case] max: &str,
    ) -> VortexResult<()> {
        assert_eq!(
            utf8_min_max(values)?,
            Some((min.to_string(), max.to_string()))
        );
        Ok(())
    }

    #[rstest]
    #[case::empty(&[])]
    #[case::all_null(&[None, None])]
    fn utf8_no_extrema(#[case] values: &[Option<&str>]) -> VortexResult<()> {
        assert_eq!(utf8_min_max(values)?, None);
        Ok(())
    }

    /// Random values over a tiny alphabet, checked against a sorted reference. Short values and
    /// a three-symbol alphabet make ties on the inline prefix and values that are prefixes of
    /// one another common, and the varying lengths cover the pairwise fold's odd tail.
    #[test]
    fn matches_sorted_reference() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(0);
        let mut ctx = SESSION.create_execution_ctx();
        for _ in 0..2000 {
            let values = (0..rng.random_range(0..24))
                .map(|_| {
                    (!rng.random_bool(0.25)).then(|| {
                        (0..rng.random_range(0..20))
                            .map(|_| b"ab\x00"[rng.random_range(0..3)])
                            .collect::<Vec<u8>>()
                    })
                })
                .collect::<Vec<_>>();

            let array = VarBinViewArray::from_iter(
                values.iter().map(|v| v.as_deref()),
                DType::Binary(Nullability::Nullable),
            )
            .into_array();
            let result = min_max(&array, &mut ctx, NumericalAggregateOpts::default())?;

            let mut valid = values.iter().flatten().collect::<Vec<_>>();
            valid.sort();
            let expected = valid.first().map(|min| MinMaxResult {
                min: Scalar::binary((*min).clone(), NonNullable),
                max: Scalar::binary(valid[valid.len() - 1].clone(), NonNullable),
            });
            assert_eq!(result, expected, "mismatch for {values:?}");
        }
        Ok(())
    }

    /// Binary values order by unsigned byte value, including bytes that are not valid UTF-8 and
    /// interior nulls that the inline prefix cannot distinguish from padding.
    #[test]
    fn binary_extrema_order_by_byte_value() -> VortexResult<()> {
        let values: Vec<&[u8]> = vec![
            b"\x00\x00\x00\x00\xff",
            b"\x00",
            b"\x80\x01",
            b"\xff\xff\xff\xff\x00",
            b"\x00\x00\x00\x00",
        ];
        let array = VarBinViewArray::from_iter(
            values.iter().map(|v| Some(*v)),
            DType::Binary(Nullability::Nullable),
        )
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let result = min_max(&array, &mut ctx, NumericalAggregateOpts::default())?
            .vortex_expect("non-empty array has extrema");

        assert_eq!(result.min, Scalar::binary(b"\x00".to_vec(), NonNullable));
        assert_eq!(
            result.max,
            Scalar::binary(b"\xff\xff\xff\xff\x00".to_vec(), NonNullable)
        );
        Ok(())
    }
}
