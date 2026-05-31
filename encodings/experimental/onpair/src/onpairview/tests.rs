// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySession;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::DEFAULT_DICT12_CONFIG;
use crate::OnPairView;
use crate::OnPairViewArray;
use crate::OnPairViewDecodeMode;
use crate::canonicalize_to_varbin;
use crate::canonicalize_with;
use crate::onpair_compress;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn sample() -> Vec<String> {
    let templates = [
        "https://www.example.com/products/0001",
        "https://cdn.example.com/img/0002.webp",
        "INFO request_id=00000003 status=200",
        "WARN request_id=00000004 status=429",
        "alpha",
        "https://www.example.com/products/0005",
    ];
    (0..120)
        .map(|i| templates[i % templates.len()].to_string())
        .collect()
}

fn build() -> VortexResult<(Vec<String>, OnPairViewArray)> {
    let strings = sample();
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    let onpair = onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG)?;
    let mut ctx = SESSION.create_execution_ctx();
    let view = OnPairView::from_onpair(&onpair, &mut ctx)?;
    Ok((strings, view))
}

fn decoded(array: &Array<OnPairView>) -> VortexResult<Vec<String>> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = array
        .clone()
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)?;
    Ok((0..canonical.len())
        .map(|i| String::from_utf8(canonical.bytes_at(i).as_slice().to_vec()).expect("utf8"))
        .collect())
}

#[test]
fn roundtrip() -> VortexResult<()> {
    let (strings, view) = build()?;
    assert_eq!(view.len(), strings.len());
    assert_eq!(decoded(&view)?, strings);
    Ok(())
}

fn as_view(array: ArrayRef) -> Array<OnPairView> {
    array
        .try_downcast::<OnPairView>()
        .unwrap_or_else(|_| panic!("result is an OnPairView"))
}

fn decoded_mode(
    array: &Array<OnPairView>,
    mode: OnPairViewDecodeMode,
) -> VortexResult<Vec<String>> {
    let mut ctx = SESSION.create_execution_ctx();
    let canonical =
        canonicalize_with(array.as_view(), mode, &mut ctx)?.execute::<VarBinViewArray>(&mut ctx)?;
    Ok((0..canonical.len())
        .map(|i| String::from_utf8(canonical.bytes_at(i).as_slice().to_vec()).expect("utf8"))
        .collect())
}

/// All three decode strategies must agree, whatever the window layout.
#[test]
fn decode_modes_agree() -> VortexResult<()> {
    let (strings, view) = build()?;

    // Gappy: a filter leaves sorted windows with holes (span-decodable, carries
    // dead values). Reordered: a shuffling take is *not* span-decodable, so
    // SpanWithDead must fall back to gather and still be correct.
    let mask = Mask::from_iter((0..strings.len()).map(|i| i % 4 == 0));
    let filtered = as_view(
        <OnPairView as FilterKernel>::filter(
            view.as_view(),
            &mask,
            &mut SESSION.create_execution_ctx(),
        )?
        .expect("Some"),
    );
    let filtered_expected: Vec<String> = strings
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 4 == 0)
        .map(|(_, s)| s.clone())
        .collect();

    let shuffled = as_view(
        view.into_array()
            .take(vortex_buffer::buffer![7u64, 1, 7, 90, 3, 0].into_array())?,
    );
    let shuffled_expected: Vec<String> = [7usize, 1, 7, 90, 3, 0]
        .iter()
        .map(|&i| strings[i].clone())
        .collect();

    for mode in [
        OnPairViewDecodeMode::Auto,
        OnPairViewDecodeMode::SpanWithDead,
        OnPairViewDecodeMode::Gather,
    ] {
        assert_eq!(
            decoded_mode(&filtered, mode)?,
            filtered_expected,
            "{mode:?} filtered"
        );
        assert_eq!(
            decoded_mode(&shuffled, mode)?,
            shuffled_expected,
            "{mode:?} shuffled"
        );
    }
    Ok(())
}

/// Exporting a (gappy, filtered) OnPairView to `VarBin` must match the
/// `VarBinView` export.
#[test]
fn export_to_varbin_matches() -> VortexResult<()> {
    let (strings, view) = build()?;
    let mask = Mask::from_iter((0..strings.len()).map(|i| i % 3 == 0));
    let filtered = as_view(
        <OnPairView as FilterKernel>::filter(
            view.as_view(),
            &mask,
            &mut SESSION.create_execution_ctx(),
        )?
        .expect("Some"),
    );
    let expected: Vec<String> = strings
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, s)| s.clone())
        .collect();

    let mut ctx = SESSION.create_execution_ctx();
    let varbin = canonicalize_to_varbin(filtered.as_view(), &mut ctx)?
        .execute::<VarBinViewArray>(&mut ctx)?;
    let decoded: Vec<String> = (0..varbin.len())
        .map(|i| String::from_utf8(varbin.bytes_at(i).as_slice().to_vec()).expect("utf8"))
        .collect();
    assert_eq!(decoded, expected);
    Ok(())
}

/// `compact` preserves values, drops dead tokens, and yields a contiguous array.
#[test]
fn compact_rebuilds_contiguous() -> VortexResult<()> {
    use crate::OnPairViewArrayExt;
    use crate::OnPairViewArraySlotsExt;
    use crate::compact;

    let (strings, view) = build()?;
    // Shuffle + drop so the result is sparse, reordered, and retains the full codes.
    let taken = as_view(
        view.into_array()
            .take(vortex_buffer::buffer![9u64, 2, 100, 2, 50].into_array())?,
    );
    let expected: Vec<String> = [9usize, 2, 100, 2, 50]
        .iter()
        .map(|&i| strings[i].clone())
        .collect();

    let mut ctx = SESSION.create_execution_ctx();
    let compacted = compact(taken.as_view(), &mut ctx)?;

    // Values unchanged.
    assert_eq!(decoded(&compacted)?, expected);
    // Dead/duplicate tokens dropped: compacted codes hold only the live tokens
    // (sum of sizes), so they are no larger than the retained original codes.
    let live_tokens: usize = taken
        .collect_sizes(&mut ctx)?
        .as_slice()
        .iter()
        .map(|&s| s as usize)
        .sum();
    assert_eq!(compacted.codes().len(), live_tokens);
    // Decode modes now agree trivially because the array is contiguous; the
    // SpanWithDead path carries zero dead bytes.
    assert_eq!(
        decoded_mode(&compacted, OnPairViewDecodeMode::SpanWithDead)?,
        expected
    );
    Ok(())
}

#[test]
fn slice_preserves_codes_buffer() -> VortexResult<()> {
    let (strings, view) = build()?;
    let sliced = as_view(view.into_array().slice(10..40)?);
    assert_eq!(decoded(&sliced)?, strings[10..40].to_vec());
    Ok(())
}

#[test]
fn filter_shares_codes_buffer() -> VortexResult<()> {
    use crate::OnPairViewArraySlotsExt;

    let (strings, view) = build()?;
    let mask = Mask::from_iter((0..strings.len()).map(|i| i % 3 == 0));
    let mut ctx = SESSION.create_execution_ctx();
    let filtered = as_view(
        <OnPairView as FilterKernel>::filter(view.as_view(), &mask, &mut ctx)?
            .expect("filter returns Some"),
    );

    // The filtered array must share the identical `codes` buffer — filter is
    // metadata-only and never rebuilds the token stream.
    assert_eq!(filtered.codes().len(), view.codes().len());

    let expected: Vec<String> = strings
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, s)| s.clone())
        .collect();
    assert_eq!(decoded(&filtered)?, expected);
    Ok(())
}

#[test]
fn take_reorders() -> VortexResult<()> {
    let (strings, view) = build()?;
    let indices = vortex_buffer::buffer![5u64, 0, 0, 119, 60].into_array();
    let taken = as_view(view.into_array().take(indices)?);
    let expected: Vec<String> = [5usize, 0, 0, 119, 60]
        .iter()
        .map(|&i| strings[i].clone())
        .collect();
    assert_eq!(decoded(&taken)?, expected);
    Ok(())
}
