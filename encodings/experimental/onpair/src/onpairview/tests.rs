// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySession;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::DEFAULT_DICT12_CONFIG;
use crate::OnPairView;
use crate::OnPairViewArray;
use crate::OnPairViewArraySlotsExt;
use crate::canonicalize_to_varbin;
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

fn as_view(array: ArrayRef) -> Array<OnPairView> {
    array
        .try_downcast::<OnPairView>()
        .unwrap_or_else(|_| panic!("result is an OnPairView"))
}

#[test]
fn roundtrip() -> VortexResult<()> {
    let (strings, view) = build()?;
    assert_eq!(view.len(), strings.len());
    assert_eq!(decoded(&view)?, strings);
    Ok(())
}

/// `slice`, `filter` and `take` are all metadata-only: they rewrite the per-row
/// children but **share the `codes` token buffer verbatim** (never rebuild it),
/// and the shared buffer still decodes correctly whether the surviving windows
/// are contiguous (`slice`), gappy (`filter`), or reordered with duplicates
/// (`take`).
#[test]
fn metadata_only_ops_share_codes() -> VortexResult<()> {
    let (strings, view) = build()?;
    let codes_len = view.codes().len();

    // slice — contiguous window.
    let sliced = as_view(view.clone().into_array().slice(10..40)?);
    assert_eq!(sliced.codes().len(), codes_len, "slice shares codes");
    assert_eq!(decoded(&sliced)?, strings[10..40].to_vec());

    // filter — gappy (sorted windows with holes).
    let mask = Mask::from_iter((0..strings.len()).map(|i| i % 3 == 0));
    let filtered = as_view(
        <OnPairView as FilterKernel>::filter(
            view.as_view(),
            &mask,
            &mut SESSION.create_execution_ctx(),
        )?
        .expect("filter returns Some"),
    );
    assert_eq!(filtered.codes().len(), codes_len, "filter shares codes");
    let filter_expected: Vec<String> = strings
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, s)| s.clone())
        .collect();
    assert_eq!(decoded(&filtered)?, filter_expected);

    // take — reordered, with duplicates (not span-decodable, must gather).
    let taken = as_view(
        view.into_array()
            .take(vortex_buffer::buffer![7u64, 1, 7, 90, 3, 0].into_array())?,
    );
    assert_eq!(taken.codes().len(), codes_len, "take shares codes");
    let take_expected: Vec<String> = [7usize, 1, 7, 90, 3, 0]
        .iter()
        .map(|&i| strings[i].clone())
        .collect();
    assert_eq!(decoded(&taken)?, take_expected);

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
    // (sum of sizes).
    let live_tokens: usize = taken
        .collect_sizes(&mut ctx)?
        .as_slice()
        .iter()
        .map(|&s| s as usize)
        .sum();
    assert_eq!(compacted.codes().len(), live_tokens);
    Ok(())
}

/// Serialize an `OnPairView` array through the wire format and decode it back,
/// asserting the result is still `OnPairView`-encoded and round-trips its
/// values. This exercises the same `serialize`/`deserialize` path used by the
/// file writer once the encoding is registered (`register_default_encodings`).
#[test]
fn serde_roundtrip() -> VortexResult<()> {
    // The encoding must be registered for the session to (de)serialize it.
    let session = VortexSession::empty().with::<ArraySession>();
    session.arrays().register(OnPairView);

    let (strings, view) = build()?;
    let array = view.into_array();
    let dtype = array.dtype().clone();
    let len = array.len();

    let ctx = ArrayContext::empty();
    let serialized = array.serialize(&ctx, &session, &SerializeOptions::default())?;

    let mut concat = ByteBufferMut::empty();
    for buf in serialized {
        concat.extend_from_slice(buf.as_ref());
    }
    let parts = SerializedArray::try_from(concat.freeze())?;
    let decoded_array = parts.decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)?;

    assert_eq!(decoded_array.dtype(), &dtype);
    assert_eq!(decoded_array.len(), len);
    let decoded_view = as_view(decoded_array);
    assert_eq!(decoded(&decoded_view)?, strings);
    Ok(())
}
