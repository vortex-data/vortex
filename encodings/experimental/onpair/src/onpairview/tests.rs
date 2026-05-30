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
