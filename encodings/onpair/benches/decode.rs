// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Decode-path microbenchmarks for the OnPair Vortex array.
//!
//! * `decode_rows_unchecked` — the production decoder hot loop (combined
//!   `(offset << 16) | length` table, fixed 16-byte over-copy, 4× unrolled).
//!   Measured by hand-driving `DecodeView::decode_rows_unchecked` straight
//!   into a `Vec<u8>` so the time reflects the inner loop only.
//! * `canonicalize_to_varbinview` — the full Vortex
//!   `OnPair → VarBinViewArray` path callers actually hit. Includes
//!   `OwnedDecodeInputs::collect`, the build_views step, allocation, etc.
//!
//! Historical experiments (padded-dict, NT stores) lived here briefly and
//! were dropped after benchmarking — see git history.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::panic,
    clippy::tests_outside_test_module,
    clippy::redundant_clone,
    clippy::missing_safety_doc
)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySession;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::MAX_TOKEN_SIZE;
use vortex_onpair::OnPairArray;
use vortex_onpair::decode::OwnedDecodeInputs;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn corpus(n: usize) -> Vec<String> {
    let templates: &[&str] = &[
        "https://www.example.com/products/{id}",
        "https://cdn.example.com/img/{id}.webp",
        "https://api.example.com/v2/orders/{id}",
        "https://www.example.com/users/{id}/profile",
        "INFO  request_id={id} status=200 method=GET",
        "WARN  request_id={id} status=429 method=POST",
        "ERROR request_id={id} status=500 method=PUT",
    ];
    let mut out = Vec::with_capacity(n);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pick = (state as usize) % templates.len();
        let id = state as u32;
        out.push(templates[pick].replace("{id}", &format!("{id:08x}")));
    }
    out
}

fn compress(n: usize) -> OnPairArray {
    let strings = corpus(n);
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );
    onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG)
        .unwrap_or_else(|e| panic!("onpair_compress failed: {e}"))
}

fn materialise(arr: &OnPairArray) -> (OwnedDecodeInputs, usize, usize) {
    let mut ctx = SESSION.create_execution_ctx();
    let inputs = OwnedDecodeInputs::collect(arr.as_view(), &mut ctx)
        .unwrap_or_else(|e| panic!("collect: {e}"));
    let n = arr.len();
    let dict_offsets = inputs.dict_offsets.as_slice();
    let total: usize = inputs
        .codes
        .as_slice()
        .iter()
        .map(|&c| (dict_offsets[c as usize + 1] - dict_offsets[c as usize]) as usize)
        .sum();
    (inputs, n, total)
}

const SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

/// Raw decode loop time, excluding `OwnedDecodeInputs::collect` and
/// the allocation. Hits `DecodeView::decode_rows_unchecked` directly.
#[divan::bench(args = SIZES)]
fn decode_rows_unchecked(bencher: Bencher, n: usize) {
    let arr = compress(n);
    let (inputs, n_rows, total) = materialise(&arr);
    bencher.bench_local(|| {
        let mut out: Vec<u8> = Vec::with_capacity(total + MAX_TOKEN_SIZE);
        let dv = inputs.view();
        unsafe {
            let written = dv.decode_rows_unchecked(0, n_rows, out.as_mut_ptr());
            out.set_len(written);
        }
        divan::black_box(out);
    });
}

/// Full Vortex canonicalisation, including `execute<>` on every child,
/// building the view buffer + `BinaryView` list, etc.
#[divan::bench(args = SIZES)]
fn canonicalize_to_varbinview(bencher: Bencher, n: usize) {
    let arr = compress(n);
    bencher
        .with_inputs(|| arr.clone().into_array())
        .bench_local_values(|arr| {
            let mut ctx = SESSION.create_execution_ctx();
            divan::black_box(
                arr.execute::<VarBinViewArray>(&mut ctx)
                    .unwrap_or_else(|e| panic!("canonicalize failed: {e}")),
            )
        });
}

fn main() {
    divan::main();
}
