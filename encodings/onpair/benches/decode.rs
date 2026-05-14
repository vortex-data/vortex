// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Decode-path microbenchmarks. Drives the full `OnPairArray ->
//! VarBinViewArray` canonicalisation through Vortex's `execute::<>` API,
//! which exercises the C++-style fixed-16-byte over-copy decode loop
//! introduced to match `onpair_cpp/include/onpair/decoding/decoder.h`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::panic,
    clippy::tests_outside_test_module
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
use vortex_onpair::OnPairArray;
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

/// Canonicalise an OnPair-encoded column — the hot path readers hit.
#[divan::bench(args = [10_000usize, 100_000usize, 1_000_000usize])]
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
