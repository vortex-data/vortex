// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! End-to-end smoke test on a realistically-sized input. Validates the
//! pure-Rust decode path and pushdown predicates end-to-end through the new
//! u16-codes layout.

#![allow(
    clippy::cast_possible_truncation,
    clippy::redundant_clone,
    clippy::tests_outside_test_module,
    clippy::use_debug
)]

use std::sync::LazyLock;
use std::time::Instant;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn corpus(n: usize) -> Vec<String> {
    let templates: &[&str] = &[
        "GET /api/v1/users/{id}/profile HTTP/1.1",
        "POST /api/v1/users/{id}/sessions HTTP/1.1",
        "GET /static/js/app.{id}.js HTTP/1.1",
        "GET /static/css/app.{id}.css HTTP/1.1",
        "https://www.example.com/products/{id}",
        "https://cdn.example.com/img/{id}.webp",
        "https://api.example.com/v2/orders/{id}",
        "ftp://files.example.com/dump/{id}.tar.gz",
        "ssh://deploy@build-{id}.internal:22",
        "redis://cache-{id}.svc.cluster.local:6379",
        "INFO  request_id={id} method=GET status=200",
        "WARN  request_id={id} method=POST status=429",
        "ERROR request_id={id} method=PUT  status=500",
    ];
    let mut out = Vec::with_capacity(n);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let pick = (state as usize) % templates.len();
        let id = state as u32;
        out.push(templates[pick].replace("{id}", &format!("{:08x}", id)));
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn smoke_100k_rows() {
    let n = 100_000;
    let strings = corpus(n);
    let raw_bytes: usize = strings.iter().map(|s| s.len()).sum();

    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_bytes())),
        DType::Utf8(Nullability::NonNullable),
    );

    let t0 = Instant::now();
    let arr = onpair_compress(&varbin, varbin.len(), varbin.dtype(), DEFAULT_DICT12_CONFIG)
        .expect("compress");
    let compress_elapsed = t0.elapsed();
    let bits = arr.bits();
    eprintln!(
        "compressed {} rows ({} raw bytes) in {:?}, bits={}",
        n, raw_bytes, compress_elapsed, bits
    );

    let arr_ref = arr.into_array();
    let mut ctx = SESSION.create_execution_ctx();

    // Full canonical round-trip via the pure-Rust decoder.
    let t0 = Instant::now();
    let decoded = arr_ref
        .clone()
        .execute::<VarBinViewArray>(&mut ctx)
        .expect("canonicalize");
    eprintln!("canonicalized in {:?}", t0.elapsed());

    assert_eq!(decoded.len(), n);
    decoded
        .with_iterator(|iter| {
            for (i, got) in iter.enumerate() {
                let want = strings[i].as_bytes();
                assert_eq!(got, Some(want), "row {} mismatch", i);
            }
            Ok::<_, vortex_error::VortexError>(())
        })
        .unwrap();
    eprintln!("roundtrip OK on all {} rows", n);

    // Equality pushdown: pick a specific row's value and ensure the kernel
    // finds all occurrences.
    let needle_row = 42;
    let needle = strings[needle_row].clone();
    let want_eq = strings.iter().filter(|s| **s == needle).count();
    let eq = arr_ref
        .binary(
            ConstantArray::new(needle.as_str(), n).into_array(),
            Operator::Eq,
        )
        .unwrap()
        .execute::<vortex_array::Canonical>(&mut ctx)
        .unwrap()
        .into_array();
    assert_eq!(eq.as_bool_typed().true_count().unwrap(), want_eq);
    eprintln!("eq pushdown matches reference count ({})", want_eq);
}
