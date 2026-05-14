// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! End-to-end smoke test on a realistically-sized input. Not part of the unit
//! suite; run with `cargo test -p vortex-onpair --test big_data -- --nocapture`.

use std::sync::LazyLock;
use std::time::Instant;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::session::ArraySession;
use vortex_onpair::DEFAULT_DICT12_CONFIG;
use vortex_onpair::onpair_compress;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Fake-but-realistic corpus: 100k log/URL-like rows drawn from a handful of
/// templates with varying tail content. Models the kind of column OnPair
/// actually targets (high lexical repetition, short-to-medium strings).
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

    let column_bytes = arr.column_bytes().len();
    let ratio = raw_bytes as f64 / column_bytes as f64;
    eprintln!(
        "compressed {} rows ({} bytes) -> {} bytes (ratio {:.2}x) in {:?}",
        n, raw_bytes, column_bytes, ratio, compress_elapsed
    );
    eprintln!("dict_size={} bits={}", arr.dict_size(), arr.bits());

    let mut ctx = SESSION.create_execution_ctx();

    // Full canonicalisation round-trip.
    let t0 = Instant::now();
    let decoded = arr
        .clone()
        .into_array()
        .execute::<VarBinViewArray>(&mut ctx)
        .expect("canonicalize");
    let decompress_elapsed = t0.elapsed();
    eprintln!("canonicalized in {:?}", decompress_elapsed);

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

    // Predicate spot-checks: numbers must match a brute-force scan.
    let column = arr.column().expect("materialize column");

    let needle_eq = strings[42].as_bytes();
    let want_eq = strings.iter().filter(|s| s.as_bytes() == needle_eq).count();
    let bits = column.equals_bitmap(needle_eq).unwrap();
    let got_eq = popcount(&bits, n);
    eprintln!(
        "equals('row 42 payload')  expected={} got={}",
        want_eq, got_eq
    );
    assert_eq!(got_eq, want_eq);

    let prefix = b"https://www.";
    let want_prefix = strings
        .iter()
        .filter(|s| s.as_bytes().starts_with(prefix))
        .count();
    let bits = column.starts_with_bitmap(prefix).unwrap();
    let got_prefix = popcount(&bits, n);
    eprintln!(
        "starts_with('https://www.')  expected={} got={}",
        want_prefix, got_prefix
    );
    assert_eq!(got_prefix, want_prefix);

    let needle_sub = b"status=500";
    let want_sub = strings
        .iter()
        .filter(|s| {
            s.as_bytes()
                .windows(needle_sub.len())
                .any(|w| w == needle_sub)
        })
        .count();
    let bits = column.contains_bitmap(needle_sub).unwrap();
    let got_sub = popcount(&bits, n);
    eprintln!(
        "contains('status=500')       expected={} got={}",
        want_sub, got_sub
    );
    assert_eq!(got_sub, want_sub);
}

fn popcount(bits: &[u8], n: usize) -> usize {
    let mut c = 0;
    for i in 0..n {
        if (bits[i / 8] >> (i % 8)) & 1 == 1 {
            c += 1;
        }
    }
    c
}
