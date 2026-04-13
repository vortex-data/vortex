// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::like;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_fsst::test_utils::HIGH_MATCH_DOMAIN;
use vortex_fsst::test_utils::LOW_MATCH_DOMAIN;
use vortex_fsst::test_utils::NUM_STRINGS;
use vortex_fsst::test_utils::generate_url_data;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const NUM_URLS: usize = NUM_STRINGS;

static URL_DATA: LazyLock<VarBinArray> = LazyLock::new(generate_url_data);

// ---------------------------------------------------------------------------
// Eq compare benchmarks (FSST pushdown vs canonicalize)
// ---------------------------------------------------------------------------

/// Pick a concrete URL from the dataset that uses the given domain.
fn pick_url_with_domain(data: &VarBinArray, domain: &str) -> String {
    use vortex_array::accessor::ArrayAccessor;
    data.with_iterator(|iter| {
        iter.flatten()
            .map(|b| std::str::from_utf8(b).unwrap().to_string())
            .find(|u| u.contains(domain))
            .unwrap_or_else(|| format!("http://{domain}/missing"))
    })
}

#[divan::bench]
fn eq_pushdown_high_match(bencher: Bencher) {
    let data = &*URL_DATA;
    let compressor = fsst_train_compressor(data);
    let fsst_array = fsst_compress(data, data.len(), data.dtype(), &compressor);
    let match_url = pick_url_with_domain(data, HIGH_MATCH_DOMAIN);
    let constant = ConstantArray::new(Scalar::from(match_url.as_str()), NUM_URLS);

    bencher
        .with_inputs(|| (&fsst_array, &constant, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, constant, ctx)| {
            fsst_array
                .clone()
                .into_array()
                .binary(constant.clone().into_array(), Operator::Eq)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn eq_pushdown_low_match(bencher: Bencher) {
    let data = &*URL_DATA;
    let compressor = fsst_train_compressor(data);
    let fsst_array = fsst_compress(data, data.len(), data.dtype(), &compressor);
    let match_url = pick_url_with_domain(data, LOW_MATCH_DOMAIN);
    let constant = ConstantArray::new(Scalar::from(match_url.as_str()), NUM_URLS);

    bencher
        .with_inputs(|| (&fsst_array, &constant, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, constant, ctx)| {
            fsst_array
                .clone()
                .into_array()
                .binary(constant.clone().into_array(), Operator::Eq)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn eq_canonicalize_high_match(bencher: Bencher) {
    let data = &*URL_DATA;
    let compressor = fsst_train_compressor(data);
    let fsst_array = fsst_compress(data, data.len(), data.dtype(), &compressor);
    let match_url = pick_url_with_domain(data, HIGH_MATCH_DOMAIN);
    let constant = ConstantArray::new(Scalar::from(match_url.as_str()), NUM_URLS);

    bencher
        .with_inputs(|| (&fsst_array, &constant, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, constant, ctx)| {
            fsst_array
                .to_canonical()
                .unwrap()
                .into_array()
                .binary(constant.clone().into_array(), Operator::Eq)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn eq_canonicalize_low_match(bencher: Bencher) {
    let data = &*URL_DATA;
    let compressor = fsst_train_compressor(data);
    let fsst_array = fsst_compress(data, data.len(), data.dtype(), &compressor);
    let match_url = pick_url_with_domain(data, LOW_MATCH_DOMAIN);
    let constant = ConstantArray::new(Scalar::from(match_url.as_str()), NUM_URLS);

    bencher
        .with_inputs(|| (&fsst_array, &constant, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, constant, ctx)| {
            fsst_array
                .to_canonical()
                .unwrap()
                .into_array()
                .binary(constant.clone().into_array(), Operator::Eq)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

// ---------------------------------------------------------------------------
// LIKE substring benchmarks (always goes through canonicalization for FSST)
// ---------------------------------------------------------------------------

#[divan::bench]
fn like_substr_high_match(bencher: Bencher) {
    let data = &*URL_DATA;
    let compressor = fsst_train_compressor(data);
    let fsst_array = fsst_compress(data, data.len(), data.dtype(), &compressor);
    let pattern = format!("%{HIGH_MATCH_DOMAIN}%");
    let expr = like(root(), lit(pattern.as_str()));

    bencher
        .with_inputs(|| (&fsst_array, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, ctx)| {
            fsst_array
                .clone()
                .into_array()
                .apply(&expr)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench]
fn like_substr_low_match(bencher: Bencher) {
    let data = &*URL_DATA;
    let compressor = fsst_train_compressor(data);
    let fsst_array = fsst_compress(data, data.len(), data.dtype(), &compressor);
    let pattern = format!("%{LOW_MATCH_DOMAIN}%");
    let expr = like(root(), lit(pattern.as_str()));

    bencher
        .with_inputs(|| (&fsst_array, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, ctx)| {
            fsst_array
                .clone()
                .into_array()
                .apply(&expr)
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}
