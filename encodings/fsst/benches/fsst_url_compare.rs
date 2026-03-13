// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::compute::warm_up_vtables;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::like;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

fn main() {
    warm_up_vtables();
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const NUM_URLS: usize = 100_000;

/// A high-frequency domain that appears in ~50% of generated URLs.
const HIGH_MATCH_DOMAIN: &str = "smeshariki.ru";

/// A low-frequency domain that appears in ~1% of generated URLs.
const LOW_MATCH_DOMAIN: &str = "rare-example-domain.com";

// Domains modeled after real ClickBench URL distributions.
const DOMAINS: &[(&str, u32)] = &[
    ("smeshariki.ru", 500),          // ~50%
    ("auto.ru", 150),                // ~15%
    ("komme.ru", 100),               // ~10%
    ("yandex.ru", 80),               //  ~8%
    ("mail.ru", 60),                 //  ~6%
    ("livejournal.com", 40),         //  ~4%
    ("vk.com", 30),                  //  ~3%
    ("avito.ru", 20),                //  ~2%
    ("kinopoisk.ru", 10),            //   ~1%
    ("rare-example-domain.com", 10), //  ~1%
];

const PATHS: &[&str] = &[
    "/GameMain.aspx",
    "/index.php",
    "/catalog/item",
    "/search",
    "/news/article",
    "/user/profile",
    "/collection/view",
    "/cars/used/sale",
    "/forum/thread",
    "/photo/album",
    "/video/watch",
    "/download/file",
    "/api/v1/resource",
    "/shop/product",
    "/blog/post",
];

/// Generate 100k realistic ClickBench-style URLs.
fn generate_url_data() -> VarBinArray {
    let mut rng = StdRng::seed_from_u64(42);

    // Build a weighted domain lookup.
    let total_weight: u32 = DOMAINS.iter().map(|(_, w)| w).sum();
    let urls: Vec<Option<Box<[u8]>>> = (0..NUM_URLS)
        .map(|_| {
            let domain_roll = rng.random_range(0..total_weight);
            let mut cumulative = 0u32;
            let mut domain = DOMAINS[0].0;
            for &(d, w) in DOMAINS {
                cumulative += w;
                if domain_roll < cumulative {
                    domain = d;
                    break;
                }
            }

            let path = PATHS[rng.random_range(0..PATHS.len())];
            let query_id: u32 = rng.random_range(1..100_000);
            let tab: u16 = rng.random_range(1..20);

            let url = format!("http://{domain}{path}?id={query_id}&tab={tab}#ref={query_id}");
            Some(url.into_bytes().into_boxed_slice())
        })
        .collect();

    VarBinArray::from_iter(urls, DType::Utf8(Nullability::NonNullable))
}

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
    let fsst_array = fsst_compress(data, &compressor);
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
    let fsst_array = fsst_compress(data, &compressor);
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
    let fsst_array = fsst_compress(data, &compressor);
    let match_url = pick_url_with_domain(data, HIGH_MATCH_DOMAIN);
    let constant = ConstantArray::new(Scalar::from(match_url.as_str()), NUM_URLS);

    bencher
        .with_inputs(|| (&fsst_array, &constant, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, constant, ctx)| {
            fsst_array
                .to_canonical()
                .unwrap()
                .as_ref()
                .to_array()
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
    let fsst_array = fsst_compress(data, &compressor);
    let match_url = pick_url_with_domain(data, LOW_MATCH_DOMAIN);
    let constant = ConstantArray::new(Scalar::from(match_url.as_str()), NUM_URLS);

    bencher
        .with_inputs(|| (&fsst_array, &constant, SESSION.create_execution_ctx()))
        .bench_refs(|(fsst_array, constant, ctx)| {
            fsst_array
                .to_canonical()
                .unwrap()
                .as_ref()
                .to_array()
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
    let fsst_array = fsst_compress(data, &compressor);
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
    let fsst_array = fsst_compress(data, &compressor);
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
