// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(clippy::expect_used, clippy::unwrap_used)]

//! Training and parsing benchmarks over a prepared 32 MiB real-data corpus.
//!
//! Prepare corpora with `scripts/prepare_corpora.py`, then select one with
//! `ONPAIR_BENCH_CORPUS`. Corpus loading and dictionary construction for the
//! parse benchmark happen outside its timed loop.

mod corpus;

use std::env;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;
use std::time::Instant;

use corpus::Corpus;
use divan::Bencher;
use vortex_onpair_rs::DynamicThreshold;
use vortex_onpair_rs::HybridReversedMatcher;
use vortex_onpair_rs::ReversedAhoCorasickMatcher;
use vortex_onpair_rs::Store;
use vortex_onpair_rs::ThresholdSpec;
use vortex_onpair_rs::TrainResult;
use vortex_onpair_rs::TrainingConfig;
use vortex_onpair_rs::parse;
use vortex_onpair_rs::parse_reversed;
use vortex_onpair_rs::parse_reversed_avx512;
use vortex_onpair_rs::parse_reversed_hybrid_avx512;
use vortex_onpair_rs::parse_reversed_interleaved;
use vortex_onpair_rs::train;

fn bits() -> u8 {
    env::var("ONPAIR_BITS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(12)
}

fn corpus() -> &'static Corpus {
    static CORPUS: OnceLock<Corpus> = OnceLock::new();
    CORPUS.get_or_init(|| {
        let path = env::var("ONPAIR_BENCH_CORPUS")
            .expect("set ONPAIR_BENCH_CORPUS to a file made by scripts/prepare_corpora.py");
        let corpus = Corpus::load(Path::new(&path)).unwrap();
        eprintln!(
            "[onpair bench] {}: {} rows, {:.2} MiB",
            corpus.source,
            corpus.rows(),
            corpus.bytes.len() as f64 / (1024.0 * 1024.0)
        );
        corpus
    })
}

fn sample_fraction() -> f64 {
    env::var("ONPAIR_SAMPLE_FRACTION")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.5)
}

fn training_config() -> TrainingConfig {
    TrainingConfig {
        bits: bits(),
        threshold: ThresholdSpec::Dynamic(DynamicThreshold {
            sample_fraction: sample_fraction(),
        }),
        seed: Some(42),
    }
}

fn trained() -> &'static TrainResult {
    static TRAINED: OnceLock<TrainResult> = OnceLock::new();
    TRAINED.get_or_init(|| {
        let corpus = corpus();
        train(
            &corpus.bytes,
            &corpus.offsets_u32,
            corpus.rows(),
            &training_config(),
        )
    })
}

fn reversed_matcher() -> &'static ReversedAhoCorasickMatcher {
    static MATCHER: OnceLock<ReversedAhoCorasickMatcher> = OnceLock::new();
    MATCHER.get_or_init(|| {
        let matcher = ReversedAhoCorasickMatcher::from_dictionary(&trained().dict);
        eprintln!(
            "[onpair bench] reversed automaton: {} states, {:.2} MiB, {} transitions",
            matcher.num_states(),
            matcher.automaton_bytes() as f64 / (1024.0 * 1024.0),
            if matcher.uses_dense_transitions() {
                "dense DFA"
            } else {
                "sparse AC"
            }
        );
        matcher
    })
}

fn report_iterations() -> usize {
    env::var("ONPAIR_REPORT_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5)
}

fn report_timing(mut run: impl FnMut(), bytes: Option<usize>) {
    run();
    let mut times = Vec::with_capacity(report_iterations());
    for _ in 0..report_iterations() {
        let start = Instant::now();
        run();
        times.push(start.elapsed());
    }
    times.sort_unstable();
    let min = times[0];
    let median = times[times.len() / 2];
    let max = times[times.len() - 1];
    print!(
        "min_ms={:.3} median_ms={:.3} max_ms={:.3}",
        milliseconds(min),
        milliseconds(median),
        milliseconds(max)
    );
    if let Some(bytes) = bytes {
        let mib_per_second = bytes as f64 / (1024.0 * 1024.0) / median.as_secs_f64();
        print!(" median_mib_s={mib_per_second:.2}");
    }
    println!();
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn local_report() {
    let corpus = corpus();
    let config = training_config();
    print!("train_dictionary ");
    report_timing(
        || {
            std::hint::black_box(train(
                std::hint::black_box(&corpus.bytes),
                std::hint::black_box(&corpus.offsets_u32),
                corpus.rows(),
                &config,
            ));
        },
        Some(corpus.bytes.len()),
    );

    let trained = train(&corpus.bytes, &corpus.offsets_u32, corpus.rows(), &config);
    print!("build_reversed_automaton ");
    report_timing(
        || {
            std::hint::black_box(ReversedAhoCorasickMatcher::from_dictionary(
                std::hint::black_box(&trained.dict),
            ));
        },
        None,
    );
    let matcher = ReversedAhoCorasickMatcher::from_dictionary(&trained.dict);
    println!(
        "automaton states={} trie_states={} bytes={} representation={}",
        matcher.num_states(),
        matcher.trie_states(),
        matcher.automaton_bytes(),
        if matcher.uses_dense_transitions() {
            "dense_dfa"
        } else {
            "sparse_ac"
        }
    );

    let mut greedy = Store::default();
    let mut reversed = Store::default();
    let mut interleaved = Store::default();
    let mut avx512 = Store::default();
    parse(
        &corpus.bytes,
        &corpus.offsets_u32,
        corpus.rows(),
        &trained.lpm,
        bits(),
        &mut greedy,
    );
    parse_reversed(
        &corpus.bytes,
        &corpus.offsets_u32,
        corpus.rows(),
        &matcher,
        bits(),
        &mut reversed,
    );
    parse_reversed_interleaved(
        &corpus.bytes,
        &corpus.offsets_u32,
        corpus.rows(),
        &matcher,
        bits(),
        &mut interleaved,
    );
    parse_reversed_avx512(
        &corpus.bytes,
        &corpus.offsets_u32,
        corpus.rows(),
        &matcher,
        bits(),
        &mut avx512,
    );
    assert_eq!(reversed.boundaries, greedy.boundaries);
    assert_eq!(reversed.packed, greedy.packed);
    assert_eq!(interleaved.boundaries, greedy.boundaries);
    assert_eq!(interleaved.packed, greedy.packed);
    assert_eq!(avx512.boundaries, greedy.boundaries);
    assert_eq!(avx512.packed, greedy.packed);

    print!("parse_greedy ");
    report_timing(
        || {
            let mut store = Store::default();
            parse(
                std::hint::black_box(&corpus.bytes),
                std::hint::black_box(&corpus.offsets_u32),
                corpus.rows(),
                std::hint::black_box(&trained.lpm),
                bits(),
                &mut store,
            );
            std::hint::black_box(store);
        },
        Some(corpus.bytes.len()),
    );
    print!("parse_reversed_interleaved ");
    report_timing(
        || {
            let mut store = Store::default();
            parse_reversed_interleaved(
                std::hint::black_box(&corpus.bytes),
                std::hint::black_box(&corpus.offsets_u32),
                corpus.rows(),
                std::hint::black_box(&matcher),
                bits(),
                &mut store,
            );
            std::hint::black_box(store);
        },
        Some(corpus.bytes.len()),
    );
    print!("parse_reversed_avx512 ");
    report_timing(
        || {
            let mut store = Store::default();
            parse_reversed_avx512(
                std::hint::black_box(&corpus.bytes),
                std::hint::black_box(&corpus.offsets_u32),
                corpus.rows(),
                std::hint::black_box(&matcher),
                bits(),
                &mut store,
            );
            std::hint::black_box(store);
        },
        Some(corpus.bytes.len()),
    );
    print!("parse_reversed_dfa ");
    report_timing(
        || {
            let mut store = Store::default();
            parse_reversed(
                std::hint::black_box(&corpus.bytes),
                std::hint::black_box(&corpus.offsets_u32),
                corpus.rows(),
                std::hint::black_box(&matcher),
                bits(),
                &mut store,
            );
            std::hint::black_box(store);
        },
        Some(corpus.bytes.len()),
    );

    print!("build_hybrid_matcher ");
    report_timing(
        || {
            std::hint::black_box(HybridReversedMatcher::from_dictionary(
                std::hint::black_box(&trained.dict),
            ));
        },
        None,
    );
    let hybrid = HybridReversedMatcher::from_dictionary(&trained.dict);
    println!(
        "hybrid automaton states={} bytes={} long_tokens={}",
        hybrid.num_states(),
        hybrid.automaton_bytes(),
        hybrid.long_tokens()
    );
    let mut hybrid_store = Store::default();
    parse_reversed_hybrid_avx512(
        &corpus.bytes,
        &corpus.offsets_u32,
        corpus.rows(),
        &hybrid,
        bits(),
        &mut hybrid_store,
    );
    assert_eq!(hybrid_store.boundaries, greedy.boundaries);
    assert_eq!(hybrid_store.packed, greedy.packed);
    print!("parse_reversed_hybrid_avx512 ");
    report_timing(
        || {
            let mut store = Store::default();
            parse_reversed_hybrid_avx512(
                std::hint::black_box(&corpus.bytes),
                std::hint::black_box(&corpus.offsets_u32),
                corpus.rows(),
                std::hint::black_box(&hybrid),
                bits(),
                &mut store,
            );
            std::hint::black_box(store);
        },
        Some(corpus.bytes.len()),
    );
}

#[divan::bench]
fn train_dictionary(bencher: Bencher) {
    let corpus = corpus();
    bencher
        .counter(divan::counter::BytesCount::new(corpus.bytes.len()))
        .bench(|| {
            train(
                divan::black_box(&corpus.bytes),
                divan::black_box(&corpus.offsets_u32),
                corpus.rows(),
                &training_config(),
            )
        });
}

#[divan::bench]
fn parse_greedy(bencher: Bencher) {
    let corpus = corpus();
    let trained = trained();
    bencher
        .counter(divan::counter::BytesCount::new(corpus.bytes.len()))
        .bench_local(|| {
            let mut store = Store::default();
            parse(
                divan::black_box(&corpus.bytes),
                divan::black_box(&corpus.offsets_u32),
                corpus.rows(),
                divan::black_box(&trained.lpm),
                bits(),
                &mut store,
            );
            divan::black_box(store)
        });
}

#[divan::bench]
fn build_reversed_automaton(bencher: Bencher) {
    let dict = &trained().dict;
    bencher.bench(|| ReversedAhoCorasickMatcher::from_dictionary(divan::black_box(dict)));
}

#[divan::bench]
fn parse_reversed_dfa(bencher: Bencher) {
    let corpus = corpus();
    let matcher = reversed_matcher();
    bencher
        .counter(divan::counter::BytesCount::new(corpus.bytes.len()))
        .bench_local(|| {
            let mut store = Store::default();
            parse_reversed(
                divan::black_box(&corpus.bytes),
                divan::black_box(&corpus.offsets_u32),
                corpus.rows(),
                divan::black_box(matcher),
                bits(),
                &mut store,
            );
            divan::black_box(store)
        });
}

#[divan::bench]
fn parse_reversed_interleaved_rows(bencher: Bencher) {
    let corpus = corpus();
    let matcher = reversed_matcher();
    bencher
        .counter(divan::counter::BytesCount::new(corpus.bytes.len()))
        .bench_local(|| {
            let mut store = Store::default();
            parse_reversed_interleaved(
                divan::black_box(&corpus.bytes),
                divan::black_box(&corpus.offsets_u32),
                corpus.rows(),
                divan::black_box(matcher),
                bits(),
                &mut store,
            );
            divan::black_box(store)
        });
}

#[divan::bench]
fn parse_reversed_avx512_rows(bencher: Bencher) {
    let corpus = corpus();
    let matcher = reversed_matcher();
    bencher
        .counter(divan::counter::BytesCount::new(corpus.bytes.len()))
        .bench_local(|| {
            let mut store = Store::default();
            parse_reversed_avx512(
                divan::black_box(&corpus.bytes),
                divan::black_box(&corpus.offsets_u32),
                corpus.rows(),
                divan::black_box(matcher),
                bits(),
                &mut store,
            );
            divan::black_box(store)
        });
}

fn main() {
    let _ = corpus();
    if env::var_os("ONPAIR_LOCAL_REPORT").is_some() {
        local_report();
        return;
    }
    divan::main();
}
