// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! High-level dispatch over every available backend.
//!
//! [`run_backend`] takes a [`BackendKind`], a [`BackendConfig`], and a
//! corpus; trains/compresses; runs equality, contains, and starts_with
//! pushdowns; and returns a single [`BackendResult`] holding wall-clock
//! timings plus the compressed-size figures. Both the report binary and the
//! divan harness call into this so the two stay in lock-step.

use std::time::{Duration, Instant};

use crate::backends::{Backend, BackendConfig, Pushdown, fsst_rs_backend::FsstRsBackend};

#[cfg(feature = "fsst-cpp")]
use crate::backends::{fsst_cpp_8::FsstCpp8Backend, fsst_cpp_12::FsstCpp12Backend};
#[cfg(feature = "onpair")]
use crate::backends::{onpair16_backend::OnPair16Backend, onpair_backend::OnPairBackend};
#[cfg(feature = "onpair-cpp")]
use crate::backends::onpair_cpp_backend::OnPairCppBackend;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    FsstRs,
    #[cfg(feature = "fsst-cpp")]
    FsstCpp8,
    #[cfg(feature = "fsst-cpp")]
    FsstCpp12,
    #[cfg(feature = "onpair")]
    OnPair,
    #[cfg(feature = "onpair")]
    OnPair16,
    #[cfg(feature = "onpair-cpp")]
    OnPairCpp,
}

impl BackendKind {
    /// Every backend that compiled in for this binary, in a stable order.
    pub fn all() -> &'static [BackendKind] {
        &[
            BackendKind::FsstRs,
            #[cfg(feature = "fsst-cpp")]
            BackendKind::FsstCpp8,
            #[cfg(feature = "fsst-cpp")]
            BackendKind::FsstCpp12,
            #[cfg(feature = "onpair")]
            BackendKind::OnPair,
            #[cfg(feature = "onpair")]
            BackendKind::OnPair16,
            #[cfg(feature = "onpair-cpp")]
            BackendKind::OnPairCpp,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            BackendKind::FsstRs => "fsst-rs",
            #[cfg(feature = "fsst-cpp")]
            BackendKind::FsstCpp8 => "fsst-cpp-8",
            #[cfg(feature = "fsst-cpp")]
            BackendKind::FsstCpp12 => "fsst-cpp-12",
            #[cfg(feature = "onpair")]
            BackendKind::OnPair => "onpair",
            #[cfg(feature = "onpair")]
            BackendKind::OnPair16 => "onpair16",
            #[cfg(feature = "onpair-cpp")]
            BackendKind::OnPairCpp => "onpair-cpp",
        }
    }

    pub fn supports_compressed_equality(self) -> bool {
        // Both FSST variants support a true compressed-domain equality
        // pushdown: compress the needle once and memcmp the codes. OnPair
        // (Rust port) does not expose its dictionary so it has to
        // decompress-then-compare.
        match self {
            BackendKind::FsstRs => true,
            #[cfg(feature = "fsst-cpp")]
            BackendKind::FsstCpp8 | BackendKind::FsstCpp12 => true,
            #[cfg(feature = "onpair")]
            BackendKind::OnPair | BackendKind::OnPair16 => false,
            // onpair_cpp ships an EqAutomaton -- this is a true
            // compressed-domain equality pushdown.
            #[cfg(feature = "onpair-cpp")]
            BackendKind::OnPairCpp => true,
        }
    }

    pub fn supports_compressed_substring(self) -> bool {
        // Only `onpair-cpp` ships a KMP automaton over compressed tokens.
        // FSST has no equivalent (a needle can straddle code boundaries),
        // and onpair_rs does not expose its dictionary.
        match self {
            #[cfg(feature = "onpair-cpp")]
            BackendKind::OnPairCpp => true,
            _ => false,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MeasureOpts {
    /// Number of times to compress for averaging. The harness picks the best
    /// run rather than the average so a single noisy iteration cannot poison
    /// the report.
    pub compress_iters: u32,
    /// Same idea for decompression.
    pub decompress_iters: u32,
    /// How many pushdown evaluations of each needle to run.
    pub pushdown_iters: u32,
}

impl Default for MeasureOpts {
    fn default() -> Self {
        Self { compress_iters: 3, decompress_iters: 3, pushdown_iters: 3 }
    }
}

#[derive(Clone, Debug)]
pub struct BackendResult {
    pub backend: &'static str,
    pub dataset: &'static str,
    pub rows: usize,
    pub uncompressed_bytes: usize,
    pub compressed_payload_bytes: usize,
    pub total_compressed_bytes: usize,
    pub compress: Duration,
    pub decompress: Duration,
    pub equality_pushdown: Option<Duration>,
    pub contains_pushdown: Option<Duration>,
    pub starts_with_pushdown: Option<Duration>,
    pub equality_is_compressed_domain: bool,
    pub substring_is_compressed_domain: bool,
    pub roundtrip_ok: bool,
    /// Number of rows matched by the first equality needle (sanity check).
    pub equality_hits: usize,
    pub contains_hits: usize,
    pub starts_with_hits: usize,
}

pub fn run_backend(
    kind: BackendKind,
    strings: &[Vec<u8>],
    dataset_name: &'static str,
    needles: &[Vec<u8>],
    cfg: BackendConfig,
    opts: MeasureOpts,
) -> BackendResult {
    match kind {
        BackendKind::FsstRs => measure(
            kind,
            || FsstRsBackend::train_and_compress(strings),
            strings,
            dataset_name,
            needles,
            opts,
        ),
        #[cfg(feature = "fsst-cpp")]
        BackendKind::FsstCpp8 => measure(
            kind,
            || FsstCpp8Backend::train_and_compress(strings),
            strings,
            dataset_name,
            needles,
            opts,
        ),
        #[cfg(feature = "fsst-cpp")]
        BackendKind::FsstCpp12 => measure(
            kind,
            || FsstCpp12Backend::train_and_compress(strings),
            strings,
            dataset_name,
            needles,
            opts,
        ),
        #[cfg(feature = "onpair")]
        BackendKind::OnPair => measure(
            kind,
            || OnPairBackend::train_and_compress(strings, cfg.onpair_threshold),
            strings,
            dataset_name,
            needles,
            opts,
        ),
        #[cfg(feature = "onpair")]
        BackendKind::OnPair16 => measure(
            kind,
            || OnPair16Backend::train_and_compress(strings, cfg.onpair_threshold),
            strings,
            dataset_name,
            needles,
            opts,
        ),
        #[cfg(feature = "onpair-cpp")]
        BackendKind::OnPairCpp => measure(
            kind,
            || {
                OnPairCppBackend::train_and_compress(
                    strings,
                    cfg.onpair_cpp_bits,
                    cfg.onpair_cpp_seed,
                )
            },
            strings,
            dataset_name,
            needles,
            opts,
        ),
    }
}

fn measure<B: Backend + Pushdown, F: Fn() -> B>(
    kind: BackendKind,
    factory: F,
    strings: &[Vec<u8>],
    dataset_name: &'static str,
    needles: &[Vec<u8>],
    opts: MeasureOpts,
) -> BackendResult {
    let uncompressed: usize = strings.iter().map(|s| s.len()).sum();

    // Compression: best-of-N over training + encoding. Both phases are
    // included because users see them as a single "ingest" cost; isolating
    // training is more interesting in the divan suite.
    let mut compressed = factory();
    let mut best_compress = Duration::MAX;
    for _ in 0..opts.compress_iters {
        let t = Instant::now();
        compressed = factory();
        best_compress = best_compress.min(t.elapsed());
    }

    // Decompression: best-of-N over a full corpus decode.
    let mut decoded: Vec<Vec<u8>> = Vec::new();
    let mut best_decompress = Duration::MAX;
    for _ in 0..opts.decompress_iters {
        let t = Instant::now();
        decoded = compressed.decompress_all();
        best_decompress = best_decompress.min(t.elapsed());
    }

    let roundtrip_ok = decoded.len() == strings.len()
        && decoded.iter().zip(strings.iter()).all(|(a, b)| a == b);

    // Pushdown: pick the first needle so the timings are comparable across
    // datasets. Each pushdown shape is timed independently.
    let mut equality_pushdown = None;
    let mut contains_pushdown = None;
    let mut starts_with_pushdown = None;
    let mut equality_hits = 0;
    let mut contains_hits = 0;
    let mut starts_with_hits = 0;

    if !needles.is_empty() {
        let needle = &needles[0];
        equality_hits = compressed.equals(needle).len();
        contains_hits = compressed.contains(needle).len();
        starts_with_hits = compressed.starts_with(needle).len();

        let mut best_eq = Duration::MAX;
        for _ in 0..opts.pushdown_iters {
            let t = Instant::now();
            drop(compressed.equals(needle));
            best_eq = best_eq.min(t.elapsed());
        }
        equality_pushdown = Some(best_eq);

        let mut best_contains = Duration::MAX;
        for _ in 0..opts.pushdown_iters {
            let t = Instant::now();
            drop(compressed.contains(needle));
            best_contains = best_contains.min(t.elapsed());
        }
        contains_pushdown = Some(best_contains);

        let mut best_sw = Duration::MAX;
        for _ in 0..opts.pushdown_iters {
            let t = Instant::now();
            drop(compressed.starts_with(needle));
            best_sw = best_sw.min(t.elapsed());
        }
        starts_with_pushdown = Some(best_sw);
    }

    BackendResult {
        backend: kind.name(),
        dataset: dataset_name,
        rows: strings.len(),
        uncompressed_bytes: uncompressed,
        compressed_payload_bytes: compressed.compressed_payload_bytes(),
        total_compressed_bytes: compressed.total_compressed_bytes(),
        compress: best_compress,
        decompress: best_decompress,
        equality_pushdown,
        contains_pushdown,
        starts_with_pushdown,
        equality_is_compressed_domain: kind.supports_compressed_equality(),
        substring_is_compressed_domain: kind.supports_compressed_substring(),
        roundtrip_ok,
        equality_hits,
        contains_hits,
        starts_with_hits,
    }
}
