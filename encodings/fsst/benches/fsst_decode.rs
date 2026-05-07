// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark for the bulk FSST decoder.
//!
//! Compares the upstream `fsst::Decompressor::decompress_into` against the
//! local `vortex-fsst` `Decoder::decompress_into`. Both decoders see identical
//! inputs (compressed bytes plus the symbol table), so the only thing the
//! benchmark exercises is the inner-loop throughput.

use std::mem::MaybeUninit;

use divan::Bencher;
use divan::counter::BytesCount;
use fsst::Compressor;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_fsst::bench_decoder::Decoder;

fn main() {
    divan::main();
}

// (avg_string_len, unique_chars, total_input_bytes)
const ARGS: &[(usize, u8, usize)] = &[
    (16, 8, 1 << 20),   // 1 MiB,    short strings, low entropy
    (64, 8, 1 << 20),   // 1 MiB,    medium strings
    (16, 26, 1 << 20),  // 1 MiB,    full lowercase alphabet
    (64, 26, 4 << 20),  // 4 MiB,    closer to canonicalize_into workload
    (64, 26, 16 << 20), // 16 MiB,   pushes past L2
];

struct Workload {
    compressor: Compressor,
    compressed: Vec<u8>,
    decompressed_len: usize,
}

fn build_workload(avg_len: usize, unique_chars: u8, total_bytes: usize) -> Workload {
    let mut rng = StdRng::seed_from_u64(0);
    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut size = 0;
    while size < total_bytes {
        let len = avg_len * rng.random_range(50..=150) / 100;
        let s: Vec<u8> = (0..len)
            .map(|_| rng.random_range(b'a'..(b'a' + unique_chars)))
            .collect();
        size += s.len();
        samples.push(s);
    }

    let refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
    let compressor = Compressor::train(&refs);

    // Compress every sample into one contiguous buffer so the decoder operates
    // on a realistic 1 MiB+ block, not many tiny chunks.
    let mut compressed = Vec::with_capacity(total_bytes);
    for s in &samples {
        let mut buf = Vec::with_capacity(s.len() * 2 + 8);
        // SAFETY: pre-sized buffer to upstream's worst case.
        unsafe { compressor.compress_into(s, &mut buf) };
        compressed.extend_from_slice(&buf);
    }

    let decompressed_len: usize = samples.iter().map(|s| s.len()).sum();
    Workload {
        compressor,
        compressed,
        decompressed_len,
    }
}

#[divan::bench(args = ARGS)]
fn upstream(bencher: Bencher, args: (usize, u8, usize)) {
    let (avg_len, uniq, total) = args;
    let wl = build_workload(avg_len, uniq, total);
    let decomp = wl.compressor.decompressor();
    let mut out: Vec<MaybeUninit<u8>> = Vec::with_capacity(wl.decompressed_len + 64);
    out.resize(out.capacity(), MaybeUninit::uninit());

    bencher
        .counter(BytesCount::new(wl.decompressed_len))
        .bench_local(|| {
            let n = decomp.decompress_into(&wl.compressed, &mut out[..]);
            divan::black_box(n);
        });
}

#[divan::bench(args = ARGS)]
fn local(bencher: Bencher, args: (usize, u8, usize)) {
    let (avg_len, uniq, total) = args;
    let wl = build_workload(avg_len, uniq, total);
    let decoder = Decoder::new(wl.compressor.symbol_table(), wl.compressor.symbol_lengths());
    let mut out: Vec<MaybeUninit<u8>> = Vec::with_capacity(wl.decompressed_len + 64);
    out.resize(out.capacity(), MaybeUninit::uninit());

    bencher
        .counter(BytesCount::new(wl.decompressed_len))
        .bench_local(|| {
            let n = decoder.decompress_into(&wl.compressed, &mut out[..]);
            divan::black_box(n);
        });
}
