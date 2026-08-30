//! Decompression benchmark for the two Rust OnPair implementations:
//! `spiraldb/onpair` (the one Vortex uses) and `onpair_rs` (the paper's own).
//!
//! Measures bulk decode of a whole column and random-access decode of rows in
//! shuffled order, and verifies every decode against the original bytes.
//!
//! usage: decode-rs DATASET.txt [bits] [iterations]

use onpair::DictionaryView;
use std::mem::MaybeUninit;
use std::time::Instant;

const MAX_TOKEN: usize = 16;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

/// Deterministic shuffle, so every implementation sees the same access order.
fn shuffled(rows: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows).collect();
    let mut state = 42u64;
    for i in (1..rows).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        order.swap(i, (state >> 33) as usize % (i + 1));
    }
    order
}

fn report(
    name: &str,
    dataset: &str,
    bits: u32,
    rows: usize,
    bytes: usize,
    compressed: usize,
    bulk_ms: f64,
    random_ms: f64,
    bulk_ok: bool,
    random_ok: bool,
) {
    let mib = bytes as f64 / 1048576.0;
    println!(
        "decode,impl={name},dataset={dataset},bits={bits},rows={rows},mib={mib:.2},\
         compressed_mib={:.2},bulk_ms={bulk_ms:.2},bulk_mibs={:.1},random_ms={random_ms:.2},\
         random_ns_per_row={:.1},bulk_ok={bulk_ok},random_ok={random_ok}",
        compressed as f64 / 1048576.0,
        mib / (bulk_ms / 1000.0),
        random_ms * 1e6 / rows as f64,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dataset = args.get(1).expect("usage: decode-rs DATASET.txt [bits] [iters]");
    let bits: u8 = args.get(2).map_or(16, |a| a.parse().unwrap());
    let iterations: usize = args.get(3).map_or(5, |a| a.parse().unwrap());

    let text = std::fs::read(dataset).expect("cannot read dataset");
    let mut data: Vec<u8> = Vec::with_capacity(text.len());
    let mut offsets: Vec<u32> = vec![0];
    for line in text.split(|&b| b == b'\n') {
        if line.is_empty() && data.len() == text.len() {
            continue;
        }
        data.extend_from_slice(line);
        offsets.push(data.len() as u32);
    }
    offsets.pop();
    let rows = offsets.len() - 1;
    data.truncate(offsets[rows] as usize);
    let order = shuffled(rows);

    // ── spiraldb/onpair ──────────────────────────────────────────────────────
    {
        let cfg = onpair::Config {
            max_dict_bits: onpair::MaxDictBits::new(bits).unwrap(),
            threshold: onpair::Threshold::new(0.15).unwrap(),
            seed: Some(42),
        };
        let column = onpair::compress(&data, &offsets, cfg).unwrap();
        let view = column.view();

        let mut out: Vec<MaybeUninit<u8>> =
            vec![MaybeUninit::uninit(); view.decoded_len() + onpair::DECODE_PADDING];
        let mut bulk = Vec::new();
        let mut written = 0;
        for _ in 0..iterations {
            let start = Instant::now();
            written = unsafe { view.decompress_into(&mut out) };
            bulk.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let decoded: &[u8] =
            unsafe { std::slice::from_raw_parts(out.as_ptr().cast::<u8>(), written) };
        let bulk_ok = written == data.len() && decoded == &data[..];

        let mut row: Vec<MaybeUninit<u8>> = vec![MaybeUninit::uninit(); 64 * 1024 + onpair::DECODE_PADDING];
        let mut random = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            let mut total = 0usize;
            for &k in &order {
                total += unsafe { view.decompress_row_into(k, &mut row) };
            }
            std::hint::black_box(total);
            random.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let mut random_ok = true;
        for &k in &order {
            let n = unsafe { view.decompress_row_into(k, &mut row) };
            let got: &[u8] = unsafe { std::slice::from_raw_parts(row.as_ptr().cast::<u8>(), n) };
            let expected = &data[offsets[k] as usize..offsets[k + 1] as usize];
            if got != expected {
                random_ok = false;
                break;
            }
        }

        // Match the C++ accounting: bit-packed code stream + dictionary + row layer.
        let code_bits = onpair::code_bits_for_num_tokens(view.dict.num_tokens()) as usize;
        let compressed = (view.codes.len() * code_bits).div_ceil(8)
            + {
                // Sum of token lengths: decoded_len over every token id once.
                let ids: Vec<onpair::Token> = (0..view.dict.num_tokens() as u32).map(|t| t as onpair::Token).collect();
                onpair::decoded_len(&ids, view.dict) + (view.dict.num_tokens() + 1) * 4
            }
            + view.row_offsets.len() * 4;
        report("spiraldb_onpair", dataset, bits as u32, rows, data.len(), compressed,
               median(bulk), median(random), bulk_ok, random_ok);
    }

    // ── onpair_rs (the paper's implementation) ───────────────────────────────
    // Its dictionary is fixed at 65,536 tokens, so only bits == 16 is comparable.
    if bits == 16 {
        let ends: Vec<usize> = offsets.iter().map(|&o| o as usize).collect();
        for variant in ["onpair_rs_OnPair16", "onpair_rs_OnPair"] {
            let mut bulk = Vec::new();
            let mut random = Vec::new();
            let mut out = vec![0u8; data.len() + MAX_TOKEN * 2];
            let mut row = vec![0u8; 64 * 1024 + MAX_TOKEN * 2];
            let (mut bulk_ok, mut random_ok, mut compressed) = (false, true, 0usize);

            macro_rules! run {
                ($compressor:expr) => {{
                    let mut c = $compressor;
                    c.compress_bytes(&data, &ends);
                    compressed = c.space_used();
                    let mut written = 0;
                    for _ in 0..iterations {
                        let start = Instant::now();
                        written = c.decompress_all(&mut out);
                        bulk.push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    bulk_ok = written == data.len() && out[..written] == data[..];
                    for _ in 0..iterations {
                        let start = Instant::now();
                        let mut total = 0usize;
                        for &k in &order {
                            total += c.decompress_string(k, &mut row);
                        }
                        std::hint::black_box(total);
                        random.push(start.elapsed().as_secs_f64() * 1000.0);
                    }
                    for &k in &order {
                        let n = c.decompress_string(k, &mut row);
                        if row[..n] != data[offsets[k] as usize..offsets[k + 1] as usize] {
                            random_ok = false;
                            break;
                        }
                    }
                }};
            }

            if variant == "onpair_rs_OnPair16" {
                run!(onpair_rs::OnPair16::new(5));
            } else {
                run!(onpair_rs::OnPair::new(5));
            }
            report(variant, dataset, bits as u32, rows, data.len(), compressed,
                   median(bulk), median(random), bulk_ok, random_ok);
        }
    }
}
