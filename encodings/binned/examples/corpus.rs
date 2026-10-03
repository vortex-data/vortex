// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares vortex-binned against Pco (configured as `vortex-pco` uses it) on a corpus of real
//! columns: compressed size, full decode throughput, and single-value access latency.
//!
//! Usage: `cargo run --release -p vortex-binned --example corpus -- <corpus dir> [filter]`
//! where the directory holds `manifest.jsonl` and `data/<id>.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::use_debug)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use pco::ChunkConfig;
use pco::PagingSpec;
use pco::wrapped::FileCompressor;
use pco::wrapped::FileDecompressor;
use vortex_binned::Config;
use prost::Message;
use vortex_binned::Decoder;
use vortex_binned::Encoded;
use vortex_binned::Number;
use vortex_binned::Numeric;
use vortex_binned::compress;

const PCO_CHUNK: usize = 1 << 18;
const PCO_PAGE: usize = 8192;
const N_PROBES: usize = 2000;

#[derive(Default, Clone)]
struct Measure {
    describe: String,
    bytes: usize,
    compress: Duration,
    decode: Duration,
    probe_ns: f64,
}

#[derive(Clone)]
struct Row {
    id: String,
    domain: String,
    dtype: String,
    n: usize,
    raw: usize,
    ours: Measure,
    pco: Option<Measure>,
    mode: String,
}

fn best_of(mut f: impl FnMut()) -> Duration {
    let mut best = Duration::MAX;
    let mut total = Duration::ZERO;
    let mut runs = 0;
    while runs < 3 || (total < Duration::from_millis(200) && runs < 50) {
        let t = Instant::now();
        f();
        let e = t.elapsed();
        best = best.min(e);
        total += e;
        runs += 1;
    }
    best
}

fn probe_indices(n: usize) -> Vec<usize> {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    (0..N_PROBES)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        })
        .collect()
}

fn bits_eq<T: Number>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_latent() == y.to_latent())
}

fn describe<T: Numeric>(enc: &Encoded<T>) -> String {
    let stream = |s: &vortex_binned::stream::Stream<T::L>| {
        format!("d{}b{}", s.delta_order, s.bins.len())
    };
    let mode = match enc.mode {
        vortex_binned::Mode::Classic => "classic".to_string(),
        vortex_binned::Mode::IntMult(b) => format!("intmult({b})"),
        vortex_binned::Mode::FloatMult(b) => format!("floatmult({b:e})"),
        vortex_binned::Mode::FloatQuant(k) => format!("quant({k})"),
    };
    let lb = if enc.lookbacks.is_some() { " lookback" } else { "" };
    match &enc.secondary {
        Some(sec) => format!("{mode}{lb} {}+{}", stream(&enc.primary), stream(sec)),
        None => format!("{mode}{lb} {}", stream(&enc.primary)),
    }
}

fn run_ours<T: Numeric>(values: &[T]) -> (Measure, String) {
    let t = Instant::now();
    let enc = compress(values, &Config::default()).unwrap();
    let compress_time = t.elapsed();
    // Decode from the serialized form, so the size and the roundtrip are both of what a file
    // would hold.
    let (meta, buffers) = enc.to_parts();
    let bytes = meta.encoded_len() + buffers.iter().map(|b| b.len()).sum::<usize>();
    let enc = Arc::new(Encoded::<T>::from_parts(values.len(), &meta, &buffers).unwrap());
    let dec = Decoder::new(Arc::clone(&enc)).unwrap();
    assert!(bits_eq(&dec.decode(), values), "binned roundtrip failed");
    let decode = best_of(|| {
        black_box(Decoder::new(Arc::clone(&enc)).unwrap().decode());
    });
    let probes = probe_indices(values.len());
    for &i in probes.iter().take(50) {
        assert!(bits_eq(&[dec.get(i)], &[values[i]]));
    }
    let t = Instant::now();
    for &i in &probes {
        black_box(dec.get(i));
    }
    let probe_ns = t.elapsed().as_nanos() as f64 / N_PROBES as f64;
    (
        Measure {
            describe: String::new(),
            bytes,
            compress: compress_time,
            decode,
            probe_ns,
        },
        describe(&enc),
    )
}

struct PcoData {
    header: Vec<u8>,
    /// Mode and delta encoding of the first chunk.
    describe: String,
    chunks: Vec<(Vec<u8>, Vec<(Vec<u8>, usize)>)>,
}

impl PcoData {
    fn nbytes(&self) -> usize {
        self.header.len()
            + self
                .chunks
                .iter()
                .map(|(m, pages)| m.len() + pages.iter().map(|(p, _)| p.len()).sum::<usize>())
                .sum::<usize>()
    }
}

fn pco_compress<T: pco::data_types::Number>(values: &[T]) -> PcoData {
    let config = ChunkConfig::default()
        .with_compression_level(pco::DEFAULT_COMPRESSION_LEVEL)
        .with_paging_spec(PagingSpec::EqualPagesUpTo(PCO_PAGE));
    let fc = FileCompressor::default();
    let mut header = Vec::new();
    fc.write_header(&mut header).unwrap();
    let mut chunks = Vec::new();
    let mut describe = String::new();
    for chunk in values.chunks(PCO_CHUNK) {
        let mut cc = fc.chunk_compressor(chunk, &config).unwrap();
        if describe.is_empty() {
            let short = |s: String| s.split([' ', '(', '{']).next().unwrap_or("").to_string();
            describe = format!(
                "{}/{}",
                short(format!("{:?}", cc.meta().mode)),
                short(format!("{:?}", cc.meta().delta_encoding))
            );
        }
        let mut meta = Vec::new();
        cc.write_meta(&mut meta).unwrap();
        let mut pages = Vec::new();
        for (i, n) in cc.n_per_page().into_iter().enumerate() {
            let mut page = Vec::new();
            cc.write_page(i, &mut page).unwrap();
            pages.push((page, n));
        }
        chunks.push((meta, pages));
    }
    PcoData {
        header,
        describe,
        chunks,
    }
}

fn pco_decode<T: pco::data_types::Number + Default>(data: &PcoData, n: usize) -> Vec<T> {
    let (fd, _) = FileDecompressor::new(data.header.as_slice()).unwrap();
    let mut out = vec![T::default(); n];
    let mut pos = 0;
    for (meta, pages) in &data.chunks {
        let (mut cd, _) = fd.chunk_decompressor::<T, _>(meta.as_slice()).unwrap();
        for (page, page_n) in pages {
            let mut pd = cd.page_decompressor(page.as_slice(), *page_n).unwrap();
            pd.read(&mut out[pos..pos + page_n]).unwrap();
            pos += page_n;
        }
    }
    out
}

/// What `vortex-pco` does to read one value: parse the chunk meta and decode its whole page.
fn pco_get<T: pco::data_types::Number + Default>(data: &PcoData, index: usize) -> T {
    let (fd, _) = FileDecompressor::new(data.header.as_slice()).unwrap();
    let chunk = index / PCO_CHUNK;
    let (meta, pages) = &data.chunks[chunk];
    let mut local = index % PCO_CHUNK;
    let (mut cd, _) = fd.chunk_decompressor::<T, _>(meta.as_slice()).unwrap();
    for (page, page_n) in pages {
        if local < *page_n {
            let mut buf = vec![T::default(); *page_n];
            let mut pd = cd.page_decompressor(page.as_slice(), *page_n).unwrap();
            pd.read(&mut buf).unwrap();
            return buf[local];
        }
        local -= page_n;
    }
    unreachable!()
}

fn run_pco<T: pco::data_types::Number + Numeric + Default>(values: &[T]) -> Measure {
    let t = Instant::now();
    let data = pco_compress(values);
    let compress_time = t.elapsed();
    assert!(bits_eq(&pco_decode::<T>(&data, values.len()), values));
    let decode = best_of(|| {
        black_box(pco_decode::<T>(&data, values.len()));
    });
    let probes = probe_indices(values.len());
    let t = Instant::now();
    for &i in &probes {
        black_box(pco_get::<T>(&data, i));
    }
    Measure {
        describe: data.describe.clone(),
        bytes: data.nbytes(),
        compress: compress_time,
        decode,
        probe_ns: t.elapsed().as_nanos() as f64 / N_PROBES as f64,
    }
}

fn parse<T: Copy, const N: usize>(bytes: &[u8], f: fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .map(|c| f(c.try_into().unwrap()))
        .collect()
}

macro_rules! bench {
    ($bytes:expr, $t:ty, $n:literal, pco) => {{
        let v = parse::<$t, $n>($bytes, <$t>::from_le_bytes);
        let (ours, mode) = run_ours(&v);
        (v.len(), ours, Some(run_pco(&v)), mode)
    }};
    ($bytes:expr, $t:ty, $n:literal) => {{
        let v = parse::<$t, $n>($bytes, <$t>::from_le_bytes);
        let (ours, mode) = run_ours(&v);
        (v.len(), ours, None, mode)
    }};
}

fn run_sample(dir: &Path, entry: &serde_json::Value) -> Option<Row> {
    let id = entry["id"].as_str()?.to_string();
    let dtype = entry["dtype"].as_str()?.to_string();
    let bytes = fs::read(dir.join("data").join(format!("{id}.bin"))).ok()?;
    if bytes.is_empty() {
        return None;
    }
    let (n, ours, pco, mode) = match dtype.as_str() {
        "u8" => bench!(&bytes, u8, 1),
        "i8" => bench!(&bytes, i8, 1),
        "u16" => bench!(&bytes, u16, 2, pco),
        "i16" => bench!(&bytes, i16, 2, pco),
        "u32" => bench!(&bytes, u32, 4, pco),
        "i32" => bench!(&bytes, i32, 4, pco),
        "u64" => bench!(&bytes, u64, 8, pco),
        "i64" => bench!(&bytes, i64, 8, pco),
        "f32" => bench!(&bytes, f32, 4, pco),
        "f64" => bench!(&bytes, f64, 8, pco),
        _ => return None,
    };
    Some(Row {
        id,
        domain: entry["domain"].as_str().unwrap_or("?").to_string(),
        dtype,
        n,
        raw: bytes.len(),
        ours,
        pco,
        mode,
    })
}

fn geomean(xs: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = xs.fold((0.0, 0usize), |(s, n), x| (s + x.ln(), n + 1));
    (sum / n.max(1) as f64).exp()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("usage: corpus <dir> [filter]"));
    let filter = args.get(2).cloned().unwrap_or_default();
    let manifest = fs::read_to_string(dir.join("manifest.jsonl")).expect("manifest.jsonl");

    println!(
        "id,domain,dtype,n,raw,ours_bytes,pco_bytes,size_vs_pco,ours_dec_gbps,pco_dec_gbps,ours_get_ns,pco_get_ns,ours_comp_ms,pco_comp_ms,mode,pco_mode"
    );
    let mut rows = Vec::new();
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let entry: serde_json::Value = serde_json::from_str(line).expect("manifest line");
        if !filter.is_empty() && !entry["id"].as_str().unwrap_or("").contains(&filter) {
            continue;
        }
        let Some(row) = run_sample(dir, &entry) else {
            continue;
        };
        let gbps = |d: Duration| row.raw as f64 / d.as_secs_f64() / 1e9;
        let p = row.pco.clone().unwrap_or_default();
        println!(
            "{},{},{},{},{},{},{},{:.3},{:.2},{:.2},{:.0},{:.0},{:.1},{:.1},{},{}",
            row.id,
            row.domain,
            row.dtype,
            row.n,
            row.raw,
            row.ours.bytes,
            p.bytes,
            if p.bytes > 0 { row.ours.bytes as f64 / p.bytes as f64 } else { f64::NAN },
            gbps(row.ours.decode),
            if row.pco.is_some() { gbps(p.decode) } else { f64::NAN },
            row.ours.probe_ns,
            p.probe_ns,
            row.ours.compress.as_secs_f64() * 1e3,
            p.compress.as_secs_f64() * 1e3,
            row.mode,
            p.describe,
        );
        rows.push(row);
    }

    let both: Vec<&Row> = rows.iter().filter(|r| r.pco.is_some()).collect();
    let pco = |r: &&Row| r.pco.clone().unwrap_or_default();
    let raw: usize = both.iter().map(|r| r.raw).sum();
    let ours: usize = both.iter().map(|r| r.ours.bytes).sum();
    let pco_bytes: usize = both.iter().map(|r| pco(r).bytes).sum();
    let ours_dec: f64 = both.iter().map(|r| r.ours.decode.as_secs_f64()).sum();
    let pco_dec: f64 = both.iter().map(|r| pco(r).decode.as_secs_f64()).sum();
    eprintln!("\n== {} samples ({} comparable with Pco) ==", rows.len(), both.len());
    eprintln!(
        "compression ratio (raw/compressed, aggregate): ours {:.3}  pco {:.3}",
        raw as f64 / ours as f64,
        raw as f64 / pco_bytes as f64
    );
    eprintln!(
        "size vs pco: geomean {:.4}  median {:.4}  better-or-equal on {}/{}",
        geomean(both.iter().map(|r| r.ours.bytes as f64 / pco(r).bytes as f64)),
        {
            let mut v: Vec<f64> = both
                .iter()
                .map(|r| r.ours.bytes as f64 / pco(r).bytes as f64)
                .collect();
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        },
        both.iter().filter(|r| r.ours.bytes <= pco(r).bytes).count(),
        both.len()
    );
    eprintln!(
        "full decode (aggregate GB/s of raw): ours {:.2}  pco {:.2}  | per-sample speedup geomean {:.2}x",
        raw as f64 / ours_dec / 1e9,
        raw as f64 / pco_dec / 1e9,
        geomean(both.iter().map(|r| pco(r).decode.as_secs_f64() / r.ours.decode.as_secs_f64()))
    );
    eprintln!(
        "single-value get (geomean ns): ours {:.0}  pco {:.0}",
        geomean(both.iter().map(|r| r.ours.probe_ns)),
        geomean(both.iter().map(|r| pco(r).probe_ns))
    );
    let mut worst: Vec<&&Row> = both.iter().collect();
    worst.sort_by(|a, b| {
        (b.ours.bytes as f64 / pco(b).bytes as f64)
            .total_cmp(&(a.ours.bytes as f64 / pco(a).bytes as f64))
    });
    eprintln!("worst size vs pco:");
    for r in worst.iter().take(12) {
        eprintln!(
            "  {:.3}  {} ({}, {}, {} | pco {})",
            r.ours.bytes as f64 / pco(r).bytes as f64,
            r.id,
            r.domain,
            r.dtype,
            r.mode,
            pco(r).describe
        );
    }
}
