// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_ptr_alignment,
    clippy::clone_on_ref_ptr,
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::missing_safety_doc,
    clippy::print_stdout,
    clippy::unwrap_used
)]
//
// B2 de-risking prototype. The OnPair parse hot path is bound by the latency of
// a single hashbrown probe into the `long_map` (an 8-byte prefix → bucket
// lookup) whose working set exceeds L2. The proposed win (PERFORMANCE.md idea
// B2) is to keep many *independent* probes in flight so the memory subsystem
// overlaps their miss latency (memory-level parallelism), via an AVX-512 gather.
//
// Before rewriting the parser, this isolates the core question: on THIS machine,
// does an AVX-512 masked-gather lookup over a custom open-addressing table beat
// serial scalar probes for the same key stream? We build the table from the
// REAL `l_comment` long-prefix set and probe it with REAL corpus windows.
//
//   ONPAIR_BENCH_PARQUET=target/l_comment.parquet ONPAIR_BENCH_COLUMN=l_comment \
//     cargo run --release -p vortex-onpair-rs --example bench_gather
//
// Env: ONPAIR_BENCH_MAX_BYTES (corpus cap, default 256 MiB),
//      PROBES (number of probe keys, default 20_000_000),
//      ONPAIR_BENCH_ITERS (timed iters, default 5).

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use hashbrown::HashMap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_onpair_rs::OnPairTrainingConfig;
use vortex_onpair_rs::TrainingConfig;
use vortex_onpair_rs::train;

const EMPTY: u64 = u64::MAX;
const FIB: u64 = 0x9E37_79B9_7F4A_7C15;

/// Open-addressing (linear-probe) table, gather-friendly: separate `u64` key and
/// descriptor arrays so a single `vpgatherqq` fetches 8 slots in parallel.
struct FlatTable {
    keys: Vec<u64>,
    desc: Vec<u64>,
    mask: u64,
    log: u32,
}

impl FlatTable {
    fn new(n: usize) -> Self {
        let cap = (2 * n.max(1)).next_power_of_two();
        Self {
            keys: vec![EMPTY; cap],
            desc: vec![0u64; cap],
            mask: cap as u64 - 1,
            log: cap.trailing_zeros(),
        }
    }

    #[inline]
    fn home(&self, key: u64) -> usize {
        (key.wrapping_mul(FIB) >> (64 - self.log)) as usize
    }

    fn insert(&mut self, key: u64, d: u64) {
        let mut i = self.home(key);
        loop {
            if self.keys[i] == EMPTY {
                self.keys[i] = key;
                self.desc[i] = d;
                return;
            }
            if self.keys[i] == key {
                return;
            }
            i = (i + 1) & self.mask as usize;
        }
    }

    #[inline]
    fn get_scalar(&self, key: u64) -> Option<u64> {
        let mut i = self.home(key);
        loop {
            let k = self.keys[i];
            if k == key {
                return Some(self.desc[i]);
            }
            if k == EMPTY {
                return None;
            }
            i = (i + 1) & self.mask as usize;
        }
    }
}

fn main() {
    let max_bytes = env::var("ONPAIR_BENCH_MAX_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256 << 20);
    let n_probes = env::var("PROBES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20_000_000);
    let iters = env::var("ONPAIR_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);

    // Synthetic large-table mode: isolate the pure hardware question — does an
    // AVX-512 gather overlap DRAM-miss latency? Build a table far bigger than L3
    // with random keys and random probes, so (almost) every probe misses to DRAM.
    if let Some(syn) = env::var("SYN_KEYS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        && syn > 0
    {
        synthetic_latency_test(syn, n_probes, iters);
        return;
    }

    let (bytes, offsets) = load_corpus(max_bytes).expect("set ONPAIR_BENCH_PARQUET to l_comment");
    let n = offsets.len() - 1;
    let off32: Vec<u32> = offsets.iter().map(|&o| o as u32).collect();
    println!(
        "corpus {:.1} MiB, {n} rows",
        bytes.len() as f64 / (1024.0 * 1024.0)
    );

    // Train bits=16 to get a realistic dictionary, then extract the distinct
    // 8-byte prefixes of every long (9..=16 byte) token = the long_map key set.
    let cfg = TrainingConfig::from(OnPairTrainingConfig {
        bits: 16,
        threshold: 0.2,
        seed: 42,
    });
    let result = train(&bytes, &off32, n, &cfg);
    let dict = &result.dict;
    let mut prefix_set: HashMap<u64, u64> = HashMap::new();
    for i in 0..dict.num_tokens() {
        let tok = dict.data(i as u16);
        if tok.len() > 8 {
            let p = u64::from_le_bytes(tok[..8].try_into().unwrap());
            if p != EMPTY {
                let next = prefix_set.len() as u64;
                prefix_set.entry(p).or_insert(next);
            }
        }
    }
    let prefixes: Vec<(u64, u64)> = prefix_set.iter().map(|(&k, &v)| (k, v)).collect();
    println!("distinct long prefixes = {}", prefixes.len());

    // hashbrown reference + custom flat table, same contents.
    let mut hb: HashMap<u64, u64> = HashMap::with_capacity(prefixes.len() * 2);
    let mut ft = FlatTable::new(prefixes.len());
    for &(k, d) in &prefixes {
        hb.insert(k, d);
        ft.insert(k, d);
    }
    println!(
        "flat table cap = {} (load {:.2})",
        ft.keys.len(),
        prefixes.len() as f64 / ft.keys.len() as f64
    );

    // Probe-key stream: real 8-byte corpus windows, strided to sample the whole
    // corpus. The table accesses scatter by hash regardless of stream order, so
    // this reproduces the parse probe's cache/MLP behavior.
    let windows = bytes.len().saturating_sub(8);
    let stride = (windows / n_probes).max(1);
    let mut probes: Vec<u64> = Vec::with_capacity(n_probes);
    let mut p = 0usize;
    while p < windows && probes.len() < n_probes {
        probes.push(u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()));
        p += stride;
    }
    // Pad to a multiple of 8 for the gather kernel.
    while !probes.len().is_multiple_of(8) {
        probes.push(probes[probes.len() - 1]);
    }
    let np = probes.len();
    println!("probes = {np}");

    let mib = (np * 8) as f64 / (1024.0 * 1024.0);
    let mut hits_ref = (0u64, 0u64);

    // (a) serial hashbrown
    let mut secs_hb = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let (mut sum, mut hits) = (0u64, 0u64);
        for &k in &probes {
            if let Some(&d) = hb.get(&k) {
                sum = sum.wrapping_add(d);
                hits += 1;
            }
        }
        secs_hb = secs_hb.min(t.elapsed().as_secs_f64());
        hits_ref = (sum, hits);
    }

    // (b) serial custom table
    let mut secs_ft = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let (mut sum, mut hits) = (0u64, 0u64);
        for &k in &probes {
            if let Some(d) = ft.get_scalar(k) {
                sum = sum.wrapping_add(d);
                hits += 1;
            }
        }
        secs_ft = secs_ft.min(t.elapsed().as_secs_f64());
        assert_eq!((sum, hits), hits_ref, "scalar flat table disagrees");
    }

    println!(
        "\n(a) hashbrown serial : {:.3}s  {:.1} Mprobe/s  ({:.2} ns/probe)",
        secs_hb,
        np as f64 / secs_hb / 1e6,
        secs_hb / np as f64 * 1e9
    );
    println!(
        "(b) flat     serial : {:.3}s  {:.1} Mprobe/s  ({:.2} ns/probe)",
        secs_ft,
        np as f64 / secs_ft / 1e6,
        secs_ft / np as f64 * 1e9
    );
    let _ = mib;

    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
        let mut secs_g = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            let got = unsafe { gather_probe(&ft, &probes) };
            secs_g = secs_g.min(t.elapsed().as_secs_f64());
            assert_eq!(got, hits_ref, "gather kernel disagrees");
        }
        println!(
            "(c) flat   gather512: {:.3}s  {:.1} Mprobe/s  ({:.2} ns/probe)",
            secs_g,
            np as f64 / secs_g / 1e6,
            secs_g / np as f64 * 1e9
        );
        println!(
            "\n  gather vs hashbrown-serial: {:.2}x   gather vs flat-serial: {:.2}x",
            secs_hb / secs_g,
            secs_ft / secs_g
        );
    } else {
        println!("(c) AVX-512 not available on this CPU; skipping gather kernel");
    }
}

/// Pure hardware test: a table far larger than L3 with random keys, probed by a
/// random hit/miss stream so nearly every lookup is a DRAM miss. Compares serial
/// scalar vs AVX-512 gather to measure whether the gather overlaps miss latency.
fn synthetic_latency_test(syn_keys: usize, n_probes: usize, iters: usize) {
    let mut x = 0x1234_5678_9abc_def0u64;
    let mut rng = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };

    let mut ft = FlatTable::new(syn_keys);
    let mut hb: HashMap<u64, u64> = HashMap::with_capacity(syn_keys * 2);
    let mut inserted: Vec<u64> = Vec::with_capacity(syn_keys);
    for d in 0..syn_keys {
        let k = rng() & !(1u64 << 63); // keep != EMPTY (u64::MAX)
        let k = if k == EMPTY { 1 } else { k };
        inserted.push(k);
        ft.insert(k, d as u64);
        hb.insert(k, d as u64);
    }
    let table_mib = (ft.keys.len() * 16) as f64 / (1024.0 * 1024.0);
    println!(
        "synthetic: {} keys, table cap {} = {:.0} MiB (>L3 ⇒ DRAM), load {:.2}",
        syn_keys,
        ft.keys.len(),
        table_mib,
        syn_keys as f64 / ft.keys.len() as f64
    );

    // ~50% hits: half the probes are real keys, half random (mostly miss).
    let mut probes: Vec<u64> = Vec::with_capacity(n_probes);
    for j in 0..n_probes {
        if j & 1 == 0 {
            probes.push(inserted[(rng() as usize) % inserted.len()]);
        } else {
            let k = rng() & !(1u64 << 63);
            probes.push(if k == EMPTY { 1 } else { k });
        }
    }
    while !probes.len().is_multiple_of(8) {
        probes.push(probes[probes.len() - 1]);
    }
    let np = probes.len();
    println!("probes = {np}");

    let mut hits_ref = (0u64, 0u64);
    let mut secs_hb = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let (mut sum, mut hits) = (0u64, 0u64);
        for &k in &probes {
            if let Some(&d) = hb.get(&k) {
                sum = sum.wrapping_add(d);
                hits += 1;
            }
        }
        secs_hb = secs_hb.min(t.elapsed().as_secs_f64());
        hits_ref = (sum, hits);
    }
    let mut secs_ft = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        let (mut sum, mut hits) = (0u64, 0u64);
        for &k in &probes {
            if let Some(d) = ft.get_scalar(k) {
                sum = sum.wrapping_add(d);
                hits += 1;
            }
        }
        secs_ft = secs_ft.min(t.elapsed().as_secs_f64());
        assert_eq!((sum, hits), hits_ref);
    }
    println!(
        "\n(a) hashbrown serial : {:.2} ns/probe  ({:.1} Mprobe/s)",
        secs_hb / np as f64 * 1e9,
        np as f64 / secs_hb / 1e6
    );
    println!(
        "(b) flat     serial : {:.2} ns/probe  ({:.1} Mprobe/s)",
        secs_ft / np as f64 * 1e9,
        np as f64 / secs_ft / 1e6
    );
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
        let mut secs_g = f64::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            let got = unsafe { gather_probe(&ft, &probes) };
            secs_g = secs_g.min(t.elapsed().as_secs_f64());
            assert_eq!(got, hits_ref, "gather kernel disagrees");
        }
        println!(
            "(c) flat   gather512: {:.2} ns/probe  ({:.1} Mprobe/s)",
            secs_g / np as f64 * 1e9,
            np as f64 / secs_g / 1e6
        );
        println!(
            "\n  gather vs hashbrown-serial: {:.2}x   gather vs flat-serial: {:.2}x",
            secs_hb / secs_g,
            secs_ft / secs_g
        );
    }
}

/// AVX-512 masked-gather probe: process 8 independent keys per iteration, issuing
/// 8 table loads in one `vpgatherqq` so the memory subsystem overlaps the misses.
/// Returns (desc-sum, hit-count) to match the scalar paths exactly.
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn gather_probe(ft: &FlatTable, probes: &[u64]) -> (u64, u64) {
    use std::arch::x86_64::*;
    unsafe {
        const MAXSTEP: usize = 16;
        let keys_ptr = ft.keys.as_ptr() as *const i64;
        let desc_ptr = ft.desc.as_ptr() as *const i64;
        let maskv = _mm512_set1_epi64(ft.mask as i64);
        let fibv = _mm512_set1_epi64(FIB as i64);
        let emptyv = _mm512_set1_epi64(EMPTY as i64);
        let one = _mm512_set1_epi64(1);
        let shift = 64 - ft.log;

        let mut sum: u64 = 0;
        let mut hits: u64 = 0;

        let mut i = 0;
        while i + 8 <= probes.len() {
            let k = _mm512_loadu_si512(probes.as_ptr().add(i) as *const __m512i);
            // home = (k * FIB) >> (64 - log)
            let prod = _mm512_mullo_epi64(k, fibv);
            let mut idx = _mm512_srlv_epi64(prod, _mm512_set1_epi64(shift as i64));
            idx = _mm512_and_si512(idx, maskv);

            let mut active: __mmask8 = 0xFF;
            let mut step = 0;
            while active != 0 && step < MAXSTEP {
                // gather table keys at idx for active lanes (inactive -> EMPTY)
                let tk = _mm512_mask_i64gather_epi64::<8>(emptyv, active, idx, keys_ptr);
                let eq = _mm512_mask_cmpeq_epi64_mask(active, tk, k);
                let emp = _mm512_mask_cmpeq_epi64_mask(active, tk, emptyv);
                if eq != 0 {
                    let td =
                        _mm512_mask_i64gather_epi64::<8>(_mm512_setzero_si512(), eq, idx, desc_ptr);
                    sum = sum.wrapping_add(_mm512_reduce_add_epi64(td) as u64);
                    hits += (eq as u8).count_ones() as u64;
                }
                active &= !(eq | emp);
                idx = _mm512_and_si512(_mm512_add_epi64(idx, one), maskv);
                step += 1;
            }
            // scalar fallback for any lane that exceeded MAXSTEP probes (rare).
            if active != 0 {
                let mut lanes = [0u64; 8];
                _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, k);
                for lane in 0..8 {
                    if active & (1 << lane) != 0
                        && let Some(d) = ft.get_scalar(lanes[lane])
                    {
                        sum = sum.wrapping_add(d);
                        hits += 1;
                    }
                }
            }
            i += 8;
        }
        (sum, hits)
    }
}

fn load_corpus(max_bytes: usize) -> Option<(Vec<u8>, Vec<u64>)> {
    let path = env::var("ONPAIR_BENCH_PARQUET").ok()?;
    let file = File::open(PathBuf::from(&path)).ok()?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
    let schema = builder.schema().clone();
    let col_name = env::var("ONPAIR_BENCH_COLUMN").ok();
    let picked = match col_name.as_deref() {
        Some(name) => schema.fields().iter().position(|f| f.name() == name)?,
        None => schema.fields().iter().position(|f| {
            use arrow_schema::DataType::*;
            matches!(f.data_type(), Utf8 | LargeUtf8 | Utf8View)
        })?,
    };
    let mut bytes = Vec::new();
    let mut offsets = vec![0u64];
    let reader = builder.build().ok()?;
    'outer: for batch in reader.flatten() {
        let arr = batch.column(picked);
        use arrow_schema::DataType::*;
        macro_rules! push_iter {
            ($it:expr) => {
                for s in $it {
                    let b = s.unwrap_or("").as_bytes();
                    bytes.extend_from_slice(b);
                    offsets.push(bytes.len() as u64);
                    if bytes.len() >= max_bytes {
                        break 'outer;
                    }
                }
            };
        }
        match arr.data_type() {
            Utf8 => push_iter!(arr.as_string::<i32>().iter()),
            LargeUtf8 => push_iter!(arr.as_string::<i64>().iter()),
            Utf8View => push_iter!(arr.as_string_view().iter()),
            _ => return None,
        }
    }
    Some((bytes, offsets))
}
