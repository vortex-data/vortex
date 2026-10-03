// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The smallest compact Vortex encoding of a label map found by a search over shredding options
//! and map layouts, against Parquet zstd at levels 3 and 22.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Instant;

use arrow_array::Array as _;
use arrow_array::RecordBatch;
use mimalloc::MiMalloc;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_shredded_map::ShredOptions;
use vortex_shredded_map::keyset::KeySetOptions;
use vortex_shredded_map::ops;

#[path = "../benches/common/mod.rs"]
mod common;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn parquet(map: &arrow_array::MapArray, level: i32) -> usize {
    let batch =
        RecordBatch::try_from_iter([("labels", Arc::new(map.clone()) as arrow_array::ArrayRef)])
            .unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(level).unwrap()))
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    buf.len()
}

/// Every row of `got` equals the same row of `expected`.
fn same_rows(expected: &ArrayRef, got: &ArrayRef) -> bool {
    let mut ctx = common::SESSION.create_execution_ctx();
    let expected = ops::encoded::to_utf8_map(expected, &mut ctx).unwrap().into_array();
    let got = ops::encoded::to_utf8_map(got, &mut ctx).unwrap().into_array();
    expected.len() == got.len()
        && (0..expected.len()).all(|i| {
            expected.execute_scalar(i, &mut ctx).unwrap() == got.execute_scalar(i, &mut ctx).unwrap()
        })
}

struct Trial {
    name: String,
    bytes: u64,
    array: ArrayRef,
}

fn main() {
    let data = LazyLock::force(&common::DATA);
    let mut ctx = common::SESSION.create_execution_ctx();
    let map = &data.map;
    let pq3 = parquet(&data.arrow, 3);
    let pq22 = parquet(&data.arrow, 22);

    let mut trials: Vec<Trial> = Vec::new();
    let mut run = |name: String, encoded: ArrayRef, trials: &mut Vec<Trial>| {
        let t = Instant::now();
        let array = common::compress_encoded(&encoded, true);
        let bytes = array.nbytes();
        println!("  {name:<58} {:>12} B  ({:.1}s)", bytes, t.elapsed().as_secs_f64());
        trials.push(Trial { name, bytes, array });
    };

    run("map".into(), map.clone(), &mut trials);
    let keyset = vortex_shredded_map::keyset::keyset_encode(
        map,
        KeySetOptions { dedup_values: true },
        &mut ctx,
    )
    .unwrap()
    .into_array();
    let t = Instant::now();
    let keyset = common::compress_children(&keyset, true);
    println!("  {:<58} {:>12} B  ({:.1}s)", "keyset", keyset.nbytes(), t.elapsed().as_secs_f64());
    trials.push(Trial { name: "keyset".into(), bytes: keyset.nbytes(), array: keyset });

    // One option at a time from the defaults, each shredded directly and with a row dictionary.
    let base = ShredOptions::default();
    let mut variants: Vec<(String, ShredOptions)> = vec![("default".into(), base.clone())];
    for v in [0.0005, 0.002, 0.03] {
        variants.push((format!("min_frequency={v}"), ShredOptions { min_frequency: v, ..base.clone() }));
    }
    for v in [0.5, 0.95] {
        variants.push((format!("sparse_below={v}"), ShredOptions { sparse_below: v, ..base.clone() }));
    }
    for v in [16, 256, 1024] {
        variants.push((format!("max_sparse_columns={v}"), ShredOptions { max_sparse_columns: v, ..base.clone() }));
    }
    variants.push(("dictionary=false".into(), ShredOptions { dictionary: false, ..base.clone() }));

    let shred_trial = |name: &str, opts: &ShredOptions, rows: bool, trials: &mut Vec<Trial>, run: &mut dyn FnMut(String, ArrayRef, &mut Vec<Trial>)| {
        let mut ctx = common::SESSION.create_execution_ctx();
        let (label, encoded) = if rows {
            let opts = ShredOptions { max_distinct_rows: 1.0, ..opts.clone() };
            (format!("rowdict+shred {name}"), vortex_shredded_map::encode(map, &opts, &mut ctx).unwrap())
        } else {
            (format!("shred {name}"), vortex_shredded_map::shred(map, opts, &mut ctx).unwrap().into_array())
        };
        run(label, encoded, trials);
    };
    for rows in [false, true] {
        for (name, opts) in &variants {
            shred_trial(name, opts, rows, &mut trials, &mut run);
        }
    }

    // Combine the best value of each option, for the better of direct and row dictionary.
    let best_of = |prefix: &str, trials: &[Trial]| -> ShredOptions {
        let mut opts = base.clone();
        let size = |n: &str| trials.iter().find(|t| t.name == format!("{prefix} {n}")).map(|t| t.bytes);
        let pick = |names: &[String]| names.iter().min_by_key(|n| size(n).unwrap_or(u64::MAX)).cloned().unwrap();
        let min_f = pick(&["default".into(), "min_frequency=0.0005".into(), "min_frequency=0.002".into(), "min_frequency=0.03".into()]);
        if let Some(v) = min_f.strip_prefix("min_frequency=") { opts.min_frequency = v.parse().unwrap(); }
        let sb = pick(&["default".into(), "sparse_below=0.5".into(), "sparse_below=0.95".into()]);
        if let Some(v) = sb.strip_prefix("sparse_below=") { opts.sparse_below = v.parse().unwrap(); }
        let ns = pick(&["default".into(), "max_sparse_columns=16".into(), "max_sparse_columns=256".into(), "max_sparse_columns=1024".into()]);
        if let Some(v) = ns.strip_prefix("max_sparse_columns=") { opts.max_sparse_columns = v.parse().unwrap(); }
        let d = pick(&["default".into(), "dictionary=false".into()]);
        opts.dictionary = d == "default";
        opts
    };
    for (rows, prefix) in [(false, "shred"), (true, "rowdict+shred")] {
        let opts = best_of(prefix, &trials);
        let name = format!(
            "combined(min_frequency={}, sparse_below={}, max_sparse_columns={}, dictionary={})",
            opts.min_frequency, opts.sparse_below, opts.max_sparse_columns, opts.dictionary
        );
        shred_trial(&name, &opts, rows, &mut trials, &mut run);
    }

    let t = Instant::now();
    let auto = common::compress_auto(map, true);
    println!("  {:<58} {:>12} B  ({:.1}s)", "btrblocks auto pick", auto.nbytes(), t.elapsed().as_secs_f64());
    trials.push(Trial { name: "btrblocks auto pick".into(), bytes: auto.nbytes(), array: auto });

    // Squeeze the smallest layouts with high-level zstd.
    let level: i32 = std::env::var("SQUEEZE_LEVEL").ok().and_then(|v| v.parse().ok()).unwrap_or(19);
    let mut ranked: Vec<usize> = (0..trials.len()).filter(|&i| trials[i].name != "map").collect();
    ranked.sort_by_key(|&i| trials[i].bytes);
    let mut picks: Vec<usize> = ranked.into_iter().take(3).collect();
    if let Some(i) = trials.iter().position(|t| t.name == "shred default") && !picks.contains(&i) {
        picks.push(i);
    }
    for i in picks {
        let t = Instant::now();
        let array = vortex_shredded_map::squeeze::squeeze(&trials[i].array, level, &mut ctx).unwrap();
        let name = format!("squeezed(zstd {level}) {}", trials[i].name);
        println!("  {name:<58} {:>12} B  ({:.1}s)", array.nbytes(), t.elapsed().as_secs_f64());
        trials.push(Trial { name, bytes: array.nbytes(), array });
    }

    let best = trials.iter().filter(|t| t.name != "map").min_by_key(|t| t.bytes).unwrap();
    let map_bytes = trials[0].bytes;
    let verified = same_rows(map, &best.array);
    println!(
        "BEST rows={} parquet3={pq3} parquet22={pq22} map_compact={map_bytes} best={} verified={verified} config={}",
        data.arrow.len(),
        best.bytes,
        best.name
    );
}
