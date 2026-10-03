// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `SELECT k1, k2, ... WHERE labels[filter_key] = value`: filter rows on one label, then project
//! a few labels of the matching rows, on Arrow and on each Vortex representation.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::sync::LazyLock;
use std::time::Instant;

use arrow_array::Array as _;
use arrow_array::StringArray;
use arrow_array::builder::StringBuilder;
use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinViewArray;
use vortex_shredded_map::ops::query;

#[path = "../benches/common/mod.rs"]
mod common;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn best<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut out = None;
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        out = Some(r);
    }
    (best, out.unwrap())
}

/// Row-at-a-time scan over the decoded Arrow map: find the filter key in each row, then the
/// projected keys in each matching row.
fn arrow_query(arrow: &arrow_array::MapArray, filter_key: &str, value: &str, keys: &[&str]) -> Vec<StringArray> {
    let k = arrow.keys().as_any().downcast_ref::<StringArray>().unwrap();
    let v = arrow.values().as_any().downcast_ref::<StringArray>().unwrap();
    let offsets = arrow.value_offsets();
    let mut builders: Vec<StringBuilder> = keys.iter().map(|_| StringBuilder::new()).collect();
    let find = |row: usize, key: &str| {
        (offsets[row] as usize..offsets[row + 1] as usize).find(|&j| k.value(j) == key)
    };
    for row in 0..arrow.len() {
        if arrow.is_null(row) {
            continue;
        }
        let hit = find(row, filter_key).is_some_and(|j| v.is_valid(j) && v.value(j) == value);
        if !hit {
            continue;
        }
        for (key, b) in keys.iter().zip(&mut builders) {
            match find(row, key) {
                Some(j) if v.is_valid(j) => b.append_value(v.value(j)),
                _ => b.append_null(),
            }
        }
    }
    builders.iter_mut().map(StringBuilder::finish).collect()
}

/// Bytes in an Arrow array's buffers, counting only the used length of each buffer.
fn arrow_data_bytes(data: &arrow_data::ArrayData) -> usize {
    data.buffers().iter().map(|b| b.len()).sum::<usize>()
        + data.nulls().map_or(0, |n| n.buffer().len())
        + data.child_data().iter().map(arrow_data_bytes).sum::<usize>()
}

/// Parquet zstd(3) bytes of the map column, and the time to read it back into Arrow.
fn parquet(map: &arrow_array::MapArray) -> (usize, f64) {
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::basic::Compression;
    use parquet::basic::ZstdLevel;
    use parquet::file::properties::WriterProperties;

    let batch = arrow_array::RecordBatch::try_from_iter([(
        "labels",
        std::sync::Arc::new(map.clone()) as arrow_array::ArrayRef,
    )])
    .unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let bytes = bytes::Bytes::from(buf);
    let (ms, rows) = best(|| {
        ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .with_batch_size(8192)
            .build()
            .unwrap()
            .map(|b| b.unwrap().num_rows())
            .sum::<usize>()
    });
    assert_eq!(rows, map.len());
    (bytes.len(), ms)
}

/// Queries chosen from the data alone, so every dataset gets the same kind of workload.
///
/// The filter key is the most common key with between 5 and 10,000 distinct values, filtered
/// on the values closest to 0.5%, 5% and 30% of rows. The projection takes the two most common
/// other keys, keys closest to 20% and 5% of rows, and the key closest to 0.3% of rows.
fn pick_queries(arrow: &arrow_array::MapArray) -> (String, Vec<String>) {
    use std::collections::HashMap;

    let k = arrow.keys().as_any().downcast_ref::<StringArray>().unwrap();
    let v = arrow.values().as_any().downcast_ref::<StringArray>().unwrap();
    let offsets = arrow.value_offsets();
    let rows = arrow.len();
    let mut key_rows: HashMap<&str, usize> = HashMap::new();
    let mut value_rows: HashMap<&str, HashMap<&str, usize>> = HashMap::new();
    for row in 0..rows {
        let mut seen: Vec<&str> = Vec::new();
        for j in offsets[row] as usize..offsets[row + 1] as usize {
            let key = k.value(j);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            *key_rows.entry(key).or_default() += 1;
            let values = value_rows.entry(key).or_default();
            // Stop tracking values of keys that are clearly not categorical.
            if values.len() <= 10_000 && v.is_valid(j) {
                *values.entry(v.value(j)).or_default() += 1;
            }
        }
    }
    let mut by_count: Vec<(&str, usize)> = key_rows.iter().map(|(k, c)| (*k, *c)).collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let filter_key = by_count
        .iter()
        .map(|(k, _)| *k)
        .find(|k| (5..=10_000).contains(&value_rows[k].len()))
        .or_else(|| by_count.first().map(|(k, _)| *k))
        .unwrap();
    let mut values: Vec<(&str, usize)> =
        value_rows[filter_key].iter().map(|(v, c)| (*v, *c)).collect();
    values.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let closest = |items: &[(&str, usize)], target: f64, skip: &[&str]| -> Option<String> {
        items
            .iter()
            .filter(|(x, _)| !skip.contains(x))
            .min_by(|a, b| {
                let da = (a.1 as f64 / rows as f64 - target).abs();
                let db = (b.1 as f64 / rows as f64 - target).abs();
                da.total_cmp(&db)
            })
            .map(|(x, _)| x.to_string())
    };
    let mut filters = Vec::new();
    for target in [0.005, 0.05, 0.3] {
        if let Some(value) = closest(&values, target, &[])
            && !filters.iter().any(|f: &String| f.ends_with(&format!("={value}")))
        {
            filters.push(format!("{filter_key}={value}"));
        }
    }
    let mut project: Vec<String> = by_count
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| *k != filter_key)
        .take(2)
        .map(str::to_string)
        .collect();
    for target in [0.2, 0.05, 0.003] {
        let skip: Vec<&str> = project.iter().map(String::as_str).chain([filter_key]).collect();
        if let Some(key) = closest(&by_count, target, &skip) {
            project.push(key);
        }
    }
    (filters.join(";"), project)
}

fn strings(a: &VarBinViewArray) -> Vec<Option<String>> {
    let mut ctx = common::SESSION.create_execution_ctx();
    let a = a.clone().into_array();
    (0..a.len())
        .map(|i| {
            let s = a.execute_scalar(i, &mut ctx).unwrap();
            s.as_utf8().value().map(|v| v.to_string())
        })
        .collect()
}

fn main() {
    let data = LazyLock::force(&common::DATA);
    let mut ctx = common::SESSION.create_execution_ctx();
    let (auto_filters, auto_project) = pick_queries(&data.arrow);
    let project: Vec<String> = std::env::var("PROJECT")
        .map(|p| p.split(',').map(str::to_string).collect())
        .unwrap_or(auto_project);
    let project: Vec<&str> = project.iter().map(String::as_str).collect();
    let filters = std::env::var("FILTERS").unwrap_or(auto_filters);
    println!("project {project:?}\n");
    let shredded_bytes = |s: &vortex_shredded_map::ShreddedMapArray| s.clone().into_array().nbytes();
    let (parquet_bytes, parquet_read_ms) = parquet(&data.arrow);
    // BtrBlocks' compact preset adds zstd and pco, the fair match for Parquet zstd(3).
    let map_compact = common::compress_with(&data.map, true);
    let shredded_compact = common::compress_shredded_with(&data.shredded, true);
    println!(
        "SIZE rows={} parquet={} arrow={} arrow_alloc={} map={} map_btr={} map_compact={} shredded={} shredded_btr={} shredded_compact={} parquet_read_ms={:.1}\n",
        data.arrow.len(),
        parquet_bytes,
        arrow_data_bytes(&data.arrow.to_data()),
        data.arrow.get_array_memory_size(),
        data.map.nbytes(),
        data.map_compressed.nbytes(),
        map_compact.nbytes(),
        shredded_bytes(&data.shredded),
        shredded_bytes(&data.shredded_compressed),
        shredded_bytes(&shredded_compact),
        parquet_read_ms,
    );
    println!(
        "{:<28} {:>8} | {:>9} {:>9} {:>9} {:>9} {:>12}  (ms, best of 5)",
        "filter", "rows", "arrow", "map", "map_btr", "shredded", "shredded_btr"
    );
    for filter in filters.split(';') {
        let (key, value) = filter.split_once('=').unwrap();
        let (arrow_ms, expected) = best(|| arrow_query(&data.arrow, key, value, &project));
        let expected: Vec<Vec<Option<String>>> = expected
            .iter()
            .map(|a| (0..a.len()).map(|i| a.is_valid(i).then(|| a.value(i).to_string())).collect())
            .collect();
        let check = |name: &str, got: Vec<VarBinViewArray>| {
            for (i, a) in got.iter().enumerate() {
                assert_eq!(strings(a), expected[i], "{name} {filter} {}", project[i]);
            }
        };
        let (map_ms, got) = best(|| query::map(&data.map, key, value, &project, &mut ctx).unwrap());
        check("map", got);
        let (map_btr_ms, got) =
            best(|| query::map(&data.map_compressed, key, value, &project, &mut ctx).unwrap());
        check("map_btr", got);
        let (shredded_ms, got) =
            best(|| query::shredded(&data.shredded, key, value, &project, &mut ctx).unwrap());
        check("shredded", got);
        let (shredded_btr_ms, got) = best(|| {
            query::shredded(&data.shredded_compressed, key, value, &project, &mut ctx).unwrap()
        });
        check("shredded_btr", got);
        let (map_compact_ms, got) =
            best(|| query::map(&map_compact, key, value, &project, &mut ctx).unwrap());
        check("map_compact", got);
        let (shredded_compact_ms, got) =
            best(|| query::shredded(&shredded_compact, key, value, &project, &mut ctx).unwrap());
        check("shredded_compact", got);
        println!(
            "{filter:<28} {:>8} | {arrow_ms:>9.2} {map_ms:>9.2} {map_btr_ms:>9.2} {shredded_ms:>9.2} {shredded_btr_ms:>12.2}",
            expected[0].len()
        );
        println!(
            "QUERY filter={filter} rows={} arrow={arrow_ms:.3} map={map_ms:.3} map_btr={map_btr_ms:.3} shredded={shredded_ms:.3} shredded_btr={shredded_btr_ms:.3} map_compact={map_compact_ms:.3} shredded_compact={shredded_compact_ms:.3}",
            expected[0].len()
        );
    }
}
