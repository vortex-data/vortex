// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Loads the LO2v2 Prometheus label dataset and builds every representation the benchmarks use.
//!
//! Set `LO2_PATH` to the series JSONL file and `LO2_ROWS` to the number of sample rows to load.
//! Each row is one sample and carries its series' label map, like a long-format metrics table.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::io::BufRead;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Instant;

use arrow_array::Array as _;
use arrow_array::BooleanArray;
use arrow_array::ListArray;
use arrow_array::MapArray as ArrowMapArray;
use arrow_array::StringArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::builder::MapBuilder;
use arrow_array::builder::StringBuilder;
use arrow_buffer::OffsetBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::Nullability;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_session::VortexSession;
use vortex_shredded_map::ShredOptions;
use vortex_shredded_map::ShreddedMap;
use vortex_shredded_map::ShreddedMapArray;
use vortex_shredded_map::ShreddedMapArraySlotsExt;
use vortex_shredded_map::labels::LabelMapBuilder;
use vortex_shredded_map::labels::LabelValue;
use vortex_utils::aliases::hash_set::HashSet;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_shredded_map::initialize(&session);
    session
});

pub struct Data {
    pub rows: usize,
    pub series: usize,
    pub load_secs: f64,
    /// Canonical `Map<Utf8, Union<str, int, float, bool>>`.
    pub map: ArrayRef,
    /// The canonical map compressed with BtrBlocks.
    pub map_compressed: ArrayRef,
    pub shredded: ShreddedMapArray,
    /// The shredded map with each child compressed with BtrBlocks.
    pub shredded_compressed: ShreddedMapArray,
    /// Arrow `Map<Utf8, Utf8>`, the usual stringly-typed label column.
    pub arrow: ArrowMapArray,
    pub shred_secs: f64,
}

pub static DATA: LazyLock<Data> = LazyLock::new(load);

type Labels = Vec<(String, String)>;

fn read_series(path: &str) -> Vec<(Arc<Labels>, usize)> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    std::io::BufReader::with_capacity(1 << 20, file)
        .lines()
        .map(|line| {
            let line = line.unwrap();
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            let labels: BTreeMap<String, String> = value["labels"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect();
            let samples = value["ts"].as_array().unwrap().len();
            (Arc::new(labels.into_iter().collect()), samples)
        })
        .collect()
}

/// Picks every `stride`-th series so the sampled rows span all metrics, then truncates.
fn sample_rows(series: &[(Arc<Labels>, usize)], target: usize) -> (Vec<Arc<Labels>>, usize) {
    let total: usize = series.iter().map(|s| s.1).sum();
    let stride = total.div_ceil(target.max(1)).max(1);
    let mut rows = Vec::with_capacity(target);
    let mut used = 0;
    for (labels, samples) in series.iter().step_by(stride) {
        if rows.len() >= target {
            break;
        }
        used += 1;
        let take = (*samples).min(target - rows.len());
        rows.extend(std::iter::repeat_n(labels.clone(), take));
    }
    (rows, used)
}

fn build_arrow(rows: &[Arc<Labels>]) -> ArrowMapArray {
    let mut builder = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    for row in rows {
        for (k, v) in row.iter() {
            builder.keys().append_value(k);
            builder.values().append_value(v);
        }
        builder.append(true).unwrap();
    }
    builder.finish()
}

pub fn compress(array: &ArrayRef) -> ArrayRef {
    compress_with(array, false)
}

/// Compresses with BtrBlocks, or with its compact preset (adds Zstd and Pco) when `compact`.
pub fn compress_with(array: &ArrayRef, compact: bool) -> ArrayRef {
    let mut builder = BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted();
    if compact {
        builder = builder.with_compact();
    }
    let compressor = builder.build();
    compressor
        .compress(array, &mut SESSION.create_execution_ctx())
        .unwrap()
}

pub fn compress_shredded(shredded: &ShreddedMapArray) -> ShreddedMapArray {
    compress_shredded_with(shredded, false)
}

pub fn compress_shredded_with(shredded: &ShreddedMapArray, compact: bool) -> ShreddedMapArray {
    let residual = compress_with(shredded.residual(), compact);
    let columns = shredded
        .columns()
        .iter()
        .map(|c| compress_with(c, compact))
        .collect();
    ShreddedMap::try_new(residual, shredded.data().columns().to_vec(), columns).unwrap()
}

fn load() -> Data {
    let path = std::env::var("LO2_PATH").unwrap_or_else(|_| "/home/user/data/lo2_series.jsonl".into());
    let target: usize = std::env::var("LO2_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let start = Instant::now();
    let all_series = read_series(&path);
    // `LO2_MODE=series` keeps one row per series instead of one row per sample.
    let (rows, series) = if std::env::var("LO2_MODE").as_deref() == Ok("series") {
        let rows: Vec<Arc<Labels>> = all_series.iter().map(|s| s.0.clone()).take(target).collect();
        let n = rows.len();
        (rows, n)
    } else {
        sample_rows(&all_series, target)
    };
    drop(all_series);

    let mut builder = LabelMapBuilder::new(Nullability::NonNullable, Nullability::NonNullable);
    for row in &rows {
        let typed: Vec<(&str, LabelValue)> = row
            .iter()
            .map(|(k, v)| (k.as_str(), LabelValue::infer(v)))
            .collect();
        builder.push_row(typed.iter().map(|(k, v)| (*k, Some(v))));
    }
    let map = builder.finish().unwrap().into_array();
    let arrow = build_arrow(&rows);
    let load_secs = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let shredded = vortex_shredded_map::shred(
        &map,
        &ShredOptions::default(),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();
    let shred_secs = start.elapsed().as_secs_f64();

    let map_compressed = compress(&map);
    let shredded_compressed = compress_shredded(&shredded);
    Data {
        rows: rows.len(),
        series,
        load_secs,
        map,
        map_compressed,
        shredded,
        shredded_compressed,
        arrow,
        shred_secs,
    }
}

/// Arrow baseline implementations of the label operations.
pub mod arrow_ops {
    use super::*;

    fn keys(map: &ArrowMapArray) -> &StringArray {
        map.keys().as_any().downcast_ref::<StringArray>().unwrap()
    }

    fn values(map: &ArrowMapArray) -> &StringArray {
        map.values().as_any().downcast_ref::<StringArray>().unwrap()
    }

    pub fn label_names(map: &ArrowMapArray) -> ListArray {
        let field = Arc::new(Field::new("item", DataType::Utf8, false));
        ListArray::new(
            field,
            map.offsets().clone(),
            map.keys().clone(),
            map.nulls().cloned(),
        )
    }

    pub fn distinct_label_names(map: &ArrowMapArray) -> Vec<String> {
        let keys = keys(map);
        let offsets = map.value_offsets();
        let mut seen: HashSet<&str> = HashSet::new();
        for row in 0..map.len() {
            if map.is_valid(row) {
                for j in offsets[row] as usize..offsets[row + 1] as usize {
                    seen.insert(keys.value(j));
                }
            }
        }
        let mut names: Vec<String> = seen.into_iter().map(str::to_string).collect();
        names.sort();
        names
    }

    pub fn get_label(map: &ArrowMapArray, key: &str) -> StringArray {
        let keys = keys(map);
        let values = values(map);
        let offsets = map.value_offsets();
        (0..map.len())
            .map(|row| {
                if !map.is_valid(row) {
                    return None;
                }
                (offsets[row] as usize..offsets[row + 1] as usize)
                    .find(|&j| keys.value(j) == key)
                    .map(|j| values.value(j))
            })
            .collect()
    }

    pub fn project(map: &ArrowMapArray, wanted: &[&str]) -> ArrowMapArray {
        let keys = keys(map);
        let wanted: HashSet<&str> = wanted.iter().copied().collect();
        let offsets = map.value_offsets();
        let mut keep = Vec::with_capacity(keys.len());
        let mut new_offsets = Vec::with_capacity(map.len() + 1);
        new_offsets.push(0i32);
        let mut kept = 0i32;
        for row in 0..map.len() {
            for j in offsets[row] as usize..offsets[row + 1] as usize {
                let k = wanted.contains(keys.value(j));
                keep.push(k);
                kept += i32::from(k);
            }
            new_offsets.push(kept);
        }
        let entries: &ArrowStructArray = map.entries();
        let filtered = arrow_select::filter::filter(entries, &BooleanArray::from(keep)).unwrap();
        let filtered = filtered
            .as_any()
            .downcast_ref::<ArrowStructArray>()
            .unwrap()
            .clone();
        let DataType::Map(field, sorted) = map.data_type() else {
            unreachable!()
        };
        ArrowMapArray::try_new(
            field.clone(),
            OffsetBuffer::new(new_offsets.into()),
            filtered,
            map.nulls().cloned(),
            *sorted,
        )
        .unwrap()
    }
}
