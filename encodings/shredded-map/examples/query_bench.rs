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
    let project: Vec<String> = std::env::var("PROJECT")
        .unwrap_or_else(|_| {
            "userAgent,sourceIPAddress,errorCode,errorMessage,requestParameters.roleArn,requestParameters.userName".into()
        })
        .split(',')
        .map(str::to_string)
        .collect();
    let project: Vec<&str> = project.iter().map(String::as_str).collect();
    let filters = std::env::var("FILTERS").unwrap_or_else(|_| {
        "eventName=GetRestApis;eventName=AssumeRole;eventName=RunInstances;errorCode=AccessDenied".into()
    });
    println!("project {project:?}\n");
    let shredded_bytes = |s: &vortex_shredded_map::ShreddedMapArray| s.clone().into_array().nbytes();
    println!(
        "bytes: arrow {} (buffers; {} allocated) | map {} | map_btr {} | shredded {} | shredded_btr {}\n",
        arrow_data_bytes(&data.arrow.to_data()),
        data.arrow.get_array_memory_size(),
        data.map.nbytes(),
        data.map_compressed.nbytes(),
        shredded_bytes(&data.shredded),
        shredded_bytes(&data.shredded_compressed),
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
        println!(
            "{filter:<28} {:>8} | {arrow_ms:>9.2} {map_ms:>9.2} {map_btr_ms:>9.2} {shredded_ms:>9.2} {shredded_btr_ms:>12.2}",
            expected[0].len()
        );
    }
}
