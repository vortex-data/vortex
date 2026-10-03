// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Where the bytes go in the compact row-dictionary + shredded layout: every node holding at
//! least `MIN_PCT` percent of the total, with its encoding and size.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::sync::LazyLock;

use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_shredded_map::ShredOptions;

#[path = "../benches/common/mod.rs"]
mod common;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn tree(name: &str, array: &ArrayRef, depth: usize, total: u64, min_pct: f64) {
    let pct = array.nbytes() as f64 * 100.0 / total as f64;
    if pct < min_pct {
        return;
    }
    let own: u64 = array.buffers().iter().map(|b| b.len() as u64).sum();
    println!(
        "{:indent$}{name}: {} len={} {:.1}% ({} B, own buffers {} B)",
        "",
        array.encoding_id(),
        array.len(),
        pct,
        array.nbytes(),
        own,
        indent = depth * 2
    );
    for (child, c) in array.named_children() {
        tree(&child, &c, depth + 1, total, min_pct);
    }
}

fn main() {
    let data = LazyLock::force(&common::DATA);
    let mut ctx = common::SESSION.create_execution_ctx();
    let min_pct: f64 = std::env::var("MIN_PCT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);
    let opts = ShredOptions {
        max_distinct_rows: 1.0,
        ..ShredOptions::default()
    };
    let encoded = vortex_shredded_map::encode(&data.map, &opts, &mut ctx).unwrap();
    let compressed = common::compress_encoded(&encoded, true);
    let total = compressed.nbytes();
    println!("total {total} B");
    tree("root", &compressed, 0, total, min_pct);
}
