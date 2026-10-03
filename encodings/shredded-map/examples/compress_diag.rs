// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prints the encoding tree and timing BtrBlocks produces for the label map representations.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::sync::LazyLock;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;

#[path = "../benches/common/mod.rs"]
mod common;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn print_tree(name: &str, array: &ArrayRef, depth: usize) {
    let max_depth: usize = std::env::var("DEPTH").ok().and_then(|d| d.parse().ok()).unwrap_or(4);
    if depth > max_depth || array.nbytes() < 4096 {
        return;
    }
    println!(
        "{:indent$}{name}: {} len={} {:.3} MiB",
        "",
        array.encoding_id(),
        array.len(),
        array.nbytes() as f64 / 1048576.0,
        indent = depth * 2
    );
    for (child_name, child) in array.named_children() {
        print_tree(&child_name, &child, depth + 1);
    }
}

fn show(name: &str, array: &ArrayRef, compact: bool) {
    let start = Instant::now();
    let compressed = common::compress_with(array, compact);
    let secs = start.elapsed().as_secs_f64();
    println!(
        "=== {name} compact={compact}: {} -> {} bytes in {secs:.3}s",
        array.nbytes(),
        compressed.nbytes()
    );
    if std::env::var("TREE").is_ok() {
        print_tree("root", &compressed, 0);
    }
}

fn main() {
    let data = LazyLock::force(&common::DATA);
    let which = std::env::var("WHICH").unwrap_or_else(|_| "map".into());
    for compact in [false, true] {
        match which.as_str() {
            "map" => show("map", &data.map, compact),
            _ => {
                let start = Instant::now();
                let compressed = common::compress_shredded_with(&data.shredded, compact);
                println!(
                    "=== shredded children compact={compact}: {} bytes in {:.3}s",
                    compressed.nbytes(),
                    start.elapsed().as_secs_f64()
                );
                if std::env::var("TREE").is_ok() {
                    print_tree("root", &compressed.into_array(), 0);
                }
            }
        }
    }
}
