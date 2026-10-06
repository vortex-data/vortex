// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Prints the memory layout of the compressed shredded map: every column's encoding tree and
//! bytes, including the sparse chunk offsets, next to the residual and an Arrow map.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::sync::LazyLock;

use arrow_array::Array as _;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_shredded_map::ShreddedMapArrayExt;
use vortex_shredded_map::ShreddedMapArraySlotsExt;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt;

#[path = "../benches/common/mod.rs"]
mod common;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn kib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / 1024.0)
}

fn short(array: &ArrayRef) -> String {
    let id = array.encoding_id();
    let id = id.as_ref();
    id.rsplit('.').next().unwrap_or(id).to_string()
}

/// One line per node: encoding, length and the bytes of its subtree.
fn tree(name: &str, array: &ArrayRef, depth: usize, max_depth: usize) {
    println!(
        "{:indent$}{name}: {} len={} {} KiB",
        "",
        short(array),
        array.len(),
        kib(array.nbytes()),
        indent = depth * 2 + 4
    );
    if depth < max_depth {
        for (child_name, child) in array.named_children() {
            tree(&child_name, &child, depth + 1, max_depth);
        }
    }
}

fn main() {
    let data = LazyLock::force(&common::DATA);
    let s = &data.shredded_compressed;
    let depth: usize = std::env::var("DEPTH")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(2);
    let rows = s.len();
    println!(
        "rows {rows}: shredded+btr {} KiB = residual {} KiB + repeats {} KiB + {} columns {} KiB",
        kib(s.as_ref().nbytes()),
        kib(s.residual().nbytes()),
        kib(s.repeats().map_or(0, |r| r.nbytes())),
        s.column_arrays().len(),
        kib(s.column_arrays().iter().map(|c| c.nbytes()).sum()),
    );
    println!(
        "map+btr {} KiB, arrow MapArray {} KiB\n",
        kib(data.map_compressed.nbytes()),
        kib(data.arrow.get_array_memory_size() as u64)
    );

    let (mut offsets_raw, mut offsets_compressed, mut sparse) = (0u64, 0u64, 0usize);
    println!(
        "{:<44} {:>7} {:>10} {:>11}  layout",
        "column", "present", "KiB", "chunk offs B"
    );
    for (column, array) in s.data().columns().iter().zip(s.column_arrays().iter()) {
        let (present, offsets) = match array.as_opt::<Sparse>() {
            Some(sp) => {
                let patches = sp.patches();
                let offsets = patches.chunk_offsets().as_ref().map(|o| {
                    offsets_raw += o.len() as u64 * 8;
                    offsets_compressed += o.nbytes();
                    sparse += 1;
                    format!("{} ({}x8 raw)", o.nbytes(), o.len())
                });
                (
                    patches.num_patches() as f64 / rows as f64,
                    offsets.unwrap_or_default(),
                )
            }
            None => (1.0, String::new()),
        };
        println!(
            "{:<44} {:>6.1}% {:>10} {:>11}",
            column.key.chars().take(44).collect::<String>(),
            present * 100.0,
            kib(array.nbytes()),
            offsets
        );
        if depth > 0 {
            for (child_name, child) in array.named_children() {
                tree(&child_name, &child, 0, depth - 1);
            }
        }
    }
    println!(
        "\nchunk offsets: {sparse} sparse columns, {} B compressed ({} KiB if stored raw as u64)",
        offsets_compressed,
        kib(offsets_raw)
    );
    println!("\nresidual:");
    tree("residual", s.residual(), 0, depth + 2);
}
