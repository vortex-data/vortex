// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sweeps the shredding thresholds over a dataset and prints size and speed per setting.
//!
//! ```text
//! LO2_PATH=... LO2_MODE=series cargo run --release -p vortex-shredded-map --example shred_sweep
//! ```

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::sync::LazyLock;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_shredded_map::ShredOptions;
use vortex_shredded_map::ShreddedMapArraySlotsExt;
use vortex_shredded_map::ops;
use vortex_shredded_map::shred;

#[path = "../benches/common/mod.rs"]
mod common;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn median(mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[2]
}

fn main() {
    let data = LazyLock::force(&common::DATA);
    let mut ctx = common::SESSION.create_execution_ctx();
    let key = std::env::var("SWEEP_KEY").unwrap_or_else(|_| "image".into());
    let freqs: Vec<f64> = std::env::var("SWEEP_FREQ")
        .unwrap_or_else(|_| "0.001,0.01,0.05,0.2".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let sparse: Vec<f64> = std::env::var("SWEEP_SPARSE")
        .unwrap_or_else(|_| "0,0.2,0.5,0.8,1.01".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    println!(
        "rows {}  map+btr {:.3} MiB  label key {key:?}",
        data.rows,
        data.map_compressed.nbytes() as f64 / 1048576.0
    );
    println!(
        "{:>9} {:>7} {:>5} {:>6} {:>10} {:>10} {:>9} {:>10} {:>9}",
        "min_freq", "sparse<", "cols", "sparse", "btr MiB", "compact", "shred ms", "label ms", "to_map ms"
    );
    for &min_frequency in &freqs {
        for &sparse_below in &sparse {
            let options = ShredOptions {
                min_frequency,
                max_columns: 1024,
                sparse_below,
                ..ShredOptions::default()
            };
            let t = Instant::now();
            let s = shred(&data.map, &options, &mut ctx).unwrap();
            let shred_ms = t.elapsed().as_secs_f64() * 1e3;
            let n_sparse = ShreddedMapArraySlotsExt::columns(&s)
                .iter()
                .filter(|c| c.is::<vortex_sparse::Sparse>())
                .count();
            let btr = common::compress_shredded_with(&s, false);
            let compact = common::compress_shredded_with(&s, true);
            let label = median(|| {
                let a = ops::shredded::get_label_utf8(&btr, &key, &mut ctx).unwrap();
                drop(a.execute::<RecursiveCanonical>(&mut ctx).unwrap());
            });
            let to_map = median(|| {
                let m = ops::shredded::to_map(&btr, &mut ctx).unwrap().into_array();
                drop(m.execute::<RecursiveCanonical>(&mut ctx).unwrap());
            });
            println!(
                "{:>9} {:>7} {:>5} {:>6} {:>10.3} {:>10.3} {:>9.0} {:>10.2} {:>9.1}",
                min_frequency,
                sparse_below,
                s.data().columns().len(),
                n_sparse,
                btr.nbytes() as f64 / 1048576.0,
                compact.nbytes() as f64 / 1048576.0,
                shred_ms,
                label,
                to_map
            );
        }
    }
}
