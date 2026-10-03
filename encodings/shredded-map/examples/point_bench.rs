// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Average time to read one label of one random row, per representation.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::sync::LazyLock;
use std::time::Instant;

use arrow_array::Array as _;
use arrow_array::StringArray;
use mimalloc::MiMalloc;
use vortex_array::VortexSessionExecute;
use vortex_shredded_map::ShreddedMapArraySlotsExt;
use vortex_shredded_map::point;

#[path = "../benches/common/mod.rs"]
mod common;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn per_lookup(rows: &[usize], mut f: impl FnMut(usize)) -> f64 {
    let t = Instant::now();
    for &row in rows {
        f(row);
    }
    t.elapsed().as_secs_f64() * 1e6 / rows.len() as f64
}

fn main() {
    let data = LazyLock::force(&common::DATA);
    let key = std::env::var("KEY").unwrap();
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let rows: Vec<usize> = (0..2000)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % data.rows as u64) as usize
        })
        .collect();

    let arrow = &data.arrow;
    let keys = arrow.keys().as_any().downcast_ref::<StringArray>().unwrap();
    let values = arrow.values().as_any().downcast_ref::<StringArray>().unwrap();
    let offsets = arrow.value_offsets();
    let arrow_lookup = |row: usize| {
        (offsets[row] as usize..offsets[row + 1] as usize)
            .find(|&j| keys.value(j) == key)
            .and_then(|j| values.is_valid(j).then(|| values.value(j).to_string()))
    };
    let arrow_us = per_lookup(&rows, |row| {
        std::hint::black_box(arrow_lookup(row));
    });
    let mut ctx = common::SESSION.create_execution_ctx();
    let mut once = |a: &vortex_array::ArrayRef| {
        per_lookup(&rows, |row| {
            std::hint::black_box(point::map_label_at(a, &key, row, &mut ctx).unwrap());
        })
    };
    let map_btr_once = once(&data.map_compressed);
    let mut ctx = common::SESSION.create_execution_ctx();
    let shredded_btr_once = per_lookup(&rows, |row| {
        std::hint::black_box(point::label_at(&data.shredded_compressed, &key, row, &mut ctx).unwrap());
    });

    // Vortex's own retained probe over the whole column (or the residual map).
    let s = &data.shredded_compressed;
    let target = match s.data().columns().iter().position(|c| *c.key == *key) {
        Some(k) => s.columns()[k].clone(),
        None => s.residual().clone(),
    };
    let mut stock = target.repeated_probe();
    let stock_us = per_lookup(&rows, |row| {
        std::hint::black_box(stock.execute_scalar(row, &mut ctx).unwrap());
    });

    // Layered probes: first lookup (builds the probe), then steady state over the same rows.
    let probe_run = |mut lookup: Box<dyn FnMut(usize) -> Option<String> + '_>| {
        let t = Instant::now();
        lookup(rows[0]);
        let first_us = t.elapsed().as_secs_f64() * 1e6;
        let cold_us = per_lookup(&rows, |row| drop(std::hint::black_box(lookup(row))));
        let warm_us = per_lookup(&rows, |row| drop(std::hint::black_box(lookup(row))));
        for &row in &rows {
            assert_eq!(lookup(row), arrow_lookup(row), "row {row}");
        }
        (first_us, cold_us, warm_us)
    };
    let mut sctx = common::SESSION.create_execution_ctx();
    let mut sprobe = point::ShreddedProbe::new(&data.shredded_compressed);
    let sp = probe_run(Box::new(|row| sprobe.get_label(&key, row, &mut sctx).unwrap()));
    let mut mctx = common::SESSION.create_execution_ctx();
    let mut mprobe = point::MapProbe::new(&data.map_compressed, &mut mctx).unwrap();
    let mp = probe_run(Box::new(|row| mprobe.get_label(&key, row, &mut mctx).unwrap()));
    println!(
        "{key}: arrow {arrow_us:.2} µs | one-off: map_btr {map_btr_once:.1} µs, shredded_btr {shredded_btr_once:.2} µs | stock repeated_probe {stock_us:.2} µs"
    );
    println!(
        "  ShreddedProbe(btr): first {:.0} µs, 2000 random {:.2} µs/lookup, again {:.2} µs/lookup, holds {:.1} KiB",
        sp.0, sp.1, sp.2, sprobe.nbytes() as f64 / 1024.0
    );
    println!(
        "  MapProbe(map_btr):  first {:.0} µs, 2000 random {:.2} µs/lookup, again {:.2} µs/lookup, holds {:.1} KiB",
        mp.0, mp.1, mp.2, mprobe.nbytes() as f64 / 1024.0
    );
}
