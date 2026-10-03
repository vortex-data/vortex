// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Times reading one label as a lazy dictionary versus materialized strings, and one row.

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_shredded_map::ops;

#[path = "../benches/common/mod.rs"]
mod common;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn best(mut f: impl FnMut()) -> f64 {
    (0..7)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::MAX, f64::min)
}

fn main() {
    let data = LazyLock::force(&common::DATA);
    let mut ctx = common::SESSION.create_execution_ctx();
    let key = std::env::var("KEY").unwrap();
    let s = &data.shredded_compressed;
    let lazy = best(|| drop(ops::shredded::get_label_utf8(s, &key, &mut ctx).unwrap()));
    let full = best(|| {
        let a = ops::shredded::get_label_utf8(s, &key, &mut ctx).unwrap();
        drop(a.execute::<RecursiveCanonical>(&mut ctx).unwrap());
    });
    let label = ops::shredded::get_label_utf8(s, &key, &mut ctx).unwrap();
    let one = best(|| drop(label.execute_scalar(label.len() / 2, &mut ctx).unwrap()));
    println!(
        "{key}: encoding {}  lazy {lazy:.3} ms  materialized {full:.3} ms  one row of the lazy result {:.4} ms",
        label.encoding_id(),
        one
    );
}
