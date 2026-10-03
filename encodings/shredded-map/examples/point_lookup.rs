// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Times a single-row label lookup through each layer of a compressed shredded column.

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_shredded_map::ShreddedMapArraySlotsExt;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt;

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

fn tree(a: &ArrayRef, depth: usize) {
    println!("{:indent$}{} len={}", "", a.encoding_id(), a.len(), indent = depth * 2);
    for c in a.children() {
        tree(&c, depth + 1);
    }
}

fn main() {
    let data = LazyLock::force(&common::DATA);
    let mut ctx = common::SESSION.create_execution_ctx();
    let key = std::env::var("KEY").unwrap();
    let s = &data.shredded_compressed;
    let column = match s.data().columns().iter().position(|c| &*c.key == key) {
        Some(k) => s.columns()[k].clone(),
        None => s.residual().clone(),
    };
    tree(&column, 0);
    let row = column.len() / 2;
    println!("column scalar_at: {:.4} ms", best(|| { column.execute_scalar(row, &mut ctx).unwrap(); }));
    if let Some(sparse) = column.as_opt::<Sparse>() {
        let patches = sparse.patches();
        let indices = patches.indices().clone();
        let values = patches.values().clone();
        println!("  patch indices scalar_at: {:.4} ms", best(|| drop(indices.execute_scalar(indices.len() / 2, &mut ctx).unwrap())));
        println!("  patch search_index:      {:.4} ms", best(|| { std::hint::black_box(patches.search_index(row).unwrap()); }));
        println!("  patch values scalar_at:  {:.4} ms", best(|| drop(values.execute_scalar(values.len() / 2, &mut ctx).unwrap())));
        if let Some(dict) = values.as_opt::<Dict>() {
            let codes = dict.codes().clone();
            let dvalues = dict.values().clone();
            println!("    dict codes scalar_at:  {:.4} ms", best(|| drop(codes.execute_scalar(codes.len() / 2, &mut ctx).unwrap())));
            println!("    dict values scalar_at: {:.4} ms", best(|| drop(dvalues.execute_scalar(dvalues.len() / 2, &mut ctx).unwrap())));
            println!("    dict values compressed {} B", dvalues.nbytes());
            let mut block = None;
            println!("    dict values block decode: {:.4} ms", best(|| block = Some(dvalues.slice(0..1024).unwrap().execute::<vortex_array::Canonical>(&mut ctx).unwrap().into_array())));
            let block = block.unwrap();
            println!("    block nbytes {} buffers {:?}", block.nbytes(), block.buffers().iter().map(|b| b.len()).collect::<Vec<_>>());
            let mut whole = None;
            println!("    dict values whole decode: {:.4} ms", best(|| whole = Some(dvalues.clone().execute::<vortex_array::Canonical>(&mut ctx).unwrap().into_array())));
            println!("    whole nbytes {}", whole.unwrap().nbytes());
        }
    }
}
