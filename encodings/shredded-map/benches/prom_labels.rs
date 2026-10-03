// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Label operations on real Prometheus labels (LO2v2), across representations:
//!
//! - `map`: canonical Vortex `Map<Utf8, Utf8>` (or the label union with `LO2_VALUES=union`)
//!   with one copy of the entries per row
//! - `map_btr`: the same map compressed with BtrBlocks
//! - `shared`: the map with repeated rows sharing their entries through the list-view
//! - `shared_btr`: the shared map compressed with BtrBlocks
//! - `shredded`: [`ShreddedMap`](vortex_shredded_map::ShreddedMap) with canonical children
//! - `shredded_btr`: the shredded map with BtrBlocks-compressed children
//! - `encoded`: `Dict(codes, ShreddedMap)` from `encode`, shredding only distinct label maps
//! - `encoded_btr`: the encoded map with BtrBlocks-compressed children
//! - `keyset`: `KeySetMap`, one copy of each distinct key set, repeated rows share values
//! - `keyset_btr`: the key-set map with BtrBlocks-compressed children
//! - `map_auto`: the map compressed by BtrBlocks with the map schemes registered
//! - `arrow`: Arrow `Map<Utf8, Utf8>`, the usual stringly-typed label column
//!
//! Every Vortex result is executed to [`RecursiveCanonical`] so no lazy or compressed child
//! escapes the timed region.

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_shredded_map::ShreddedMapArray;
use vortex_shredded_map::ShreddedMapArraySlotsExt;
use vortex_shredded_map::ops;

mod common;

use common::DATA;
use common::SESSION;
use common::arrow_ops;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const ALL: &[&str] = &[
    "map",
    "map_btr",
    "shredded",
    "shredded_btr",
    "encoded",
    "encoded_btr",
    "keyset",
    "keyset_btr",
    "map_auto",
    "arrow",
];
const VORTEX: &[&str] = &[
    "map",
    "map_btr",
    "shredded",
    "shredded_btr",
    "encoded",
    "encoded_btr",
    "keyset",
    "keyset_btr",
    "map_auto",
];
const COMMON_KEYS: &[&str] = &["__name__", "instance", "job"];
const RARE_KEYS: &[&str] = &["cpu", "mode"];

fn main() {
    let data = LazyLock::force(&DATA);
    eprintln!(
        "loaded {} rows from {} series in {:.1}s, shredded in {:.2}s into {} columns",
        data.rows,
        data.series,
        data.load_secs,
        data.shred_secs,
        data.shredded.data().columns().len()
    );
    divan::main();
}

fn materialize(array: ArrayRef, ctx: &mut vortex_array::ExecutionCtx) -> RecursiveCanonical {
    array.execute::<RecursiveCanonical>(ctx).unwrap()
}

fn arrow_done<T>(value: T) -> Option<RecursiveCanonical> {
    divan::black_box(value);
    None
}

enum Input {
    Map(&'static ArrayRef),
    Shredded(&'static ShreddedMapArray),
    Encoded(&'static ArrayRef),
    Arrow,
}

fn input(name: &str) -> Input {
    match name {
        "map" => Input::Map(&DATA.map),
        "map_btr" => Input::Map(&DATA.map_compressed),
        "shared" => Input::Map(&DATA.map_shared),
        "shared_btr" => Input::Map(&DATA.map_shared_compressed),
        "shredded" => Input::Shredded(&DATA.shredded),
        "shredded_btr" => Input::Shredded(&DATA.shredded_compressed),
        "encoded" => Input::Encoded(&DATA.encoded),
        "encoded_btr" => Input::Encoded(&DATA.encoded_compressed),
        "keyset" => Input::Encoded(&DATA.keyset),
        "keyset_btr" => Input::Encoded(&DATA.keyset_compressed),
        "map_auto" => Input::Encoded(&DATA.map_auto),
        "arrow" => Input::Arrow,
        _ => unreachable!(),
    }
}

#[divan::bench(
    args = [
        "map_btr",
        "shredded",
        "shredded_btr",
        "encoded",
        "encoded_btr",
        "keyset",
        "keyset_btr",
        "map_auto"
    ],
    sample_count = 10,
    sample_size = 1
)]
fn decompress_to_map(bencher: Bencher, name: &str) {
    bencher.bench(|| {
        let mut ctx = SESSION.create_execution_ctx();
        match input(name) {
            Input::Map(map) => materialize(map.clone(), &mut ctx),
            Input::Shredded(s) => {
                materialize(ops::shredded::to_map(s, &mut ctx).unwrap().into_array(), &mut ctx)
            }
            Input::Encoded(e) => {
                materialize(ops::encoded::to_map(e, &mut ctx).unwrap().into_array(), &mut ctx)
            }
            Input::Arrow => unreachable!(),
        }
    });
}

#[divan::bench(args = ALL, sample_count = 10, sample_size = 1)]
fn label_names(bencher: Bencher, name: &str) {
    bencher.bench(|| {
        let mut ctx = SESSION.create_execution_ctx();
        match input(name) {
            Input::Map(map) => Some(materialize(
                ops::map::label_names(map, &mut ctx).unwrap().into_array(),
                &mut ctx,
            )),
            Input::Shredded(s) => Some(materialize(
                ops::shredded::label_names(s, &mut ctx).unwrap().into_array(),
                &mut ctx,
            )),
            Input::Encoded(e) => Some(materialize(
                ops::encoded::label_names(e, &mut ctx).unwrap().into_array(),
                &mut ctx,
            )),
            Input::Arrow => arrow_done(arrow_ops::label_names(&DATA.arrow)),
        }
    });
}

#[divan::bench(args = ALL, sample_count = 10, sample_size = 1)]
fn distinct_label_names(bencher: Bencher, name: &str) {
    bencher.bench(|| {
        let mut ctx = SESSION.create_execution_ctx();
        match input(name) {
            Input::Map(map) => ops::map::distinct_label_names(map, &mut ctx).unwrap(),
            Input::Shredded(s) => ops::shredded::distinct_label_names(s, &mut ctx).unwrap(),
            Input::Encoded(e) => ops::encoded::distinct_label_names(e, &mut ctx).unwrap(),
            Input::Arrow => arrow_ops::distinct_label_names(&DATA.arrow),
        }
    });
}

/// The label to read, overridable per dataset through an environment variable.
fn label_key(var: &str, default: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| default.to_string())
}

fn get_label(bencher: Bencher, name: &str, key: &str) {
    bencher.bench(|| {
        let mut ctx = SESSION.create_execution_ctx();
        match input(name) {
            Input::Map(map) => Some(materialize(
                ops::map::get_label_utf8(map, key, &mut ctx).unwrap(),
                &mut ctx,
            )),
            Input::Shredded(s) => Some(materialize(
                ops::shredded::get_label_utf8(s, key, &mut ctx).unwrap(),
                &mut ctx,
            )),
            Input::Encoded(e) => Some(materialize(
                ops::encoded::get_label_utf8(e, key, &mut ctx).unwrap(),
                &mut ctx,
            )),
            Input::Arrow => arrow_done(arrow_ops::get_label(&DATA.arrow, key)),
        }
    });
}

/// `job` is on every row and shredded.
#[divan::bench(args = ALL, sample_count = 10, sample_size = 1)]
fn get_label_common(bencher: Bencher, name: &str) {
    get_label(bencher, name, &label_key("LABEL_COMMON", "job"));
}

/// `image` is on the cAdvisor rows (~18%) and shredded.
#[divan::bench(args = ALL, sample_count = 10, sample_size = 1)]
fn get_label_medium(bencher: Bencher, name: &str) {
    get_label(bencher, name, &label_key("LABEL_MEDIUM", "image"));
}

/// `mode` is on ~4% of rows and stays in the residual.
#[divan::bench(args = ALL, sample_count = 10, sample_size = 1)]
fn get_label_rare(bencher: Bencher, name: &str) {
    get_label(bencher, name, &label_key("LABEL_RARE", "mode"));
}

#[divan::bench(args = VORTEX, sample_count = 10, sample_size = 1)]
fn to_utf8_map(bencher: Bencher, name: &str) {
    bencher.bench(|| {
        let mut ctx = SESSION.create_execution_ctx();
        match input(name) {
            Input::Map(map) => materialize(
                ops::map::to_utf8_map(map, &mut ctx).unwrap().into_array(),
                &mut ctx,
            ),
            Input::Shredded(s) => materialize(
                ops::shredded::to_utf8_map(s, &mut ctx).unwrap().into_array(),
                &mut ctx,
            ),
            Input::Encoded(e) => materialize(
                ops::encoded::to_utf8_map(e, &mut ctx).unwrap().into_array(),
                &mut ctx,
            ),
            Input::Arrow => unreachable!(),
        }
    });
}

fn project(bencher: Bencher, name: &str, keys: &[&str]) {
    bencher.bench(|| {
        let mut ctx = SESSION.create_execution_ctx();
        if name == "encoded_enc" {
            // The projected map stays encoded.
            let projected = ops::encoded::project(&DATA.encoded, keys, &mut ctx).unwrap();
            return Some(divan::black_box(projected));
        }
        if name == "shredded_enc" {
            // The projected map stays shredded: no merge, only the residual is filtered.
            let projected = ops::shredded::project(&DATA.shredded, keys, &mut ctx).unwrap();
            let residual = materialize(projected.residual().clone(), &mut ctx);
            divan::black_box(residual);
            return Some(projected.into_array());
        }
        match input(name) {
            Input::Map(map) => {
                let map = ops::map::project(map, keys, &mut ctx).unwrap().into_array();
                divan::black_box(materialize(map, &mut ctx));
                None
            }
            Input::Shredded(s) => {
                let projected = ops::shredded::project(s, keys, &mut ctx).unwrap();
                let map = ops::shredded::to_map(&projected, &mut ctx).unwrap();
                divan::black_box(materialize(map.into_array(), &mut ctx));
                None
            }
            Input::Encoded(e) => {
                let projected = ops::encoded::project(e, keys, &mut ctx).unwrap();
                let map = ops::encoded::to_map(&projected, &mut ctx).unwrap();
                divan::black_box(materialize(map.into_array(), &mut ctx));
                None
            }
            Input::Arrow => {
                divan::black_box(arrow_ops::project(&DATA.arrow, keys));
                None
            }
        }
    });
}

const PROJECT: &[&str] = &[
    "map",
    "map_btr",
    "shredded",
    "shredded_btr",
    "encoded",
    "encoded_btr",
    "keyset",
    "keyset_btr",
    "map_auto",
    "shredded_enc",
    "arrow",
];

/// Projects `__name__`, `instance` and `job` into a new map.
#[divan::bench(args = PROJECT, sample_count = 10, sample_size = 1)]
fn project_common(bencher: Bencher, name: &str) {
    project(bencher, name, COMMON_KEYS);
}

/// Projects `cpu` and `mode` into a new map.
#[divan::bench(args = PROJECT, sample_count = 10, sample_size = 1)]
fn project_rare(bencher: Bencher, name: &str) {
    project(bencher, name, RARE_KEYS);
}
