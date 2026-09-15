// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-element reads from a PCO array through `execute_scalar`.
//!
//! PCO stores values in chunks of pages, and a scalar read decodes the page holding the value.
//! Two access patterns bracket the cost: one read per chunk, so every read lands on a different
//! chunk and page, and many reads inside one chunk, where reads share pages.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use rand::seq::SliceRandom;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_pco::Pco;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

/// Values per PCO chunk; `Pco::from_primitive` splits input at this granularity.
const VALUES_PER_CHUNK: usize = pco::DEFAULT_MAX_PAGE_N;
const VALUES_PER_PAGE: usize = 1024;
const CHUNKS: usize = 16;
// Sized to keep the CodSpeed simulation under a millisecond per benchmark.
const READS_IN_CHUNK: usize = 32;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

fn pco_array() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let len = u32::try_from(CHUNKS * VALUES_PER_CHUNK).unwrap();
    let values = PrimitiveArray::from_iter((0..len).map(|i| i / 7));
    Pco::from_primitive(values.as_view(), 3, VALUES_PER_PAGE, &mut ctx)
        .unwrap()
        .into_array()
}

/// One index inside every chunk, visited in a shuffled order.
fn one_index_per_chunk() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut indices: Vec<usize> = (0..CHUNKS)
        .map(|chunk| chunk * VALUES_PER_CHUNK + rng.random_range(0..VALUES_PER_CHUNK))
        .collect();
    indices.shuffle(&mut rng);
    indices
}

/// Many indices inside one chunk in the middle of the array.
fn indices_in_one_chunk() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(1);
    let chunk = CHUNKS / 2;
    (0..READS_IN_CHUNK)
        .map(|_| chunk * VALUES_PER_CHUNK + rng.random_range(0..VALUES_PER_CHUNK))
        .collect()
}

fn read_all(bencher: Bencher, array: ArrayRef, indices: Vec<usize>) {
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            for &index in &indices {
                divan::black_box(array.execute_scalar(divan::black_box(index), ctx).unwrap());
            }
        });
}

#[divan::bench]
fn scalar_at_one_per_chunk(bencher: Bencher) {
    read_all(bencher, pco_array(), one_index_per_chunk());
}

#[divan::bench]
fn scalar_at_within_one_chunk(bencher: Bencher) {
    read_all(bencher, pco_array(), indices_in_one_chunk());
}
