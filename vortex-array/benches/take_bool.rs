// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for `take` on bit-packed boolean arrays ([`BoolArray`]).
//!
//! Tests the full `DynArray::take` production path with various array sizes
//! and both uniform random and Zipfian (power-law) index distributions.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use divan::Bencher;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::*;
use rand_distr::Zipf;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_buffer::BitBuffer;

fn main() {
    divan::main();
}

/// Number of boolean values in the source array.
const BOOL_LENS: &[usize] = &[64, 512, 1024, 2048, 8192, 16384];

/// Number of indices to take (output array length).
const NUM_INDICES: &[usize] = &[1024, 10_000, 100_000];

// ===========================================================================
// Helper constructors
// ===========================================================================

fn make_bool_array(len: usize) -> BoolArray {
    let mut rng = StdRng::seed_from_u64(42);
    let buf = BitBuffer::collect_bool(len, |_| rng.random_bool(0.5));
    BoolArray::from(buf)
}

fn make_uniform_indices_array(num_indices: usize, max_val: usize) -> PrimitiveArray {
    let rng = StdRng::seed_from_u64(123);
    let dist = Uniform::new(0u32, max_val as u32).unwrap();
    PrimitiveArray::from_iter(rng.sample_iter(dist).take(num_indices))
}

fn make_zipfian_indices_array(num_indices: usize, max_val: usize) -> PrimitiveArray {
    let rng = StdRng::seed_from_u64(123);
    let zipf = Zipf::new(max_val as f64, 1.0).unwrap();
    PrimitiveArray::from_iter(
        rng.sample_iter(&zipf)
            .take(num_indices)
            .map(|i: f64| (i as u32 - 1).min(max_val as u32 - 1)),
    )
}

// ===========================================================================
// DynArray::take benchmarks (full production path)
// ===========================================================================

#[divan::bench(args = BOOL_LENS, consts = NUM_INDICES)]
fn take_bool_uniform<const N: usize>(bencher: Bencher, bool_len: usize) {
    let bools = make_bool_array(bool_len);
    let indices = make_uniform_indices_array(N, bool_len);

    bencher
        .with_inputs(|| (&bools, indices.clone().into_array()))
        .bench_refs(|(bools, indices)| bools.take(indices.to_array()).unwrap());
}

#[divan::bench(args = BOOL_LENS, consts = NUM_INDICES)]
fn take_bool_zipfian<const N: usize>(bencher: Bencher, bool_len: usize) {
    let bools = make_bool_array(bool_len);
    let indices = make_zipfian_indices_array(N, bool_len);

    bencher
        .with_inputs(|| (&bools, indices.clone().into_array()))
        .bench_refs(|(bools, indices)| bools.take(indices.to_array()).unwrap());
}
