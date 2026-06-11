// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use arrow_array::Array as ArrowArray;
use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::InterleaveArray;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 8_192;
const LARGE_ARRAY_SIZE: usize = 1 << 20;

/// Builds `num_branches` boolean value arrays plus random `(array_indices, row_indices)` selectors
/// describing a full random-access gather of `ARRAY_SIZE` output rows.
fn inputs(num_branches: usize, nullable: bool) -> (Vec<ArrayRef>, Buffer<u32>, Buffer<u32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let bit = Uniform::new(0u8, 2).unwrap();

    let values = (0..num_branches)
        .map(|_| {
            if nullable {
                BoolArray::from_iter(
                    (0..ARRAY_SIZE).map(|_| (rng.sample(bit) == 0).then_some(rng.sample(bit) == 0)),
                )
                .into_array()
            } else {
                BoolArray::from_iter((0..ARRAY_SIZE).map(|_| rng.sample(bit) == 0)).into_array()
            }
        })
        .collect();

    let branch = Uniform::new(0u32, u32::try_from(num_branches).unwrap()).unwrap();
    let row = Uniform::new(0u32, u32::try_from(ARRAY_SIZE).unwrap()).unwrap();
    let array_indices: Buffer<u32> = (0..ARRAY_SIZE).map(|_| rng.sample(branch)).collect();
    let row_indices: Buffer<u32> = (0..ARRAY_SIZE).map(|_| rng.sample(row)).collect();
    (values, array_indices, row_indices)
}

#[divan::bench(args = [2, 4])]
fn interleave_bool(bencher: Bencher, num_branches: usize) {
    let (values, array_indices, row_indices) = inputs(num_branches, false);
    let session = VortexSession::empty();
    bencher
        .with_inputs(|| {
            (
                InterleaveArray::try_new(
                    values.clone(),
                    array_indices.clone().into_array(),
                    row_indices.clone().into_array(),
                )
                .unwrap()
                .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| array.clone().execute::<Canonical>(ctx));
}

/// Builds random `(array_indices, row_indices)` selectors describing a full random-access gather
/// of `len` output rows from `num_branches` branches of `len` rows each.
fn selectors(num_branches: usize, len: usize) -> (Buffer<u32>, Buffer<u32>) {
    let mut rng = StdRng::seed_from_u64(1);
    let branch = Uniform::new(0u32, u32::try_from(num_branches).unwrap()).unwrap();
    let row = Uniform::new(0u32, u32::try_from(len).unwrap()).unwrap();
    let array_indices: Buffer<u32> = (0..len).map(|_| rng.sample(branch)).collect();
    let row_indices: Buffer<u32> = (0..len).map(|_| rng.sample(row)).collect();
    (array_indices, row_indices)
}

fn u32_values(num_branches: usize, len: usize) -> Vec<Buffer<u32>> {
    let mut rng = StdRng::seed_from_u64(2);
    let value = Uniform::new(0u32, u32::MAX).unwrap();
    (0..num_branches)
        .map(|_| (0..len).map(|_| rng.sample(value)).collect())
        .collect()
}

fn f64_values(num_branches: usize, len: usize) -> Vec<Buffer<f64>> {
    let mut rng = StdRng::seed_from_u64(2);
    let value = Uniform::new(0f64, 1f64).unwrap();
    (0..num_branches)
        .map(|_| (0..len).map(|_| rng.sample(value)).collect())
        .collect()
}

/// Benches executing a vortex `InterleaveArray` to canonical; construction is untimed.
fn bench_vortex(bencher: Bencher, values: Vec<ArrayRef>, num_branches: usize, len: usize) {
    let (array_indices, row_indices) = selectors(num_branches, len);
    let session = VortexSession::empty();
    bencher
        .with_inputs(|| {
            (
                InterleaveArray::try_new(
                    values.clone(),
                    array_indices.clone().into_array(),
                    row_indices.clone().into_array(),
                )
                .unwrap()
                .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| array.clone().execute::<Canonical>(ctx));
}

/// Benches `arrow_select::interleave::interleave` over the same gather pattern as [`bench_vortex`]
/// (same selector seed), with arrow's native `&[(usize, usize)]` indices built untimed.
fn bench_arrow(
    bencher: Bencher,
    values: Vec<arrow_array::ArrayRef>,
    num_branches: usize,
    len: usize,
) {
    let (array_indices, row_indices) = selectors(num_branches, len);
    let indices: Vec<(usize, usize)> = array_indices
        .iter()
        .zip(row_indices.iter())
        .map(|(&a, &r)| (a as usize, r as usize))
        .collect();
    let refs: Vec<&dyn ArrowArray> = values.iter().map(|v| v.as_ref()).collect();
    bencher.bench(|| arrow_select::interleave::interleave(&refs, &indices));
}

#[divan::bench(args = [2, 4, 128])]
fn interleave_u32(bencher: Bencher, num_branches: usize) {
    let values = u32_values(num_branches, ARRAY_SIZE)
        .into_iter()
        .map(IntoArray::into_array)
        .collect();
    bench_vortex(bencher, values, num_branches, ARRAY_SIZE);
}

#[divan::bench(args = [2, 4])]
fn interleave_u32_large(bencher: Bencher, num_branches: usize) {
    let values = u32_values(num_branches, LARGE_ARRAY_SIZE)
        .into_iter()
        .map(IntoArray::into_array)
        .collect();
    bench_vortex(bencher, values, num_branches, LARGE_ARRAY_SIZE);
}

#[divan::bench(args = [2, 4])]
fn interleave_f64_large(bencher: Bencher, num_branches: usize) {
    let values = f64_values(num_branches, LARGE_ARRAY_SIZE)
        .into_iter()
        .map(IntoArray::into_array)
        .collect();
    bench_vortex(bencher, values, num_branches, LARGE_ARRAY_SIZE);
}

#[divan::bench(args = [2, 4, 128])]
fn arrow_interleave_u32(bencher: Bencher, num_branches: usize) {
    let values = u32_values(num_branches, ARRAY_SIZE)
        .into_iter()
        .map(|b| {
            std::sync::Arc::new(arrow_array::UInt32Array::new(
                b.into_arrow_scalar_buffer(),
                None,
            )) as arrow_array::ArrayRef
        })
        .collect();
    bench_arrow(bencher, values, num_branches, ARRAY_SIZE);
}

#[divan::bench(args = [2, 4])]
fn arrow_interleave_u32_large(bencher: Bencher, num_branches: usize) {
    let values = u32_values(num_branches, LARGE_ARRAY_SIZE)
        .into_iter()
        .map(|b| {
            std::sync::Arc::new(arrow_array::UInt32Array::new(
                b.into_arrow_scalar_buffer(),
                None,
            )) as arrow_array::ArrayRef
        })
        .collect();
    bench_arrow(bencher, values, num_branches, LARGE_ARRAY_SIZE);
}

#[divan::bench(args = [2, 4])]
fn arrow_interleave_f64_large(bencher: Bencher, num_branches: usize) {
    let values = f64_values(num_branches, LARGE_ARRAY_SIZE)
        .into_iter()
        .map(|b| {
            std::sync::Arc::new(arrow_array::Float64Array::new(
                b.into_arrow_scalar_buffer(),
                None,
            )) as arrow_array::ArrayRef
        })
        .collect();
    bench_arrow(bencher, values, num_branches, LARGE_ARRAY_SIZE);
}

#[divan::bench(args = [2, 4])]
fn interleave_bool_nullable(bencher: Bencher, num_branches: usize) {
    let (values, array_indices, row_indices) = inputs(num_branches, true);
    let session = VortexSession::empty();
    bencher
        .with_inputs(|| {
            (
                InterleaveArray::try_new(
                    values.clone(),
                    array_indices.clone().into_array(),
                    row_indices.clone().into_array(),
                )
                .unwrap()
                .into_array(),
                session.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| array.clone().execute::<Canonical>(ctx));
}
