// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
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

/// Builds `num_branches` boolean value arrays plus random `(array_indices, row_indices)` selectors
/// describing a full random-access gather of `ARRAY_SIZE` output rows.
fn inputs(
    num_branches: usize,
    nullable: bool,
) -> (Vec<vortex_array::ArrayRef>, Buffer<u32>, Buffer<u32>) {
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
