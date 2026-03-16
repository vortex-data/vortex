// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use rand::Rng;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 65_536;

#[divan::bench]
fn compare_bool(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0u8, 1).unwrap();

    let arr1 = BoolArray::from_iter((0..ARRAY_SIZE).map(|_| rng.sample(range) == 0)).into_array();
    let arr2 = BoolArray::from_iter((0..ARRAY_SIZE).map(|_| rng.sample(range) == 0)).into_array();
    let session = VortexSession::empty();

    bencher
        .with_inputs(|| (&arr1, &arr2, session.create_execution_ctx()))
        .bench_refs(|input| {
            input
                .0
                .to_array()
                .binary(input.1.to_array(), Operator::Gte)
                .unwrap()
                .execute::<Canonical>(&mut input.2)
        });
}

#[divan::bench]
fn compare_int(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i64, 100_000_000).unwrap();

    let arr1 = (0..ARRAY_SIZE)
        .map(|_| rng.sample(range))
        .collect::<Buffer<_>>()
        .into_array();

    let arr2 = (0..ARRAY_SIZE)
        .map(|_| rng.sample(range))
        .collect::<Buffer<_>>()
        .into_array();
    let session = VortexSession::empty();

    bencher
        .with_inputs(|| (&arr1, &arr2, session.create_execution_ctx()))
        .bench_refs(|input| {
            input
                .0
                .to_array()
                .binary(input.1.to_array(), Operator::Gte)
                .unwrap()
                .execute::<Canonical>(&mut input.2)
        });
}
