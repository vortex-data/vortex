// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

#[divan::bench]
fn scalar_subtract(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0i64, 100_000_000).unwrap();
    let data1 = (0..100_000)
        .map(|_| rng.sample(range))
        .collect::<Buffer<i64>>()
        .into_array();

    let data2 = (0..100_000)
        .map(|_| rng.sample(range))
        .collect::<Buffer<i64>>()
        .into_array();

    let to_subtract = -1i64;

    let chunked = ChunkedArray::from_iter([data1, data2]).into_array();

    bencher
        .with_inputs(|| (&chunked, SESSION.create_execution_ctx()))
        .bench_refs(|(chunked, ctx)| {
            chunked
                .clone()
                .binary(
                    ConstantArray::new(
                        vortex_array::scalar::Scalar::from(to_subtract),
                        chunked.len(),
                    )
                    .into_array(),
                    Operator::Sub,
                )
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}
