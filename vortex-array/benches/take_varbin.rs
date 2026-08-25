// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const ARRAY_SIZE: usize = 20_000;
// Sized to keep CodSpeed simulation under 1ms.
const TAKE_SIZE: usize = 3_000;

#[divan::bench]
fn take_varbin(bencher: Bencher) {
    let array = VarBinArray::from_iter(
        (0..ARRAY_SIZE).map(|i| Some(format!("row-{i:0>40}"))),
        DType::Utf8(Nullability::NonNullable),
    )
    .into_array();

    let mut rng = StdRng::seed_from_u64(0);
    let indices: Buffer<u64> = (0..TAKE_SIZE)
        .map(|_| rng.random_range(0..ARRAY_SIZE) as u64)
        .collect();
    let indices = indices.into_array();

    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
        });
}
