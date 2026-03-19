// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use divan::Bencher;
use rand::distr::Distribution;
use rand::distr::StandardUniform;
use vortex_array::Canonical;
use vortex_array::DynArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::dict_test::gen_dict_primitive_chunks;
use vortex_array::builders::builder_with_capacity;
use vortex_array::dtype::NativePType;
use vortex_array::session::ArraySession;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const BENCH_ARGS: &[(usize, usize, usize)] = &[
    (1000, 10, 10),
    (1000, 100, 10),
    (1000, 1000, 10),
    (1000, 10, 100),
    (1000, 100, 100),
    (1000, 1000, 100),
];

#[divan::bench(types = [u32, u64, f32, f64], args = BENCH_ARGS)]
fn chunked_dict_primitive_canonical_into<T: NativePType>(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) where
    StandardUniform: Distribution<T>,
{
    let chunk = gen_dict_primitive_chunks::<T, u16>(len, unique_values, chunk_count);

    bencher.with_inputs(|| &chunk).bench_refs(|chunk| {
        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk
            .append_to_builder(builder.as_mut(), &mut SESSION.create_execution_ctx())
            .vortex_expect("append failed");
        builder.finish()
    })
}

#[divan::bench(types = [u32, u64, f32, f64], args = BENCH_ARGS)]
fn chunked_dict_primitive_into_canonical<T: NativePType>(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) where
    StandardUniform: Distribution<T>,
{
    let chunk = gen_dict_primitive_chunks::<T, u16>(len, unique_values, chunk_count);

    bencher
        .with_inputs(|| chunk.clone())
        .bench_values(|chunk| chunk.execute::<Canonical>(&mut SESSION.create_execution_ctx()))
}
