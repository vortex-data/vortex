// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::builders::builder_with_capacity;
use vortex_array::dtype::NativePType;
use vortex_array::session::ArraySession;
use vortex_error::VortexExpect;
use vortex_fsst::test_utils::gen_dict_fsst_test_data;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

const BENCH_ARGS: &[(usize, usize, usize)] = &[
    (1000, 10, 10),
    (1000, 100, 10),
    (1000, 1000, 10),
    (1000, 10, 100),
    (1000, 100, 100),
    (1000, 1000, 100),
];

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

fn make_dict_fsst_chunks<T: NativePType>(
    len: usize,
    unique_values: usize,
    chunk_count: usize,
) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    (0..chunk_count)
        .map(|_| gen_dict_fsst_test_data::<T>(len, unique_values, 20, 30, &mut ctx).into_array())
        .collect::<ChunkedArray>()
        .into_array()
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_dict_fsst_canonical_into(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) {
    let chunk = make_dict_fsst_chunks::<u16>(len, unique_values, chunk_count);

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| {
            let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
            chunk
                .append_to_builder(builder.as_mut(), ctx)
                .vortex_expect("append failed");
            builder.finish()
        })
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_dict_fsst_into_canonical(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) {
    let chunk = make_dict_fsst_chunks::<u16>(len, unique_values, chunk_count);

    bencher
        .with_inputs(|| (&chunk, SESSION.create_execution_ctx()))
        .bench_refs(|(chunk, ctx)| (**chunk).clone().execute::<Canonical>(ctx))
}
