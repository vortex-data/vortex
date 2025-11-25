// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::builders::builder_with_capacity;
use vortex_array::compute::warm_up_vtables;
use vortex_dtype::NativePType;
use vortex_fsst::test_utils::gen_dict_fsst_test_data;

fn main() {
    warm_up_vtables();
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

fn make_dict_fsst_chunks<T: NativePType>(
    len: usize,
    unique_values: usize,
    chunk_count: usize,
) -> ArrayRef {
    (0..chunk_count)
        .map(|_| gen_dict_fsst_test_data::<T>(len, unique_values, 20, 30).into_array())
        .collect::<ChunkedArray>()
        .into_array()
}

#[divan::bench(args = BENCH_ARGS)]
fn chunked_dict_fsst_canonical_into(
    bencher: Bencher,
    (len, unique_values, chunk_count): (usize, usize, usize),
) {
    let chunk = make_dict_fsst_chunks::<u16>(len, unique_values, chunk_count);

    bencher.with_inputs(|| chunk.clone()).bench_values(|chunk| {
        let mut builder = builder_with_capacity(chunk.dtype(), len * chunk_count);
        chunk.append_to_builder(builder.as_mut());
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
        .with_inputs(|| chunk.clone())
        .bench_values(|chunk| chunk.to_canonical())
}
