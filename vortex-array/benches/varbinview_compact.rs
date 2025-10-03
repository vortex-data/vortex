// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::compute::take;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_buffer::Buffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::VortexUnwrap;

fn main() {
    divan::main();
}

const SMALL_ARGS: &[(usize, usize)] = &[
    // (output_size, buffer_utilization_pct)
    (1 << 12, 10),
    (1 << 12, 90),
    (1 << 14, 10),
    (1 << 14, 90),
];

const LARGE_ARGS: &[(usize, usize)] = &[
    // (output_size, buffer_utilization_pct)
    (1 << 19, 10),
    (1 << 19, 90),
];

#[divan::bench(args = SMALL_ARGS)]
fn compact(bencher: Bencher, args: (usize, usize)) {
    compact_impl(bencher, args);
}

#[divan::bench(args = LARGE_ARGS, sample_count = 10)]
fn compact_large(bencher: Bencher, args: (usize, usize)) {
    compact_impl(bencher, args);
}

#[divan::bench(args = SMALL_ARGS)]
fn compact_sliced(bencher: Bencher, args: (usize, usize)) {
    compact_sliced_impl(bencher, args);
}

#[divan::bench(args = LARGE_ARGS, sample_count = 10)]
fn compact_sliced_large(bencher: Bencher, args: (usize, usize)) {
    compact_sliced_impl(bencher, args);
}

fn compact_impl(bencher: Bencher, (output_size, utilization_pct): (usize, usize)) {
    let base_size = (output_size * 100) / utilization_pct;
    let base_array = build_varbinview_fixture(base_size);
    let indices = random_indices(output_size, base_size);

    bencher
        .with_inputs(|| {
            let taken = take(base_array.as_ref(), &indices).vortex_unwrap();
            taken.to_varbinview()
        })
        .bench_values(|array| array.compact_buffers().vortex_unwrap())
}

fn compact_sliced_impl(bencher: Bencher, (output_size, utilization_pct): (usize, usize)) {
    let base_size = (output_size * 100) / utilization_pct;
    let base_array = build_varbinview_fixture(base_size);

    bencher
        .with_inputs(|| {
            let sliced = base_array.as_ref().slice(0..output_size);
            sliced.to_varbinview()
        })
        .bench_values(|array| array.compact_buffers().vortex_unwrap())
}

/// Creates a base VarBinViewArray with mix of inlined and outlined strings.
fn build_varbinview_fixture(len: usize) -> VarBinViewArray {
    let mut builder = VarBinViewBuilder::with_capacity(DType::Utf8(Nullability::NonNullable), len);
    let mut rng = StdRng::seed_from_u64(42);

    for _ in 0..len {
        // Mix of inlined (<=12 bytes) and outlined (>12 bytes) strings
        let str_len = if rng.random_bool(0.5) {
            rng.random_range(5..=12)
        } else {
            rng.random_range(13..=50)
        };

        let s: String = (0..str_len)
            .map(|_| rng.random_range(b'a'..=b'z') as char)
            .collect();

        builder.append_value(s);
    }

    builder.finish_into_varbinview()
}

fn random_indices(count: usize, range: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(123);
    Buffer::from_iter((0..count).map(|_| rng.random_range(0..range) as u64)).into_array()
}
