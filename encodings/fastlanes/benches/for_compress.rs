// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::FoRArray;

fn main() {
    divan::main();
}

fn make_primitive_nonnull(n: usize) -> PrimitiveArray {
    let data: Buffer<i64> = (0..n as i64).map(|v| v + 1_000_000).collect();
    PrimitiveArray::new(data, Validity::NonNullable)
}

fn make_primitive_nullable(n: usize) -> PrimitiveArray {
    let data: Buffer<i64> = (0..n as i64).map(|v| v + 1_000_000).collect();
    let validity = Validity::from_iter((0..n).map(|i| i % 7 != 0));
    PrimitiveArray::new(data, validity)
}

#[divan::bench(args = [1024, 65_536, 1_048_576])]
fn for_compress_nonnull(bencher: Bencher, n: usize) {
    let array = make_primitive_nonnull(n);
    bencher
        .with_inputs(|| array.clone())
        .bench_values(|arr| FoRArray::encode(arr).unwrap())
}

#[divan::bench(args = [1024, 65_536, 1_048_576])]
fn for_compress_nullable(bencher: Bencher, n: usize) {
    let array = make_primitive_nullable(n);
    bencher
        .with_inputs(|| array.clone())
        .bench_values(|arr| FoRArray::encode(arr).unwrap())
}
