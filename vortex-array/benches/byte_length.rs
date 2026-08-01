// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Baseline throughput for `byte_length` over UTF-8 view arrays.
//!
//! The arms cover inline and out-of-line views, plus nullable out-of-line views. Their names are
//! intended to remain stable across scalar-function implementation changes so CodSpeed can compare
//! them against `develop`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::MaskedArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_array::scalar_fn::fns::byte_length::ByteLength;
use vortex_array::validity::Validity;
use vortex_session::VortexSession;

// Scalar function execution allocates its output inside the timed region, so use the vendored
// allocator instead of measuring glibc differences between CodSpeed runner images.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const SIZES: &[usize] = &[4_096, 65_536];

fn long_strings(len: usize) -> ArrayRef {
    VarBinViewArray::from_iter_str((0..len).map(|i| format!("a string well past inline: {i}")))
        .into_array()
}

fn short_strings(len: usize) -> ArrayRef {
    VarBinViewArray::from_iter_str((0..len).map(|i| format!("{}", i % 1_000))).into_array()
}

fn bench_byte_length(bencher: Bencher, input: ArrayRef) {
    let len = input.len();
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| {
            (
                ScalarFnArray::try_new(
                    TypedScalarFnInstance::new(ByteLength, EmptyOptions).erased(),
                    vec![input.clone()],
                )
                .unwrap()
                .into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mut ctx)| array.execute::<Canonical>(&mut ctx).unwrap());
}

#[divan::bench(args = SIZES)]
fn inline(bencher: Bencher, len: usize) {
    bench_byte_length(bencher, short_strings(len));
}

#[divan::bench(args = SIZES)]
fn out_of_line(bencher: Bencher, len: usize) {
    bench_byte_length(bencher, long_strings(len));
}

#[divan::bench(args = SIZES)]
fn nullable_out_of_line(bencher: Bencher, len: usize) {
    let validity = Validity::from_iter((0..len).map(|i| i % 8 != 0));
    let input = MaskedArray::try_new(long_strings(len), validity)
        .unwrap()
        .into_array();
    bench_byte_length(bencher, input);
}

#[divan::bench(args = SIZES)]
fn nullable_out_of_line_90pct(bencher: Bencher, len: usize) {
    let validity = Validity::from_iter((0..len).map(|i| i.is_multiple_of(10)));
    let input = MaskedArray::try_new(long_strings(len), validity)
        .unwrap()
        .into_array();
    bench_byte_length(bencher, input);
}
