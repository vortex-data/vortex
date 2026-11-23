// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse arrays against the equivalent Patched arrays over a constant fill value: decoding,
//! comparing against a constant, and summing.

#![expect(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_sparse::initialize(&session);
    session
});

const LEN: usize = 1 << 16;
/// One patch every `STRIDE` rows.
const STRIDES: &[usize] = &[1000, 100, 10];

#[derive(Clone, Copy, Debug)]
enum Encoding {
    Sparse,
    Patched,
}

const ENCODINGS: &[Encoding] = &[Encoding::Sparse, Encoding::Patched];

fn build(encoding: Encoding, stride: usize, values: ArrayRef, fill: Scalar) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let indices: Buffer<u32> = (0..LEN as u32).step_by(stride).collect();
    match encoding {
        Encoding::Sparse => Sparse::try_new(indices.into_array(), values, LEN, fill, &mut ctx)
            .unwrap()
            .into_array(),
        Encoding::Patched => Patched::sparse(indices.into_array(), values, LEN, fill, &mut ctx)
            .unwrap()
            .into_array(),
    }
}

fn n_patches(stride: usize) -> usize {
    LEN.div_ceil(stride)
}

fn ints(encoding: Encoding, stride: usize) -> ArrayRef {
    let values = PrimitiveArray::from_iter((0..n_patches(stride) as i64).map(|i| i * 7));
    build(encoding, stride, values.into_array(), Scalar::from(3i64))
}

fn nullable_ints(encoding: Encoding, stride: usize) -> ArrayRef {
    let values = PrimitiveArray::from_option_iter(
        (0..n_patches(stride) as i64).map(|i| (i % 3 != 0).then_some(i)),
    );
    build(
        encoding,
        stride,
        values.into_array(),
        Scalar::null_native::<i64>(),
    )
}

fn strings(encoding: Encoding, stride: usize) -> ArrayRef {
    let values = VarBinViewArray::from_iter_nullable_str(
        (0..n_patches(stride)).map(|i| Some(format!("value number {i} that is not inlined"))),
    );
    build(
        encoding,
        stride,
        values.into_array(),
        Scalar::null(values_dtype()),
    )
}

fn values_dtype() -> vortex_array::dtype::DType {
    vortex_array::dtype::DType::Utf8(Nullability::Nullable)
}

fn bools(encoding: Encoding, stride: usize) -> ArrayRef {
    let values = BoolArray::from_iter((0..n_patches(stride)).map(|_| true));
    build(
        encoding,
        stride,
        values.into_array(),
        Scalar::bool(false, Nullability::NonNullable),
    )
}

fn bench_canonical(bencher: Bencher, array: ArrayRef) {
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| array.clone().execute::<Canonical>(ctx).unwrap());
}

#[divan::bench(args = ENCODINGS, consts = [1000, 100, 10])]
fn canonicalize_ints<const STRIDE: usize>(bencher: Bencher, encoding: Encoding) {
    bench_canonical(bencher, ints(encoding, STRIDE));
}

#[divan::bench(args = ENCODINGS, consts = [1000, 100, 10])]
fn canonicalize_nullable_ints<const STRIDE: usize>(bencher: Bencher, encoding: Encoding) {
    bench_canonical(bencher, nullable_ints(encoding, STRIDE));
}

#[divan::bench(args = ENCODINGS, consts = [1000, 100, 10])]
fn canonicalize_strings<const STRIDE: usize>(bencher: Bencher, encoding: Encoding) {
    bench_canonical(bencher, strings(encoding, STRIDE));
}

#[divan::bench(args = ENCODINGS, consts = [1000, 100, 10])]
fn canonicalize_bools<const STRIDE: usize>(bencher: Bencher, encoding: Encoding) {
    bench_canonical(bencher, bools(encoding, STRIDE));
}

#[divan::bench(args = ENCODINGS, consts = [1000, 100, 10])]
fn compare_ints<const STRIDE: usize>(bencher: Bencher, encoding: Encoding) {
    let array = ints(encoding, STRIDE);
    bencher
        .with_inputs(|| (array.clone(), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            array
                .binary(
                    ConstantArray::new(14i64, array.len()).into_array(),
                    Operator::Gt,
                )
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = ENCODINGS, consts = [1000, 100, 10])]
fn sum_ints<const STRIDE: usize>(bencher: Bencher, encoding: Encoding) {
    let array = ints(encoding, STRIDE);
    bencher
        // A fresh array each iteration, so that the cached sum statistic is not reused.
        .with_inputs(|| (ints(encoding, STRIDE), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| sum(array, ctx).unwrap());
    drop(array);
}

// Keeps `STRIDES` documented in one place; the benchmarks above repeat it as const generics.
const _: () = assert!(STRIDES.len() == 3);
