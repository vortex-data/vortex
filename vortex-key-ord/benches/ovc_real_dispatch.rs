// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ovc(root())` through the full Vortex expression-evaluation path.

#![expect(clippy::unwrap_used)]
#![expect(clippy::expect_used)]

use std::hint::black_box;
use std::sync::Once;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_key_ord::ovc_scalarfn::ovc;
use vortex_key_ord::ovc_scalarfn::register_ovc;

fn main() {
    divan::main();
}

const N: usize = 100_000;

static REGISTER: Once = Once::new();
fn ensure_registered() {
    REGISTER.call_once(|| register_ovc(&*LEGACY_SESSION));
}

fn make_constant() -> ArrayRef {
    ConstantArray::new(42u64, N).into_array()
}

fn make_primitive() -> ArrayRef {
    let buf: Vec<u64> = (0..N as u64).collect();
    PrimitiveArray::new(Buffer::<u64>::copy_from(&buf), Validity::NonNullable).into_array()
}

fn make_chunked_constant() -> ArrayRef {
    let chunks: Vec<ArrayRef> = (0..10)
        .map(|_| ConstantArray::new(42u64, N / 10).into_array())
        .collect();
    ChunkedArray::try_new(
        chunks,
        DType::Primitive(PType::U64, Nullability::NonNullable),
    )
    .expect("chunked")
    .into_array()
}

fn bench_apply_execute(bencher: Bencher, input: ArrayRef) {
    ensure_registered();
    let expr = ovc(root());
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let r: ArrayRef = input
                .clone()
                .apply(&expr)
                .expect("apply")
                .execute::<ArrayRef>(&mut ctx)
                .expect("execute");
            black_box(r)
        });
}

#[divan::bench(sample_count = 30)]
fn constant_via_expression(bencher: Bencher) {
    bench_apply_execute(bencher, make_constant());
}

#[divan::bench(sample_count = 30)]
fn primitive_via_expression(bencher: Bencher) {
    bench_apply_execute(bencher, make_primitive());
}

#[divan::bench(sample_count = 30)]
fn chunked_constant_via_expression(bencher: Bencher) {
    bench_apply_execute(bencher, make_chunked_constant());
}

/// `apply_ctx(session)` activates the session-registered reduce_parent
/// kernel for `(Ovc, Chunked)`, which pre-empts the static
/// `ChunkedUnaryScalarFnPushDownRule`. Don't `execute` -- it would
/// canonicalise the encoding-aware Chunked output.
#[divan::bench(sample_count = 30)]
fn chunked_constant_via_apply_ctx(bencher: Bencher) {
    ensure_registered();
    let input = make_chunked_constant();
    let expr = ovc(root());
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| {
            black_box(
                input
                    .clone()
                    .apply_ctx(&expr, &LEGACY_SESSION)
                    .expect("apply_ctx"),
            )
        });
}
